#!/usr/bin/env python3
"""
Unified dataset generation script — pluggable API, model, and pool configs.

Usage:
  python generate.py                        # Run default pool (nsfw_grok)
  python generate.py --config configs/nsfw_grok.yaml
  python generate.py --config configs/nsfw_grok.yaml --config configs/insult_grok.yaml
  python generate.py --list                 # Show available configs
  python generate.py --dry-run              # Test pipeline without API calls
  python generate.py --fresh                # Ignore checkpoint, start from scratch
  python generate.py --target 500           # Override target sample count

Config files live in configs/*.yaml. Each file specifies:
  - API endpoint, key, model
  - Concurrency & rate limits
  - Output format & file
  - Jailbreak settings (optional)
  - Prompt templates (inline)
"""

import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

# ============================================================
# Config loader
# ============================================================

SUPPORTED_FORMATS = {
    "llamafactory": "instruction/input/output/system/history",
    "sharegpt": "conversations[{from,value}]/system",
}


def load_config(path: str) -> dict:
    """Load a pool config YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Validate required fields
    for section in ["api", "generation", "templates"]:
        if section not in cfg:
            raise ValueError(f"Config missing required section: [{section}]")

    api = cfg["api"]
    for field in ["base_url", "key", "model"]:
        if field not in api:
            raise ValueError(f"api.{field} is required")

    gen = cfg["generation"]
    gen.setdefault("mode", "sft")
    gen.setdefault("format", "llamafactory")
    gen.setdefault("max_concurrent", 10)
    gen.setdefault("max_tokens", 2048)
    gen.setdefault("temperature", 0.9)
    gen.setdefault("batch_size", 10)
    gen.setdefault("output_dir", "output")
    gen.setdefault("resume", True)

    cfg.setdefault("jailbreak", {"enabled": False})

    return cfg


# ============================================================
# Response parser (XML + JSON, robust)
# ============================================================

def parse_response(raw: str) -> dict | None:
    """Extract structured data from LLM response. XML first, then JSON."""
    if not raw:
        return None

    # XML instruction format
    inst = _xml_tag(raw, "instruction")
    out = _xml_tag(raw, "output")
    if inst and out:
        inp = _xml_tag(raw, "input") or ""
        return {"instruction": inst, "input": inp, "output": out}

    # XML conversation format
    users = re.findall(r"<user>(.*?)</user>", raw, re.DOTALL)
    assts = re.findall(r"<assistant>(.*?)</assistant>", raw, re.DOTALL)
    if users and assts:
        convs = []
        for u, a in zip(users, assts):
            convs.append({"role": "user", "content": u.strip()})
            convs.append({"role": "assistant", "content": a.strip()})
        return {"conversations": convs}

    # Direct JSON
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # JSON in markdown fence
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1).strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return None


def _xml_tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


# ============================================================
# Format converters
# ============================================================

def conv_to_llamafactory(parsed: dict, system: str = "") -> dict:
    """Convert parsed data to LLaMAFactory alpaca-zh format."""
    if "conversations" in parsed:
        convs = parsed["conversations"]
        users = [c for c in convs if c.get("role") == "user"]
        assts = [c for c in convs if c.get("role") == "assistant"]
        inst = users[0].get("content", "") if users else ""
        out = assts[-1].get("content", "") if assts else ""
        history = []
        for i in range(1, min(len(users), len(assts))):
            history.append([users[i].get("content", ""),
                            assts[i-1].get("content", "") if i > 0 else ""])
        return {"instruction": inst, "input": "", "output": out,
                "system": system, "history": history}

    return {"instruction": parsed.get("instruction", ""),
            "input": parsed.get("input", ""),
            "output": parsed.get("output", ""),
            "system": system,
            "history": parsed.get("history", [])}


def conv_to_sharegpt(parsed: dict, system: str = "") -> dict:
    """Convert to ShareGPT format."""
    if "conversations" in parsed:
        role_map = {"user": "human", "assistant": "gpt", "human": "human", "gpt": "gpt"}
        convs = [{"from": role_map.get(c.get("role", ""), "human"),
                  "value": c.get("content", "")} for c in parsed["conversations"]]
        result = {"conversations": convs}
        if system:
            result["system"] = system
        return result
    convs = [{"from": "human", "value": parsed.get("instruction", "")}]
    inp = parsed.get("input", "")
    if inp:
        convs[0]["value"] += "\n\n" + inp
    convs.append({"from": "gpt", "value": parsed.get("output", "")})
    result = {"conversations": convs}
    if system:
        result["system"] = system
    return result


# ============================================================
# Quality checks
# ============================================================

def quality_check(sample: dict, mode: str = "sft") -> bool:
    """Basic quality filter."""
    inst = sample.get("instruction", "")
    if len(inst) < 3 or len(inst) > 2000:
        return False

    if mode == "dpo":
        chosen = sample.get("chosen", "")
        rejected = sample.get("rejected", "")
        if len(chosen) < 20 or len(rejected) < 5:
            return False
    else:
        out = sample.get("output", "")
        if len(out) < 20:
            return False

    return True


# ============================================================
# Prompt builder
# ============================================================

class PromptBuilder:
    """Loads templates from config, renders with random seeds."""

    def __init__(self, templates_cfg: list[dict], rng: random.Random):
        self.templates = templates_cfg
        self.rng = rng

        # Validate & set defaults
        for t in self.templates:
            t.setdefault("weight", 10)
            t.setdefault("system", "")

        total = sum(t["weight"] for t in self.templates)
        if total == 0:
            for t in self.templates:
                t["weight"] = 1

    def build(self) -> dict:
        """Sample template + seeds, render prompt. Returns {name, system, user}."""
        # Weighted selection
        total = sum(t["weight"] for t in self.templates)
        pick = self.rng.uniform(0, total)
        acc = 0
        tmpl = self.templates[-1]
        for t in self.templates:
            acc += t["weight"]
            if pick <= acc:
                tmpl = t
                break

        # Sample seeds
        vars_ = {}
        seeds = tmpl.get("seeds", {})
        for key, options in seeds.items():
            if options:
                vars_[key] = self.rng.choice(options)

        # Render
        system = tmpl.get("system", "")
        user = tmpl.get("user", "")
        for k, v in vars_.items():
            placeholder = "{" + k + "}"
            system = system.replace(placeholder, str(v))
            user = user.replace(placeholder, str(v))

        return {"name": tmpl.get("name", "unknown"),
                "system": system, "user": user}


# ============================================================
# Generator
# ============================================================

class PoolGenerator:
    """Generates a single data pool from a config."""

    def __init__(self, config: dict, config_path: str = ""):
        self.cfg = config
        self.config_dir = Path(config_path).parent if config_path else Path(".")

        api = config["api"]
        gen = config["generation"]
        jb = config.get("jailbreak", {})

        self.api_url = api["base_url"].rstrip("/") + "/chat/completions"
        self.api_key = api["key"]
        self.model = api["model"]
        self.max_tokens = gen.get("max_tokens", 2048)
        self.temperature = gen.get("temperature", 0.9)
        self.max_concurrent = gen.get("max_concurrent", 10)
        self.batch_size = gen.get("batch_size", 10)
        self.mode = gen.get("mode", "sft")
        self.output_format = gen.get("format", "llamafactory")
        self.output_dir = Path(gen.get("output_dir", "output"))
        self.output_name = gen.get("output_name", config.get("name", "data"))
        self.resume = gen.get("resume", True)
        self.seed = gen.get("seed", 42)

        # Jailbreak
        self.jb_enabled = jb.get("enabled", False)
        self.jb_system = jb.get("system_prompt", "").strip()
        self.jb_fake_reply = jb.get("fake_assistant_reply", "")

        # Paths
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.out_path = self.output_dir / f"{self.output_name}.jsonl"
        self.fail_path = self.output_dir / f"{self.output_name}_failures.jsonl"
        self.ck_path = self.output_dir / f"{self.output_name}.checkpoint"

        # Prompts
        self.rng = random.Random(self.seed)
        self.prompts = PromptBuilder(config["templates"], self.rng)

        # Stats (reset on run)
        self.done = 0
        self.failed = 0
        self.start_time = 0.0
        self.seen = set()

    def _build_messages(self, system: str, user: str) -> list[dict]:
        msgs = []
        if self.jb_enabled:
            if self.jb_system:
                msgs.append({"role": "system", "content": self.jb_system})
            if self.jb_fake_reply:
                msgs.append({"role": "assistant", "content": self.jb_fake_reply})
        elif system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        return msgs

    def load_checkpoint(self) -> int:
        if not self.resume or not self.ck_path.exists():
            return 0
        try:
            ck = json.loads(self.ck_path.read_text(encoding="utf-8"))
            done = ck.get("done", 0)
            if self.out_path.exists():
                with open(self.out_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            d = json.loads(line)
                            inst = d.get("instruction", "").strip()
                            if inst:
                                self.seen.add(inst)
                        except json.JSONDecodeError:
                            pass
            return done
        except Exception:
            return 0

    def save_checkpoint(self):
        self.ck_path.write_text(
            json.dumps({"done": self.done, "failed": self.failed, "time": time.time()},
                       ensure_ascii=False), encoding="utf-8")

    async def _make_request(self, messages: list[dict],
                            client: httpx.AsyncClient) -> tuple[str | None, str | None]:
        """Make one API call. Returns (content, error_text)."""
        try:
            resp = await client.post(self.api_url, headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }, json={
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": False,
            })
        except Exception as e:
            return None, str(e)

        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}: {resp.text[:200]}"

        data = resp.json()
        return data["choices"][0]["message"]["content"], None

    async def _generate_one(self, client: httpx.AsyncClient,
                            sem: asyncio.Semaphore) -> dict | None:
        """Full pipeline: prompt → API → parse → format → quality."""
        prompt = self.prompts.build()
        messages = self._build_messages(prompt["system"], prompt["user"])

        async with sem:
            content, error = await self._make_request(messages, client)

        if error:
            self._save_failure(f"API_ERROR: {error}", prompt["user"], prompt["system"])
            return None

        parsed = parse_response(content)
        if parsed is None:
            self._save_failure(content, prompt["user"], prompt["system"])
            return None

        # Format
        if self.output_format == "llamafactory":
            sample = conv_to_llamafactory(parsed, prompt["system"])
        elif self.output_format == "sharegpt":
            sample = conv_to_sharegpt(parsed, prompt["system"])
        else:
            sample = conv_to_llamafactory(parsed, prompt["system"])

        # Quality
        if not quality_check(sample, self.mode):
            self._save_failure(content, prompt["user"], prompt["system"])
            return None

        # Dedup
        inst = sample.get("instruction", "").strip()
        if inst in self.seen:
            return None  # duplicate, skip silently
        self.seen.add(inst)

        return sample

    def _save_failure(self, raw: str, user: str, system: str):
        """Append a failure entry for later analysis."""
        entry = {"raw_response": raw, "user_prompt": user,
                 "system_prompt": system, "timestamp": time.time()}
        with open(self.fail_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.failed += 1

    async def run(self, target: int, dry_run: bool = False,
                  fresh: bool = False) -> dict:
        """Main generation loop."""
        if fresh:
            self.ck_path.unlink(missing_ok=True)

        already = self.load_checkpoint()
        self.done = already
        remaining = target - already

        if remaining <= 0:
            print(f"[{self.output_name}] Already at target ({already}/{target}).")
            return {"done": already, "failed": self.failed, "time": 0}

        print(f"\n{'='*55}")
        print(f"  Pool: {self.output_name}")
        print(f"  Model: {self.model}  |  Concurrency: {self.max_concurrent}")
        print(f"  Target: {target}  |  Remaining: {remaining}")
        print(f"  Output: {self.out_path}")
        if self.jb_enabled:
            print(f"  Jailbreak: ON")
        print(f"{'='*55}\n")

        self.start_time = time.time()
        sem = asyncio.Semaphore(self.max_concurrent)
        batch: list[dict] = []
        total_attempts = 0

        async def worker():
            nonlocal total_attempts
            if dry_run:
                p = self.prompts.build()
                return {"instruction": f"[DRY-RUN] {p['user'][:80]}...",
                        "input": "", "output": f"[DRY-RUN] {p['name']}",
                        "system": p["system"], "history": []}
            total_attempts += 1
            return await self._generate_one(client, sem)

        async with httpx.AsyncClient(timeout=180.0) as client:
            pending: set[asyncio.Task] = set()

            # Prime pipeline
            for _ in range(min(self.max_concurrent, remaining * 2)):
                pending.add(asyncio.create_task(worker()))

            while pending and self.done < target:
                done_tasks, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED)

                for task in done_tasks:
                    try:
                        result = task.result()
                    except Exception:
                        result = None

                    if result is not None:
                        batch.append(result)
                        self.done += 1

                        if len(batch) >= self.batch_size:
                            with open(self.out_path, "a", encoding="utf-8") as f:
                                for s in batch:
                                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
                            self.save_checkpoint()
                            batch.clear()

                    # Refill
                    if self.done < target:
                        pending.add(asyncio.create_task(worker()))

                    # Progress update
                    self._print_progress(target)

                    if self.done >= target:
                        for t in pending:
                            t.cancel()
                        break

        # Final flush
        if batch:
            with open(self.out_path, "a", encoding="utf-8") as f:
                for s in batch:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            self.save_checkpoint()

        elapsed = time.time() - self.start_time
        print(f"\n  Done: {self.done} samples | Failed: {self.failed} | "
              f"Time: {elapsed/60:.1f}min | Rate: {(self.done-already)/max(elapsed,1):.1f}/s\n")

        return {"done": self.done, "failed": self.failed, "time": elapsed,
                "rate": (self.done - already) / max(elapsed, 1)}

    def _print_progress(self, target: int):
        """Print progress line (overwrites previous)."""
        elapsed = max(time.time() - self.start_time, 0.1)
        rate = (self.done - self.load_checkpoint()) / elapsed if hasattr(self, '_last_checkpoint') else 0
        eta = (target - self.done) / max(rate, 0.01)
        pct = 100 * self.done / target
        # Simple inline progress
        bar_len = 20
        filled = int(bar_len * self.done / target)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r  [{bar}] {self.done}/{target} ({pct:.0f}%) | "
              f"fail:{self.failed} | ETA:{eta/60:.0f}m{eta%60:.0f}s   ",
              end="", flush=True)


# ============================================================
# CLI
# ============================================================

def list_configs():
    """List available config files in configs/ directory."""
    configs_dir = Path("configs")
    if not configs_dir.exists():
        print("No configs/ directory found.")
        print("Create configs/*.yaml files to define generation pools.")
        return

    for f in sorted(configs_dir.glob("*.yaml")):
        try:
            cfg = load_config(str(f))
            name = cfg.get("name", f.stem)
            api = cfg.get("api", {})
            gen = cfg.get("generation", {})
            print(f"  {f.stem:30s} | {api.get('model','?'):20s} | "
                  f"{gen.get('mode','sft'):4s} | target: {gen.get('target','?')}")
        except Exception as e:
            print(f"  {f.stem:30s} | ERROR: {e}")


async def main():
    parser = argparse.ArgumentParser(
        description="Unified LLM dataset generator — pluggable multi-model support")
    parser.add_argument("--config", "-c", action="append", default=[],
                        help="Config file(s) to run (repeatable)")
    parser.add_argument("--list", action="store_true", help="List available configs")
    parser.add_argument("--target", "-t", type=int, default=None,
                        help="Override target sample count")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test pipeline without API calls")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore checkpoint, start from scratch")
    args = parser.parse_args()

    if args.list:
        list_configs()
        return

    # Default config if none specified
    configs = args.config if args.config else ["configs/nsfw_grok.yaml"]

    for config_path in configs:
        if not Path(config_path).exists():
            print(f"ERROR: Config not found: {config_path}")
            print("Use --list to see available configs.")
            continue

        try:
            cfg = load_config(config_path)
        except Exception as e:
            print(f"ERROR loading {config_path}: {e}")
            continue

        gen = PoolGenerator(cfg, config_path)
        target = args.target or cfg["generation"].get("target", 1000)

        result = await gen.run(target=target, dry_run=args.dry_run, fresh=args.fresh)
        print(f"  Final: {result['done']} samples, "
              f"{result['failed']} failed, "
              f"{result['time']/60:.1f} min")


if __name__ == "__main__":
    asyncio.run(main())
