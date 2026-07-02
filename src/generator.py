"""Core generation pipeline: orchestrate prompt → API → parse → filter → save.

Supports two modes:
  - "sft": Standard instruction-output pairs
  - "dpo": Preference pairs (chosen vs rejected)
"""

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any

from .config_loader import load_config
from .prompt_manager import PromptManager
from .llm_client import LLMClient
from .rate_limiter import RateLimiter
from .quality import QualityFilter
from .formatter import parse_llm_response, format_sample


class DatasetGenerator:
    """Main pipeline for generating SFT or DPO datasets."""

    def __init__(self, config_path: str | Path):
        self.config = load_config(config_path)
        self.rng = random.Random(self.config["generation"].get("seed", 42))

        # Generation settings (need mode early for QualityFilter)
        gen = self.config["generation"]
        self.mode: str = gen.get("mode", "sft")  # "sft" or "dpo"

        # Components
        self.rate_limiter = RateLimiter(**self.config["rate_limit"])
        self.llm_client = LLMClient(self.config, self.rate_limiter)
        self.quality = QualityFilter(self.config, self.mode)

        # Prompts dir: sibling to config file
        prompts_dir = Path(config_path).parent / "prompts"
        self.prompt_manager = PromptManager(
            prompts_dir,
            self.config.get("templates", {}),
        )

        # Generation settings
        gen = self.config["generation"]
        self.total_samples: int = gen.get("total_samples", 500)
        self.batch_size: int = gen.get("batch_size", 10)
        self.output_dir = Path(gen.get("output_dir", "output"))
        self.output_format: str = gen.get("output_format", "llamafactory")
        self.output_filename: str = gen.get("output_filename", "sft_data")
        self.resume: bool = gen.get("resume", True)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint & output
        self._checkpoint_path = self.output_dir / f"{self.output_filename}.checkpoint"
        self._output_path = self.output_dir / f"{self.output_filename}.jsonl"
        self._failure_path = self.output_dir / f"{self.output_filename}_failures.jsonl"

        # Stats
        self.samples_generated: int = 0
        self.samples_filtered: int = 0
        self.samples_duplicate: int = 0
        self.samples_failed: int = 0

    def _load_checkpoint(self) -> int:
        """Load checkpoint: how many samples already written to output."""
        if not self.resume or not self._checkpoint_path.exists():
            return 0

        try:
            with open(self._checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            count = data.get("samples_generated", 0)

            # Re-register existing for dedup
            if self._output_path.exists() and count > 0:
                with open(self._output_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                sample = json.loads(line)
                                inst = sample.get("instruction", "")
                                if inst:
                                    self.quality.add(inst)
                            except json.JSONDecodeError:
                                pass

            print(f"[Resume] Loaded checkpoint: {count} samples already generated.")
            return count
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}, starting fresh.")
            return 0

    def _save_checkpoint(self) -> None:
        """Save current progress to checkpoint file."""
        data = {
            "samples_generated": self.samples_generated,
            "samples_filtered": self.samples_filtered,
            "samples_duplicate": self.samples_duplicate,
            "samples_failed": self.samples_failed,
            "timestamp": time.time(),
        }
        with open(self._checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def _append_sample(self, sample: dict) -> None:
        """Append a single sample to the output JSONL file."""
        with open(self._output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def _save_failure(self, raw_response: str, user_prompt: str, system_prompt: str, reason: str) -> None:
        """Save a failed generation attempt for later analysis."""
        failure = {
            "raw_response": raw_response,
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "reason": reason,
        }
        with open(self._failure_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(failure, ensure_ascii=False) + "\n")

    async def _generate_one(self, dry_run: bool = False) -> dict | None:
        """Generate a single sample end-to-end. Returns formatted dict or None."""
        template_name, system, user = self.prompt_manager.generate_prompt(self.rng)

        if dry_run:
            if self.mode == "dpo":
                return {
                    "system": system,
                    "instruction": f"[DRY-RUN] {user[:80]}...",
                    "input": "",
                    "chosen": f"[DRY-RUN] Good answer for '{template_name}'",
                    "rejected": "[DRY-RUN] 作为AI助手我无法回答这个问题。",
                }
            return {
                "instruction": f"[DRY-RUN] {user[:80]}...",
                "input": "",
                "output": f"[DRY-RUN] Simulated output for template '{template_name}'",
                "system": system,
                "history": [],
            }

        raw = await self.llm_client.generate(user, system)
        if raw is None:
            self.samples_failed += 1
            return None

        parsed = parse_llm_response(raw)
        if parsed is None or not isinstance(parsed, dict):
            self._save_failure(raw or "", user, system, "json_parse_failed")
            self.samples_failed += 1
            return None

        fmt = self.output_format if self.mode == "sft" else "dpo"
        formatted = format_sample(parsed, fmt, system)

        # Quality check (DPO checks chosen/rejected; SFT checks output)
        if not self.quality.is_valid(formatted):
            self.samples_filtered += 1
            return None

        # Dedup on instruction
        instruction = formatted.get("instruction", "")
        if self.quality.is_duplicate(instruction):
            self.samples_duplicate += 1
            return None

        return formatted

    async def generate(self, dry_run: bool = False) -> dict[str, Any]:
        """Run the full generation pipeline."""
        start_time = time.time()
        already_done = self._load_checkpoint()
        self.samples_generated = already_done

        target = self.total_samples
        if already_done >= target:
            print(f"Already have {already_done}/{target} samples. Nothing to do.")
            return self._make_report(start_time)

        remaining = target - already_done
        mode_label = f"[{self.mode.upper()}]"
        print(f"\n{'[DRY-RUN] ' if dry_run else ''}{mode_label} "
              f"Generating {remaining} samples (target: {target})...")
        print(f"  Model: {self.llm_client.model_name}")
        print(f"  Concurrency: {self.config['rate_limit']['max_concurrent']} max, "
              f"{self.config['rate_limit']['delay_between_calls']}s delay")
        print(f"  Output: {self._output_path}")

        try:
            from tqdm import tqdm
            pbar = tqdm(total=remaining, desc=f"Generating [{self.mode}]", unit="samples")
        except ImportError:
            pbar = None

        max_concurrent = self.config["rate_limit"]["max_concurrent"]
        total_attempts = 0
        safety_limit = remaining * 50  # generous allowance for low-success-rate pools

        # Task-pool pattern: keep max_concurrent requests in flight at all times.
        # When one completes, immediately launch another — no waiting for batches.
        pending: set[asyncio.Task] = set()
        flush_batch: list[dict] = []

        def _launch():
            return asyncio.create_task(self._generate_one(dry_run))

        # Prime the pipeline
        for _ in range(min(max_concurrent, safety_limit)):
            pending.add(_launch())

        while pending and self.samples_generated < target and total_attempts < safety_limit:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                total_attempts += 1
                result = task.result()

                if result is not None:
                    flush_batch.append(result)
                    self.samples_generated += 1
                    if pbar:
                        pbar.update(1)

                    if len(flush_batch) >= self.batch_size:
                        for s in flush_batch:
                            self._append_sample(s)
                        self._save_checkpoint()
                        flush_batch.clear()

                # Refill pipeline — launch a new task for every completed one
                if self.samples_generated < target and total_attempts < safety_limit:
                    pending.add(_launch())

                if self.samples_generated >= target:
                    break

            # Cancel remaining pending tasks if target reached
            if self.samples_generated >= target:
                for t in pending:
                    t.cancel()
                break

        # Final flush
        if flush_batch:
            for s in flush_batch:
                self._append_sample(s)
            self._save_checkpoint()

        if pbar:
            pbar.close()

        return self._make_report(start_time)

    def _make_report(self, start_time: float) -> dict[str, Any]:
        elapsed = time.time() - start_time
        return {
            "mode": self.mode,
            "samples_generated": self.samples_generated,
            "samples_filtered": self.samples_filtered,
            "samples_duplicate": self.samples_duplicate,
            "samples_failed": self.samples_failed,
            "elapsed_seconds": round(elapsed, 1),
            "llm_calls": self.llm_client.total_calls,
            "llm_tokens": self.llm_client.total_tokens,
            "output_file": str(self._output_path),
        }
