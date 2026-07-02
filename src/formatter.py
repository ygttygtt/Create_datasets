"""Format conversion + robust response parsing (JSON + XML fallback)."""

import json
import re
from typing import Any


def parse_llm_response(raw: str) -> dict | None:
    """Extract structured data from LLM response, with robust fallback.

    Tries: 1) XML tags  2) JSON parse  3) Broken JSON regex salvage
    """
    if not raw:
        return None

    text = raw.strip()

    # === 1) XML format (most robust for content with quotes/newlines) ===
    inst = _xml_extract(text, "instruction")
    out = _xml_extract(text, "output")
    if inst and out:
        inp = _xml_extract(text, "input") or ""
        return {"instruction": inst, "input": inp, "output": out}

    # Check for conversations XML
    user_msgs = re.findall(r'<user>(.*?)</user>', text, re.DOTALL)
    asst_msgs = re.findall(r'<assistant>(.*?)</assistant>', text, re.DOTALL)
    if user_msgs and asst_msgs:
        convs = []
        for u, a in zip(user_msgs, asst_msgs):
            convs.append({"role": "user", "content": u.strip()})
            convs.append({"role": "assistant", "content": a.strip()})
        return {"conversations": convs}

    # === 2) Direct JSON ===
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # === 3) JSON in markdown fence ===
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1).strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # === 4) Salvage: regex extract from broken JSON ===
    inst = _json_extract(text, "instruction")
    out = _json_extract(text, "output")
    if inst and out:
        inp = _json_extract(text, "input") or ""
        return {"instruction": inst, "input": inp, "output": out}

    # Also try conversations
    convs = _extract_conversations_json(text)
    if convs:
        return {"conversations": convs}

    return None


def _xml_extract(text: str, tag: str) -> str | None:
    """Extract content from <tag>...</tag>, handles multiline."""
    pattern = rf'<{tag}>\s*(.*?)\s*</{tag}>'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def _json_extract(text: str, key: str) -> str | None:
    """Extract value for a JSON key, even from broken JSON.
    Handles unescaped quotes inside values by matching to the final '", ' or '"}'.
    """
    # Pattern: "key": "value" where value may contain internal quotes
    # We match from the key to the next '", "' or '"}'
    pattern = rf'"{key}"\s*:\s*"(.+?)"\s*[,}}]'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def _extract_conversations_json(text: str) -> list[dict] | None:
    """Extract conversation turns from broken JSON."""
    roles = re.findall(r'"role"\s*:\s*"(\w+)"', text)
    contents = re.findall(r'"content"\s*:\s*"(.+?)"\s*[,}]', text, re.DOTALL)
    if roles and contents and len(roles) == len(contents):
        convs = []
        for r, c in zip(roles, contents):
            convs.append({"role": r, "content": c.strip()})
        return convs
    return None


# ============================================================
# Format converters
# ============================================================

def to_llamafactory(parsed: dict, system_prompt: str = "") -> dict:
    """Convert parsed LLM response to LLaMAFactory (alpaca-zh) format."""
    if "conversations" in parsed:
        convs = parsed["conversations"]
        instruction = ""
        output = ""
        history: list[list[str]] = []

        user_turns = [c for c in convs if c.get("role") == "user"]
        assistant_turns = [c for c in convs if c.get("role") == "assistant"]

        if user_turns and assistant_turns:
            instruction = user_turns[0].get("content", "")
            output = assistant_turns[-1].get("content", "")
            for i in range(1, min(len(user_turns), len(assistant_turns))):
                history.append([
                    user_turns[i].get("content", ""),
                    assistant_turns[i - 1].get("content", "") if i > 0 else "",
                ])

        return {
            "instruction": instruction,
            "input": "",
            "output": output,
            "system": system_prompt,
            "history": history,
        }

    return {
        "instruction": parsed.get("instruction", ""),
        "input": parsed.get("input", ""),
        "output": parsed.get("output", ""),
        "system": system_prompt,
        "history": parsed.get("history", []),
    }


def to_sharegpt(parsed: dict, system_prompt: str = "") -> dict:
    """Convert parsed LLM response to ShareGPT format."""
    if "conversations" in parsed:
        role_map = {"user": "human", "assistant": "gpt", "human": "human", "gpt": "gpt"}
        convs = []
        for c in parsed["conversations"]:
            convs.append({
                "from": role_map.get(c.get("role", ""), "human"),
                "value": c.get("content", ""),
            })
        result: dict[str, Any] = {"conversations": convs}
        if system_prompt:
            result["system"] = system_prompt
        return result

    convs = [{"from": "human", "value": parsed.get("instruction", "")}]
    user_input = parsed.get("input", "")
    if user_input:
        convs[0]["value"] = convs[0]["value"] + "\n\n" + user_input
    convs.append({"from": "gpt", "value": parsed.get("output", "")})
    result = {"conversations": convs}
    if system_prompt:
        result["system"] = system_prompt
    return result


def to_dpo_format(parsed: dict, system_prompt: str = "") -> dict:
    """Convert parsed LLM response to DPO pair format."""
    return {
        "system": system_prompt,
        "instruction": parsed.get("instruction", ""),
        "input": parsed.get("input", ""),
        "chosen": parsed.get("chosen", ""),
        "rejected": parsed.get("rejected", ""),
    }


def format_sample(parsed: dict, output_format: str, system_prompt: str = "") -> dict:
    """Convert parsed LLM output to the desired dataset format."""
    if output_format == "llamafactory":
        return to_llamafactory(parsed, system_prompt)
    elif output_format == "sharegpt":
        return to_sharegpt(parsed, system_prompt)
    elif output_format == "dpo":
        return to_dpo_format(parsed, system_prompt)
    else:
        raise ValueError(f"Unknown output format: {output_format}")
