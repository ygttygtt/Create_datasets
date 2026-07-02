"""Quality filtering and deduplication for SFT and DPO samples."""

import difflib
import hashlib
from typing import Any


class QualityFilter:
    """Filters and deduplicates generated samples.

    For SFT: checks instruction + output lengths.
    For DPO: checks instruction + chosen + rejected lengths.
    """

    def __init__(self, config: dict, mode: str = "sft"):
        qc = config["quality"]
        self._mode = mode
        self.min_inst_chars: int = qc.get("min_instruction_chars", 5)
        self.max_inst_chars: int = qc.get("max_instruction_chars", 1024)
        self.min_out_chars: int = qc.get("min_output_chars", 10)
        self.max_out_chars: int = qc.get("max_output_chars", 4096)
        self.enable_dedup: bool = qc.get("enable_dedup", True)
        self.dedup_method: str = qc.get("dedup_method", "exact")
        self.fuzzy_threshold: float = qc.get("fuzzy_threshold", 0.85)

        self._seen_hashes: set[str] = set()
        self._seen_instructions: list[str] = []

    def is_valid(self, sample: dict) -> bool:
        """Check if a sample passes length and content quality checks."""
        instruction = sample.get("instruction", "")

        if len(instruction) < self.min_inst_chars:
            return False
        if len(instruction) > self.max_inst_chars:
            return False

        if self._mode == "dpo":
            chosen = sample.get("chosen", "")
            rejected = sample.get("rejected", "")
            if len(chosen) < self.min_out_chars or len(chosen) > self.max_out_chars:
                return False
            if len(rejected) < 5:  # Rejected must have some content
                return False
        else:
            output = sample.get("output", "")
            if len(output) < self.min_out_chars:
                return False
            if len(output) > self.max_out_chars:
                return False

        return True

    def is_duplicate(self, instruction: str) -> bool:
        """Check if an instruction is a duplicate of previously seen ones."""
        if not self.enable_dedup:
            return False

        text = instruction.strip()

        if self.dedup_method == "exact":
            h = hashlib.md5(text.encode("utf-8")).hexdigest()
            if h in self._seen_hashes:
                return True
            self._seen_hashes.add(h)
            return False

        elif self.dedup_method == "fuzzy":
            for prev in self._seen_instructions:
                ratio = difflib.SequenceMatcher(None, text, prev).ratio()
                if ratio >= self.fuzzy_threshold:
                    return True
            self._seen_instructions.append(text)
            return False

        return False

    def add(self, instruction: str) -> None:
        """Explicitly register an instruction as seen (for checkpoint restore)."""
        if self.dedup_method == "exact":
            h = hashlib.md5(instruction.strip().encode("utf-8")).hexdigest()
            self._seen_hashes.add(h)
        elif self.dedup_method == "fuzzy":
            self._seen_instructions.append(instruction.strip())

    def stats(self) -> dict[str, Any]:
        return {
            "mode": self._mode,
            "dedup_method": self.dedup_method,
            "seen_count": len(self._seen_hashes) if self.dedup_method == "exact"
                          else len(self._seen_instructions),
        }
