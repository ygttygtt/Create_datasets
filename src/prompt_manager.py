"""Prompt template loading, seed sampling, and prompt rendering."""

import random
from pathlib import Path
from typing import Any

import yaml


class PromptTemplate:
    """A single prompt template with seed variables."""

    def __init__(self, data: dict):
        self.name: str = data["name"]
        self.description: str = data.get("description", "")
        self.system_prompt: str = data.get("system_prompt", "").strip()
        self.user_template: str = data["user_template"]
        self.seeds: dict[str, list] = data.get("seeds", {})

    def sample_variables(self, rng: random.Random) -> dict[str, str]:
        """Randomly sample one value from each seed category."""
        chosen = {}
        for var_name, options in self.seeds.items():
            if options:
                chosen[var_name] = rng.choice(options)
        return chosen

    def render(self, variables: dict[str, str]) -> tuple[str, str]:
        """Render system and user prompts with the given variables.

        Returns (system_prompt, user_content).
        """
        system = self.system_prompt
        user = self.user_template

        # Replace single-brace variables: {var}
        for key, val in variables.items():
            placeholder = "{" + key + "}"
            system = system.replace(placeholder, str(val))
            user = user.replace(placeholder, str(val))

        return system, user


class PromptManager:
    """Manages multiple prompt templates and weighted sampling."""

    def __init__(self, prompts_dir: str | Path, template_weights: dict[str, int] | None = None):
        self.prompts_dir = Path(prompts_dir)
        self.templates: dict[str, PromptTemplate] = {}
        self.weights: dict[str, int] = {}
        self._load_all()
        self._set_weights(template_weights or {})

    def _load_all(self) -> None:
        """Load all YAML prompt templates from the prompts directory."""
        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")

        for yaml_file in sorted(self.prompts_dir.glob("*.yaml")):
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            template = PromptTemplate(data)
            self.templates[template.name] = template
            self.weights[template.name] = self.weights.get(template.name, 10)

        if not self.templates:
            raise ValueError(f"No .yaml templates found in {self.prompts_dir}")

    def _set_weights(self, weight_map: dict[str, int]) -> None:
        """Override default weights from config.

        If weight_map is provided (non-empty), templates NOT listed get weight 0.
        If weight_map is empty, all templates keep their default weight (10).
        """
        if weight_map:
            # Zero out all weights first — only explicitly listed templates get weight
            for name in self.weights:
                self.weights[name] = 0
            for name, weight in weight_map.items():
                if name in self.templates:
                    self.weights[name] = weight

    def sample_template(self, rng: random.Random) -> PromptTemplate:
        """Weighted random selection of a template."""
        names = list(self.templates.keys())
        w = [max(self.weights[n], 0) for n in names]
        if sum(w) == 0:
            # All zero weight → uniform
            w = [1] * len(names)
        chosen = rng.choices(names, weights=w, k=1)[0]
        return self.templates[chosen]

    def generate_prompt(self, rng: random.Random) -> tuple[str, str, str]:
        """Sample a template + variables, render the prompt.

        Returns (template_name, system_prompt, user_content).
        """
        template = self.sample_template(rng)
        variables = template.sample_variables(rng)
        system, user = template.render(variables)
        return template.name, system, user
