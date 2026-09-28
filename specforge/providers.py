"""Official OpenAI and Anthropic provider clients used by SpecForge."""

from __future__ import annotations

import os


class OpenAIProvider:
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int | None = None):
        from openai import OpenAI

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Set OPENAI_API_KEY before running an OpenAI experiment.")
        self.client = OpenAI(api_key=key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, system: str, prompt: str) -> str:
        options = dict(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        if self.max_tokens is not None:
            options["max_tokens"] = self.max_tokens
        response = self.client.chat.completions.create(**options)
        return response.choices[0].message.content or ""


class AnthropicProvider:
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 2048):
        from anthropic import Anthropic

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Set ANTHROPIC_API_KEY before running an Anthropic experiment.")
        self.client = Anthropic(api_key=key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, system: str, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return "".join(block.text for block in response.content if block.type == "text")


def make_provider(config: dict):
    provider = str(config.get("provider", "")).lower()
    model = config.get("model")
    if not model:
        raise ValueError("The model field is required.")
    temperature = float(config.get("temperature", 0.0))
    if provider == "openai":
        tokens = config.get("max_tokens")
        return OpenAIProvider(model, temperature, int(tokens) if tokens is not None else None)
    if provider == "anthropic":
        return AnthropicProvider(model, temperature, int(config.get("max_tokens", 2048)))
    raise ValueError("provider must be 'openai' or 'anthropic'")
