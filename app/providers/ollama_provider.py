from __future__ import annotations

import json
import os
import urllib.request

from app.models.registry import ModelConfig
from app.providers.base import DEFAULT_MAX_TOKENS, RawResult


class OllamaAdapter:
    def _call(
        self, prompt: str, model_config: ModelConfig, system: str | None = None, max_tokens: int | None = None
    ) -> RawResult:
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        payload = {
            "model": model_config.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens or DEFAULT_MAX_TOKENS},
        }
        if system:
            payload["system"] = system
        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return RawResult(
            text=data.get("response", ""),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )
