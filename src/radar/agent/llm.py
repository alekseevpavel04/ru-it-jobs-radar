"""Chat completions against the local OpenAI-compatible server at LLM_BASE_URL (llama-server).

Only a local model: no key, no paid API. Qwen3 runs with thinking off, temperature 0, so a run of
the control set is repeatable on the same server and model file.
"""

from typing import Protocol

import httpx

from radar.config import require

MODEL = "Qwen3-8B-Q4_K_M"


class Chat(Protocol):
    def __call__(self, system: str, user: str, max_tokens: int) -> str: ...


class LLMError(RuntimeError):
    pass


class LocalChat:
    def __init__(self, model: str = MODEL, timeout_s: float = 300.0) -> None:
        self.model = model
        self.client = httpx.Client(base_url=require("LLM_BASE_URL"), timeout=httpx.Timeout(timeout_s, connect=10.0))

    def check_model(self) -> None:
        served = [m["id"] for m in self.client.get("models").json().get("data", [])]
        if self.model not in served:
            raise LLMError(f"the server at LLM_BASE_URL serves {served}, not {self.model}")

    def __call__(self, system: str, user: str, max_tokens: int) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = self.client.post("chat/completions", json=body)
        if response.status_code != 200:
            raise LLMError(f"LLM server answered {response.status_code}: {response.text[:300]}")
        content: str = response.json()["choices"][0]["message"]["content"] or ""
        return content

    def close(self) -> None:
        self.client.close()
