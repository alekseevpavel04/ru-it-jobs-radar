"""The student's prompt for a pasted text, rendered by the training code itself.

radar.train.dataset (vendored into the image) cuts the input to max_input_tokens at a token boundary
and wraps it in the system prompt and the chat template with Qwen3 thinking off, with the tokenizer
the adapter was trained with. The quantized student was scored on prompts rendered this way.

    python -m app.prompt check INPUTS.jsonl PROMPTS.jsonl

compares the rendering with prompts rendered by radar.train.quantize (models/merged/<name>/prompts)
and prints how many match.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import ASSETS

from radar.train.dataset import cap_input, render_prompt, token_spans


@dataclass(frozen=True)
class Rendered:
    prompt: str
    prompt_tokens: int
    truncated: bool


class Prompter:
    def __init__(self, tokenizer_dir: Path = ASSETS / "tokenizer") -> None:
        from transformers import AutoTokenizer

        self.meta: dict[str, Any] = json.loads((tokenizer_dir / "dataset_meta.json").read_text(encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

    def render(self, text: str) -> Rendered:
        user, cut = cap_input(text, token_spans(self.tokenizer, text), self.meta["max_input_tokens"])
        prompt = render_prompt(self.tokenizer, self.meta["system_prompt"], user)
        n = len(self.tokenizer(prompt, add_special_tokens=False).input_ids)
        return Rendered(prompt, n, cut)


def check(inputs: Path, prompts: Path) -> dict[str, int]:
    rows = [json.loads(line) for line in inputs.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = {
        r["id"]: r for r in (json.loads(line) for line in prompts.read_text(encoding="utf-8").splitlines() if line)
    }
    prompter = Prompter()
    same_prompt = same_tokens = 0
    for row in rows:
        got = prompter.render(row["input"])
        want = expected[row["id"]]
        same_prompt += got.prompt == want["prompt"]
        same_tokens += got.prompt_tokens == want["prompt_tokens"]
    return {"records": len(rows), "prompt_match": same_prompt, "prompt_tokens_match": same_tokens}


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "check":
        sys.exit("usage: python -m app.prompt check INPUTS.jsonl PROMPTS.jsonl")
    print(json.dumps(check(Path(sys.argv[2]), Path(sys.argv[3]))))
