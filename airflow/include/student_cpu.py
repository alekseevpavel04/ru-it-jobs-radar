"""The student of the compose stack (service `student`, llama-server on the CPU) as a corpus_run backend.

    python -m radar.extract.corpus_run run --backend student_cpu:backend \
        --backend-opt base_url=http://student:8080 --skip-done-by qwen3_1p7b_r16_clean-awq --limit 150

Loaded by radar_ml_label from airflow/include (on PYTHONPATH in the Airflow worker) and run in the
project's environment. The corpus run built its backend with
radar.extract.student_server.backend, which renders prompts with the merged model's Hugging Face
tokenizer and so needs transformers; this environment has the project's dependencies only. Here the
same rendering reads the merged model's tokenizer.json with `tokenizers`, the library behind the
fast tokenizer (same token ids, same offsets), and asks llama-server's /apply-template for the chat
template, which gave the tokenizer's prompt on all 165 dev records for this GGUF
(template_match in data/predictions/dev/sft_1p7b_clean_q4_k_m.meta.json). The prompt then goes
through radar.train.quantize.prompt_rows and student_server.CorpusBackend unchanged: input cut at
max_input_tokens, greedy, the extraction schema, rows exactly as in the corpus run, and the server's prompt
token count checked against the local tokenizer's on every request.

model_id is `<name>-<quant>` as student_server.connect names it: qwen3_1p7b_r16_clean-q4_k_m, a
model of its own next to the AWQ checkpoint of the corpus run.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
from tokenizers import Tokenizer

from radar.extract.schema import json_schema
from radar.extract.student_server import MAX_NEW_TOKENS, MERGED, CorpusBackend, LlamaServer
from radar.train.predict import dataset_meta


@dataclass(frozen=True)
class Encoding:
    """The two parts of a Hugging Face encoding that prompt_rows reads."""

    input_ids: list[int]
    offset_mapping: list[tuple[int, int]]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class ServedTokenizer:
    """Tokens from tokenizer.json, the chat template from the server that serves the weights."""

    def __init__(self, tokenizer_file: Path, client: httpx.Client) -> None:
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self.client = client

    def __call__(self, text: str, add_special_tokens: bool = True, return_offsets_mapping: bool = False) -> Encoding:
        enc = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return Encoding(list(enc.ids), [(int(a), int(b)) for a, b in enc.offsets])

    def apply_chat_template(
        self, messages: Sequence[Mapping[str, str]], tokenize: bool, add_generation_prompt: bool, enable_thinking: bool
    ) -> str:
        # /apply-template always ends with the generation prompt, as the scored prompts do.
        if tokenize or not add_generation_prompt:
            raise ValueError("the server renders text prompts with the generation prompt only")
        r = self.client.post(
            "/apply-template",
            json={"messages": list(messages), "chat_template_kwargs": {"enable_thinking": enable_thinking}},
        )
        r.raise_for_status()
        prompt: str = r.json()["prompt"]
        return prompt


def backend(
    base_url: str,
    name: str = "qwen3_1p7b_r16_clean",
    quant: str = "Q4_K_M",
    timeout_s: str = "600",
    transport: httpx.BaseTransport | None = None,
) -> CorpusBackend:
    """Factory for `--backend student_cpu:backend`; refuses a server that serves other weights."""
    client = httpx.Client(base_url=base_url, timeout=httpx.Timeout(float(timeout_s), connect=10.0), transport=transport)
    server = LlamaServer(client, MAX_NEW_TOKENS, json_schema())
    # compose copies the GGUF to a volume, so only the file name still says which weights these are
    served = PurePosixPath(server.info()["model_path"]).stem
    if served != f"{name}-{quant}":
        raise SystemExit(f"the server serves {served}, not {name}-{quant}")
    merged = MERGED / name
    tokenizer = ServedTokenizer(merged / "tokenizer.json", client)
    return CorpusBackend(server, tokenizer, dataset_meta(merged), f"{name}-{quant.lower()}")
