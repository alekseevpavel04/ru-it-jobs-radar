"""The two extraction backends behind POST /extract.

The student is the fine-tuned Qwen3 served by llama-server. It gets the request exactly as
`radar.train.predict` builds it: the system prompt and max_input_tokens of the run's dataset meta, the
input cut at a token boundary, thinking off, greedy. The server's /tokenize gives the token pieces, so
the cut uses the served model's own tokenizer. Decoding is constrained by the extraction JSON Schema,
and the answer goes through the teacher's check (the same schema rules, a salary or experience object
without bounds read as null, an answer cut at max_tokens rejected).

The regex baseline answers when no student is configured, and in its place when the student fails;
the response names the backend that answered and why the student did not.
"""

import codecs
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

from radar.extract import baseline
from radar.extract.inputs import build_input
from radar.extract.schema import SCHEMA_VERSION, VacancyFields, json_schema, parse
from radar.extract.teacher import read_answer
from radar.train.dataset import cap_input
from radar.train.dataset import messages as chat_turns
from radar.train.predict import MAX_NEW_TOKENS, dataset_meta

log = logging.getLogger("radar.api.extract")

Backend = Literal["student", "regex"]
REGEX_MODEL = f"regex-baseline/schema-v{SCHEMA_VERSION}"


class StudentError(RuntimeError):
    pass


@dataclass(frozen=True)
class Extraction:
    fields: VacancyFields
    backend: Backend
    model: str
    prompt_sha256: str | None
    truncated_input: bool


def regex_extract(text: str) -> Extraction:
    return Extraction(baseline.extract(text), "regex", REGEX_MODEL, None, False)


def spans_from_pieces(pieces: Sequence[str | list[int]]) -> list[tuple[int, int]]:
    """Character span of every token from llama-server's pieces. A piece that is not whole UTF-8 comes
    as a list of bytes; its characters are counted on the token that completes them."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    spans, pos = [], 0
    for piece in pieces:
        data = piece.encode("utf-8") if isinstance(piece, str) else bytes(piece)
        end = pos + len(decoder.decode(data))
        spans.append((pos, end))
        pos = end
    return spans


class StudentExtractor:
    def __init__(
        self,
        url: str,
        meta: Mapping[str, Any],
        timeout_s: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.client = httpx.Client(base_url=url, timeout=httpx.Timeout(timeout_s, connect=3.0), transport=transport)
        self.system = str(meta["system_prompt"])
        self.max_input_tokens = int(meta["max_input_tokens"])
        self.prompt_sha256 = str(meta["prompt_sha256"])
        self.schema = json_schema()

    @classmethod
    def from_meta_dir(cls, url: str, adapter: Path | None, timeout_s: float) -> "StudentExtractor":
        return cls(url, dataset_meta(adapter), timeout_s)

    def identity(self) -> str:
        return f"{self.client.base_url}|{self.prompt_sha256}|{self.max_input_tokens}"

    def _post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        try:
            response = self.client.post(path, json=body)
        except httpx.HTTPError as e:
            raise StudentError(f"{path}: {type(e).__name__}") from e
        if response.status_code != 200:
            raise StudentError(f"{path} answered {response.status_code}")
        payload: dict[str, Any] = response.json()
        return payload

    def capped(self, text: str) -> tuple[str, bool]:
        tokens = self._post("/tokenize", {"content": text, "add_special": False, "with_pieces": True})["tokens"]
        try:
            spans = spans_from_pieces([t["piece"] for t in tokens])
        except UnicodeDecodeError as e:
            raise StudentError("the server's token pieces are not UTF-8") from e
        if not spans or spans[-1][1] != len(text):
            raise StudentError("the server's token pieces do not add up to the input")
        return cap_input(text, spans, self.max_input_tokens)

    def extract(self, text: str) -> Extraction:
        user, truncated = self.capped(text)
        body = {
            "messages": chat_turns(self.system, user),
            "temperature": 0,
            "max_tokens": MAX_NEW_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "vacancy", "schema": self.schema}},
        }
        answer = self._post("/v1/chat/completions", body)
        choice = answer["choices"][0]
        output = read_answer(choice, with_exclude=False)["output"]
        if output is None:
            reason = "cut at max_tokens" if choice.get("finish_reason") == "length" else "broke the schema"
            raise StudentError(f"the student's answer {reason}")
        model = str(answer.get("model") or "unknown")
        return Extraction(parse(output), "student", model, self.prompt_sha256, truncated)

    def ping(self) -> bool:
        try:
            return self.client.get("/health").status_code == 200
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        self.client.close()


def extract(text: str, student: StudentExtractor | None) -> tuple[Extraction, str | None]:
    """The answer and, when the regex stood in for a configured student, the reason."""
    text = build_input(None, text)
    if student is None:
        return regex_extract(text), None
    try:
        return student.extract(text), None
    except StudentError as e:
        log.warning("student failed, regex answers: %s", e)
        return regex_extract(text), str(e)
