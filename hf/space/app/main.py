"""The Space: a vacancy text in, the extracted fields and a salary estimate out.

llama-server, built from the llama.cpp release the quantized student was scored with, serves the
GGUF on LLAMA_URL. A request renders the prompt with the training code (app.prompt), sends it as text
to /completion with greedy decoding constrained by the extraction JSON schema (scored on dev as
sft_1p7b_clean_q4_k_m_schema; free decoding now and then writes a value outside the schema), checks
the answer against the schema and gives the fields to the salary model. Nothing is stored.
"""

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from app.config import ASSETS, LLAMA_URL, space_info
from app.prompt import Prompter
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from radar.extract.schema import dump, json_schema, try_parse
from radar.models import salary as salary_model
from radar.models.salary_features import OutOfScope

MAX_NEW_TOKENS = 512
MAX_TEXT_CHARS = 20_000
STATIC = Path(__file__).resolve().parent / "static"


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


@lru_cache(maxsize=1)
def prompter() -> Prompter:
    return Prompter()


@lru_cache(maxsize=1)
def load_salary() -> salary_model.SalaryModel:
    """The salary model the Space serves; the one place that knows where it lies."""
    return salary_model.load_model(ASSETS / "salary" / "salary.pkl")


@lru_cache(maxsize=1)
def llama() -> httpx.Client:
    return httpx.Client(base_url=LLAMA_URL, timeout=httpx.Timeout(300.0, connect=5.0))


def complete(prompt: str) -> dict[str, Any]:
    r = llama().post(
        "/completion",
        json={
            "prompt": prompt,
            "n_predict": MAX_NEW_TOKENS,
            "temperature": 0,
            "cache_prompt": True,
            "json_schema": json_schema(),
        },
    )
    r.raise_for_status()
    data: dict[str, Any] = r.json()
    return data


def estimate_salary(fields: dict[str, Any], source: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return salary_model.predict(fields, source, model=load_salary()), None
    except OutOfScope:
        return None, "non_it"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    prompter()
    load_salary()
    yield


app = FastAPI(title="ru-it-jobs-radar", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info() -> dict[str, Any]:
    return space_info()


@app.get("/health")
def health() -> JSONResponse:
    try:
        ready = llama().get("/health").status_code == 200
    except httpx.TransportError:
        ready = False
    return JSONResponse({"ready": ready}, status_code=200 if ready else 503)


@app.post("/api/extract")
def extract(body: ExtractRequest) -> dict[str, Any]:
    started = time.perf_counter()
    rendered = prompter().render(body.text)
    try:
        data = complete(rendered.prompt)
    except httpx.HTTPError as e:
        raise HTTPException(503, f"модель недоступна: {e}") from e
    after_model = time.perf_counter()
    raw = data["content"].strip()
    parsed = try_parse(raw)
    fields = dump(parsed) if parsed is not None else None
    salary, salary_note = (
        (None, "invalid") if fields is None else estimate_salary(fields, space_info()["salary_source"])
    )
    finished = time.perf_counter()
    return {
        "fields": fields,
        "valid": fields is not None,
        "raw": raw,
        "finish": "length" if data["stop_type"] == "limit" else "stop",
        "input_truncated": rendered.truncated,
        "prompt_tokens": data["tokens_evaluated"],
        "prompt_tokens_tokenizer": rendered.prompt_tokens,
        "completion_tokens": data["tokens_predicted"],
        "salary": salary,
        "salary_note": salary_note,
        "seconds": {
            "model": round(after_model - started, 3),
            "prompt": round(data["timings"]["prompt_ms"] / 1000, 3),
            "generate": round(data["timings"]["predicted_ms"] / 1000, 3),
            "salary": round(finished - after_model, 3),
            "total": round(finished - started, 3),
        },
    }
