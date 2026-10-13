"""Fingerprint of a model's files: what the registry tags a version with and what a download is
checked against. Standard library only, so the API image verifies a download without mlflow."""

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

_CHUNK = 8 << 20


@dataclass(frozen=True)
class Artifact:
    """A file or directory placed at `dest` inside the version's model directory ("." for its root)."""

    path: Path
    dest: str


def _join(dest: str, name: str) -> str:
    return name if dest in ("", ".") else f"{dest.strip('/')}/{name}"


def artifact_files(artifacts: Iterable[Artifact]) -> Iterator[tuple[str, Path]]:
    """(path inside the model directory, local file) for every file; directories are walked."""
    for a in artifacts:
        if a.path.is_file():
            yield _join(a.dest, a.path.name), a.path
        else:
            for f in sorted(p for p in a.path.rglob("*") if p.is_file()):
                yield _join(a.dest, f.relative_to(a.path).as_posix()), f


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(artifacts: Iterable[Artifact]) -> str:
    """sha256 over the sorted `path<TAB>sha256` lines: the same files give the same value however
    they were listed, and a renamed or moved file changes it."""
    lines = sorted(f"{name}\t{file_sha256(path)}\n" for name, path in artifact_files(artifacts))
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def directory_fingerprint(root: Path) -> str:
    return fingerprint([Artifact(root, ".")])
