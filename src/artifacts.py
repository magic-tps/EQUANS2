"""Resolve development artifacts or the self-contained published runtime."""
from pathlib import Path


def resolve(root: Path, relative: str) -> Path:
    local=Path(root)/relative
    if local.exists():
        return local
    published=Path(root)/"runtime"/relative
    if published.suffix==".xlsx":
        published=published.with_suffix(".csv")
    return published
