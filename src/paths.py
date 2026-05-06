from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_config_path(value: str | Path) -> str:
    return str(Path(value).expanduser().resolve())


def resolve_config_paths(value: str | Path | list[str | Path]) -> str | list[str]:
    if isinstance(value, list):
        return [resolve_config_path(item) for item in value]
    return resolve_config_path(value)


def resolve_record_image_path(image_value: str | Path, *, jsonl_path: Path) -> str:
    raw_path = Path(str(image_value).strip()).expanduser()
    if raw_path.is_absolute():
        return str(raw_path)

    candidates = (PROJECT_ROOT / raw_path, Path.cwd() / raw_path, jsonl_path.parent / raw_path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    return str((PROJECT_ROOT / raw_path).resolve())
