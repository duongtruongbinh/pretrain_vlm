from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from src.inference import eos_token_ids


def detect_devices() -> list[str]:
    try:
        import torch
        if not torch.cuda.is_available():
            return ["cpu"]
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())] + ["cpu"]
    except Exception:
        return ["cpu"]


def default_device_index(devices: list[str]) -> int:
    requested = os.environ.get("STREAMLIT_DEVICE", "").strip()
    if requested in devices:
        return devices.index(requested)
    for preferred in ("cuda:2", "cuda:0"):
        if preferred in devices:
            return devices.index(preferred)
    return 0


def device_label(device_name: str) -> str:
    try:
        import torch
        if not device_name.startswith("cuda") or not torch.cuda.is_available():
            return device_name
        device = torch.device(device_name)
        props = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        used_gb = (total_bytes - free_bytes) / 1024**3
        total_gb = total_bytes / 1024**3
        return f"{device_name} | {props.name} | {used_gb:.1f}/{total_gb:.1f} GB"
    except Exception:
        return device_name


def checkpoint_step(path: Path) -> int:
    import re
    match = re.search(r"checkpoint-(\d+)(?:\.pt)?$", path.name)
    return int(match.group(1)) if match else -1


def read_checkpoint_pointer(output_dir: Path, name: str) -> Path | None:
    pointer_path = output_dir / f"{name}_checkpoint.json"
    if not pointer_path.exists():
        return None
    try:
        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
        return Path(str(payload["checkpoint"])).expanduser().resolve()
    except Exception:
        return None


def load_checkpoint_config(checkpoint_path: Path) -> dict:
    cfg_path = checkpoint_path / "training_config.yaml"
    if not cfg_path.exists():
        return {}
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return cfg if isinstance(cfg, dict) else {}


def merge_checkpoint_config(base: dict, checkpoint_path: Path) -> dict:
    return {**base, **load_checkpoint_config(checkpoint_path)}


def default_checkpoint_index(output_dir: Path, checkpoints: list[Path]) -> int:
    resolved = [p.resolve() for p in checkpoints]
    for name in ("best", "last"):
        ptr = read_checkpoint_pointer(output_dir, name)
        if ptr and ptr in resolved:
            return resolved.index(ptr)
    return 0
