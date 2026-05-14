from __future__ import annotations

import os


def eos_token_ids(tokenizer) -> list[int]:
    ids = {tokenizer.eos_token_id}
    for token in ("<|eot_id|>", "<|end_of_text|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            ids.add(token_id)
    return sorted(i for i in ids if i is not None)


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
