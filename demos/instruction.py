from __future__ import annotations

import json
import re
from contextlib import nullcontext
from pathlib import Path

from PIL import Image
import streamlit as st
import yaml

from _utils import default_device_index, detect_devices, device_label, eos_token_ids


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "instruction_run2"


def _load_default_system_prompt() -> str:
    from src.prompts import render
    return render("vqa_system.j2")


DEFAULT_SYSTEM_PROMPT = _load_default_system_prompt()

st.set_page_config(page_title="Instruction Model Test", layout="wide")


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def _is_checkpoint(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("checkpoint-") and (path / "projector.pt").exists()


def find_checkpoints(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    return sorted([p for p in output_dir.iterdir() if _is_checkpoint(p)], key=checkpoint_step, reverse=True)


def read_checkpoint_pointer(output_dir: Path, name: str) -> Path | None:
    pointer_path = output_dir / f"{name}_checkpoint.json"
    if not pointer_path.exists():
        return None
    try:
        with pointer_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        ckpt = Path(str(payload["checkpoint"])).expanduser().resolve()
        return ckpt if _is_checkpoint(ckpt) else None
    except Exception:
        return None


def default_checkpoint_index(output_dir: Path, checkpoints: list[Path]) -> int:
    resolved = [p.resolve() for p in checkpoints]
    for name in ("best", "last"):
        ptr = read_checkpoint_pointer(output_dir, name)
        if ptr and ptr.resolve() in resolved:
            return resolved.index(ptr.resolve())
    return 0


def load_checkpoint_config(checkpoint_path: Path) -> dict:
    cfg_path = checkpoint_path / "training_config.yaml"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg if isinstance(cfg, dict) else {}


def merge_config(base: dict, checkpoint_path: Path) -> dict:
    merged = dict(base)
    merged.update(load_checkpoint_config(checkpoint_path))
    return merged


def resolve_checkpoint_sources(checkpoint_path: str | Path, fallback_llm_model: str) -> tuple[str, str]:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    llm_dir = checkpoint / "llm"
    tokenizer_dir = checkpoint / "tokenizer"
    llm_source = str(llm_dir) if llm_dir.exists() else fallback_llm_model
    tokenizer_source = str(tokenizer_dir) if tokenizer_dir.exists() else fallback_llm_model
    return llm_source, tokenizer_source


@st.cache_data(show_spinner=False)
def load_instruction_config() -> dict:
    from src.runtime import load_config

    return load_config("instruction_train")


def as_list(value) -> list:
    return value if isinstance(value, list) else [value]


@st.cache_data(show_spinner=False)
def load_eval_samples(eval_jsonl, limit: int = 200) -> list[dict]:
    from src.runtime import resolve_record_image_path

    samples: list[dict] = []
    paths = as_list(eval_jsonl)
    per_path = max(1, limit // max(len(paths), 1))
    for jsonl_path in paths:
        p = Path(jsonl_path).expanduser().resolve()
        if not p.exists():
            continue
        count = 0
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                if count >= per_path or len(samples) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                image_val = record.get("image", "")
                if not image_val:
                    continue
                image = resolve_record_image_path(image_val, jsonl_path=p)
                messages = record.get("messages", [])
                if not messages or not Path(image).exists():
                    continue
                first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
                samples.append(
                    {
                        "image": image,
                        "messages": messages,
                        "first_user": str(first_user)[:100],
                        "source": p.parent.name,
                        "sample_type": record.get("sample_type", ""),
                    }
                )
                count += 1
    return samples


@st.cache_resource(show_spinner="Đang load model...")
def load_model_resource(
    checkpoint_path: str,
    vision_model: str,
    llm_model: str,
    model_dtype: str,
    projector_dtype: str,
    max_text_tokens: int,
    device_name: str,
):
    import torch
    from src.collators import InstructionCollator
    from src.modeling import build_model
    from src.training import load_full_checkpoint

    device = torch.device(device_name)
    llm_source, tokenizer_source = resolve_checkpoint_sources(checkpoint_path, llm_model)
    collator = InstructionCollator(vision_model, tokenizer_source, max_text_tokens=max_text_tokens)
    model = build_model(
        vision_model,
        llm_source,
        tokenizer_name_or_path=tokenizer_source,
        model_dtype=model_dtype,
        projector_dtype=projector_dtype,
        image_token_id=collator.image_token_id,
        vocab_size=len(collator.tokenizer),
    )
    state = load_full_checkpoint(checkpoint_path, model)
    step = int(state.get("global_step") or checkpoint_step(Path(checkpoint_path)))
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    return model, collator, step


def generate_reply(
    model,
    collator,
    image: Image.Image,
    messages: list[dict],
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float = 1.1,
) -> tuple[str, str]:
    import torch

    device = next(model.parameters()).device
    vision_dtype = next(model.vision_tower.parameters()).dtype
    tokenizer = collator.tokenizer

    prompt_ids, attn_mask, pixel_values = collator.build_prompt_tensors(messages, image, device=device)
    pixel_values = pixel_values.to(dtype=vision_dtype)

    use_sampling = temperature > 0.0
    gen_kwargs: dict = {
        "input_ids": prompt_ids,
        "pixel_values": pixel_values,
        "attention_mask": attn_mask,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "eos_token_id": eos_token_ids(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": use_sampling,
        "repetition_penalty": repetition_penalty,
    }
    if use_sampling:
        gen_kwargs.update({"temperature": temperature, "top_p": top_p})

    autocast_ctx = nullcontext()
    if device.type == "cuda" and vision_dtype in (torch.float16, torch.bfloat16):
        autocast_ctx = torch.autocast(device_type="cuda", dtype=vision_dtype)

    with torch.inference_mode(), autocast_ctx:
        generated_ids = model.generate(**gen_kwargs)

    input_len = prompt_ids.shape[1]
    new_tokens = generated_ids[0, input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
    return text, raw_text


def main() -> None:
    st.title("Test Instruction Model")

    try:
        cfg = load_instruction_config()
    except Exception as error:
        st.error(f"Không đọc được config.yaml: {error}")
        st.stop()

    with st.sidebar:
        st.header("Checkpoint")
        output_dir_text = st.text_input("Output dir", value=str(DEFAULT_OUTPUT_DIR))
        output_dir = Path(output_dir_text).expanduser().resolve()
        checkpoints = find_checkpoints(output_dir)
        if not checkpoints:
            st.error(f"Không tìm thấy checkpoint trong {output_dir}")
            st.stop()

        checkpoint = st.selectbox(
            "Checkpoint",
            checkpoints,
            index=default_checkpoint_index(output_dir, checkpoints),
            format_func=lambda p: f"{p.name} (step {checkpoint_step(p)})",
        )
        effective_cfg = merge_config(cfg, checkpoint)

        st.header("Model")
        vision_model = st.text_input("Vision model", value=str(effective_cfg.get("vision_model", "")))
        llm_model = st.text_input("LLM model", value=str(effective_cfg.get("llm_model", "")))

        dtype_opts = ["bfloat16", "float16", "float32", "auto"]
        mdtype_val = str(effective_cfg.get("model_dtype", "bfloat16"))
        model_dtype = st.selectbox(
            "Model dtype", dtype_opts, index=dtype_opts.index(mdtype_val) if mdtype_val in dtype_opts else 0
        )

        pdtype_opts = ["float32", "bfloat16", "float16"]
        pdtype_val = str(effective_cfg.get("projector_dtype", "float32"))
        projector_dtype = st.selectbox(
            "Projector dtype",
            pdtype_opts,
            index=pdtype_opts.index(pdtype_val) if pdtype_val in pdtype_opts else 0,
        )

        max_text_tokens = st.number_input(
            "Max text tokens",
            min_value=256,
            max_value=4096,
            value=int(effective_cfg.get("max_text_tokens", 2048)),
            step=256,
        )

        devices = detect_devices()
        device_name = st.selectbox(
            "Device", devices, index=default_device_index(devices), format_func=device_label
        )

        st.header("System Prompt")
        system_prompt = st.text_area(
            "system_prompt", label_visibility="collapsed", value=DEFAULT_SYSTEM_PROMPT, height=110
        )

        st.header("Generate")
        max_new_tokens = st.slider(
            "Max new tokens", 16, 768, int(effective_cfg.get("max_new_tokens", 256)), step=16
        )
        min_new_tokens = st.slider("Min new tokens", 0, 64, 5, step=1)
        temperature = st.slider("Temperature", 0.0, 1.5, 0.0, step=0.05)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)
        repetition_penalty = st.slider("Repetition penalty", 1.0, 1.5, 1.1, step=0.05)

        if st.button("Clear model cache"):
            load_model_resource.clear()
            st.rerun()

    samples = load_eval_samples(cfg.get("eval_jsonl", []), limit=200)
    uploaded_file = st.file_uploader("Upload ảnh", type=["jpg", "jpeg", "png", "webp"])

    left, right = st.columns([0.4, 0.6], gap="large")

    with left:
        selected_sample = None
        if uploaded_file is None and samples:
            selected_sample = st.selectbox(
                "Ảnh mẫu từ eval",
                samples,
                format_func=lambda x: (
                    f"[{x.get('source', '')}] {Path(x['image']).name} | {x['first_user'][:55]}"
                ),
            )

        try:
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                image_key = uploaded_file.name
            elif selected_sample is not None:
                with Image.open(selected_sample["image"]) as img:
                    image = img.convert("RGB")
                image_key = selected_sample["image"]
            else:
                st.info("Upload một ảnh hoặc chọn từ eval để bắt đầu.")
                st.stop()
        except Exception as error:
            st.error(f"Không đọc được ảnh: {error}")
            st.stop()

        st.image(image, caption=Path(image_key).name, use_container_width=True)

        if selected_sample and uploaded_file is None:
            with st.expander("Reference conversation"):
                for msg in selected_sample["messages"]:
                    role = msg["role"]
                    content = str(msg["content"])
                    if role == "system":
                        st.caption(f"**[system]** {content}")
                    elif role == "user":
                        st.markdown(f"**User:** {content}")
                    else:
                        st.markdown(f"**Assistant:** {content}")

    with right:
        session_key = f"chat_{image_key}"
        if session_key not in st.session_state:
            st.session_state[session_key] = []

        chat_history: list[dict] = st.session_state[session_key]

        if st.button("Xóa conversation"):
            st.session_state[session_key] = []
            st.rerun()

        for msg in chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("Nhập câu hỏi...")
        if user_input:
            chat_history.append({"role": "user", "content": user_input})

            model_messages: list[dict] = []
            if system_prompt.strip():
                model_messages.append({"role": "system", "content": system_prompt.strip()})
            for msg in chat_history:
                model_messages.append({"role": msg["role"], "content": msg["content"]})

            with st.spinner("Đang sinh câu trả lời..."):
                try:
                    model, collator, step = load_model_resource(
                        str(checkpoint),
                        vision_model,
                        llm_model,
                        model_dtype,
                        projector_dtype,
                        int(max_text_tokens),
                        device_name,
                    )
                    text, raw_text = generate_reply(
                        model=model,
                        collator=collator,
                        image=image,
                        messages=model_messages,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    )
                except Exception as error:
                    st.error(f"Lỗi inference: {error}")
                    st.exception(error)
                    chat_history.pop()
                    st.stop()

            chat_history.append({"role": "assistant", "content": text or "<empty>"})
            st.rerun()


if __name__ == "__main__":
    main()
