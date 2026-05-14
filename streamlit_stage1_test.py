from __future__ import annotations

import json
import os
import re
from contextlib import nullcontext
from pathlib import Path

from PIL import Image
import streamlit as st
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage_1_projector_coco_uit_final"
DEFAULT_PROMPT = "<image>\nMô tả hình ảnh này: "


st.set_page_config(page_title="Stage 1 Projector Test", layout="wide")


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)(?:\.pt)?$", path.name)
    return int(match.group(1)) if match else -1


def _is_checkpoint(path: Path) -> bool:
    if not path.name.startswith("checkpoint-"):
        return False
    if path.is_file():
        return path.suffix == ".pt"
    return (path / "projector.pt").exists()


def find_checkpoints(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    checkpoints = [p for p in output_dir.iterdir() if _is_checkpoint(p)]
    return sorted(checkpoints, key=checkpoint_step, reverse=True)


def read_checkpoint_pointer(output_dir: Path, name: str) -> Path | None:
    pointer_path = output_dir / f"{name}_checkpoint.json"
    if not pointer_path.exists():
        return None
    try:
        with pointer_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        checkpoint = Path(str(payload["checkpoint"])).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = output_dir / checkpoint
        checkpoint = checkpoint.resolve()
        return checkpoint if _is_checkpoint(checkpoint) else None
    except Exception:
        return None


def default_checkpoint_index(output_dir: Path, checkpoints: list[Path]) -> int:
    resolved = [path.resolve() for path in checkpoints]
    for pointer_name in ("best", "last"):
        pointer = read_checkpoint_pointer(output_dir, pointer_name)
        if pointer and pointer.resolve() in resolved:
            return resolved.index(pointer.resolve())
    return 0


def load_checkpoint_training_config(checkpoint_path: Path) -> dict:
    config_path = checkpoint_path / "training_config.yaml" if checkpoint_path.is_dir() else None
    if not config_path or not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise TypeError(f"{config_path} must contain a mapping.")
    return config


def merge_checkpoint_config(base_config: dict, checkpoint_path: Path) -> dict:
    merged = dict(base_config)
    merged.update(load_checkpoint_training_config(checkpoint_path))
    return merged


def resolve_tokenizer_source(checkpoint_path: str | Path, fallback_llm_model: str) -> str:
    tokenizer_dir = Path(checkpoint_path).expanduser().resolve() / "tokenizer"
    return str(tokenizer_dir) if tokenizer_dir.exists() else fallback_llm_model


@st.cache_data(show_spinner=False)
def load_train_config() -> dict:
    from src.runtime import load_config

    return load_config("train")


def as_list(value) -> list:
    return value if isinstance(value, list) else [value]


@st.cache_data(show_spinner=False)
def load_eval_samples(eval_jsonl, limit: int = 200) -> list[dict]:
    from src.runtime import resolve_record_image_path

    samples: list[dict] = []
    jsonl_paths = as_list(eval_jsonl)
    per_path_limit = max(1, limit // max(len(jsonl_paths), 1))
    for jsonl_path in jsonl_paths:
        p = Path(jsonl_path).expanduser().resolve()
        if not p.exists():
            continue
        path_count = 0
        with p.open("r", encoding="utf-8") as handle:
            for line in handle:
                if path_count >= per_path_limit or len(samples) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                image = resolve_record_image_path(record.get("image", ""), jsonl_path=p)
                caption = str(record.get("caption", "")).strip()
                if image and Path(image).exists():
                    samples.append({"image": image, "caption": caption, "source": p.parent.name})
                    path_count += 1
    return samples


def detect_devices() -> list[str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return ["cpu"]
        return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())] + ["cpu"]
    except Exception:
        return ["cpu"]


def default_device_index(devices: list[str]) -> int:
    requested = os.environ.get("STREAMLIT_DEVICE", "").strip()
    if requested in devices:
        return devices.index(requested)
    if "cuda:2" in devices:
        return devices.index("cuda:2")
    if "cuda:0" in devices:
        return devices.index("cuda:0")
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


@st.cache_resource(show_spinner="Đang load model và checkpoint...")
def load_model_resource(
    checkpoint_path: str,
    vision_model: str,
    llm_model: str,
    model_dtype: str,
    projector_dtype: str,
    device_name: str,
):
    import torch

    from src.modeling import build_model, build_processor
    from src.training.checkpoint import load_projector_checkpoint

    device = torch.device(device_name)
    tokenizer_source = resolve_tokenizer_source(checkpoint_path, llm_model)
    processor = build_processor(vision_model, tokenizer_source)
    model = build_model(
        vision_model,
        llm_model,
        tokenizer_name_or_path=tokenizer_source,
        model_dtype=model_dtype,
        projector_dtype=projector_dtype,
        image_token_id=processor.tokenizer.convert_tokens_to_ids("<image>"),
        vocab_size=len(processor.tokenizer),
    )
    state = load_projector_checkpoint(checkpoint_path, model)
    step = int(state.get("global_step") or checkpoint_step(Path(checkpoint_path)))
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    return model, processor, step


def move_inputs_to_device(inputs: dict, model):
    import torch

    device = next(model.parameters()).device
    vision_dtype = next(model.vision_tower.parameters()).dtype
    moved = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif key == "pixel_values":
            moved[key] = value.to(device=device, dtype=vision_dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def eos_token_ids(tokenizer) -> list[int]:
    ids = {tokenizer.eos_token_id}
    for token in ("<|eot_id|>", "<|end_of_text|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            ids.add(token_id)
    return sorted(i for i in ids if i is not None)


def generate_caption(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, str]:
    import torch

    image = image.convert("RGB")
    tokenizer = processor.tokenizer
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = move_inputs_to_device(inputs, model)
    device = next(model.parameters()).device
    vision_dtype = next(model.vision_tower.parameters()).dtype
    use_sampling = temperature > 0.0

    autocast_context = nullcontext()
    if device.type == "cuda" and vision_dtype in (torch.float16, torch.bfloat16):
        autocast_context = torch.autocast(device_type="cuda", dtype=vision_dtype)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "eos_token_id": eos_token_ids(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": use_sampling,
    }
    if use_sampling:
        generation_kwargs.update({"temperature": temperature, "top_p": top_p})

    with torch.inference_mode(), autocast_context:
        generated_ids = model.generate(**inputs, **generation_kwargs)

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[0, input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
    return text, raw_text


def _tensor_stats(values) -> dict[str, float]:
    flat = values.detach().float().reshape(-1, values.shape[-1])
    token_norm = flat.norm(dim=-1)
    return {
        "std": float(flat.std().cpu()),
        "rms": float(flat.pow(2).mean().sqrt().cpu()),
        "token_norm": float(token_norm.mean().cpu()),
    }


def projector_scale_diagnostics(model, processor, image: Image.Image, prompt: str) -> dict[str, float]:
    import torch

    image = image.convert("RGB")
    tokenizer = processor.tokenizer
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = move_inputs_to_device(inputs, model)
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    device = next(model.parameters()).device
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    text_mask = input_ids != image_token_id
    if tokenizer.pad_token_id is not None:
        text_mask = text_mask & (input_ids != tokenizer.pad_token_id)

    with torch.inference_mode():
        vision_outputs = model.vision_tower(
            pixel_values=pixel_values, output_hidden_states=True, return_dict=True
        )
        feature_layer = model.config.vision_feature_layer
        select_strategy = model.config.vision_feature_select_strategy
        if isinstance(feature_layer, int):
            selected_features = vision_outputs.hidden_states[feature_layer]
            if select_strategy == "default":
                selected_features = selected_features[:, 1:]
        else:
            hidden_states = [vision_outputs.hidden_states[idx] for idx in feature_layer]
            if select_strategy == "default":
                hidden_states = [state[:, 1:] for state in hidden_states]
            selected_features = torch.cat(hidden_states, dim=-1)

        image_features = model.multi_modal_projector(selected_features)
        text_embeds = model.get_input_embeddings()(input_ids.to(device))[text_mask]

    image_stats = _tensor_stats(image_features)
    text_stats = _tensor_stats(text_embeds)
    return {
        "image_tokens": int(image_features.shape[1]),
        "projector_std": image_stats["std"],
        "projector_rms": image_stats["rms"],
        "projector_token_norm": image_stats["token_norm"],
        "text_std": text_stats["std"],
        "text_rms": text_stats["rms"],
        "text_token_norm": text_stats["token_norm"],
        "std_ratio": image_stats["std"] / max(text_stats["std"], 1e-12),
        "norm_ratio": image_stats["token_norm"] / max(text_stats["token_norm"], 1e-12),
    }


def format_diag_row(name: str, diag: dict[str, float]) -> dict[str, str]:
    return {
        "input": name,
        "image_tokens": str(diag["image_tokens"]),
        "projector_norm": f"{diag['projector_token_norm']:.3f}",
        "text_norm": f"{diag['text_token_norm']:.3f}",
        "norm_ratio": f"{diag['norm_ratio']:.1f}x",
        "projector_std": f"{diag['projector_std']:.4f}",
        "text_std": f"{diag['text_std']:.4f}",
        "std_ratio": f"{diag['std_ratio']:.1f}x",
    }


def open_image_from_upload(uploaded_file) -> Image.Image:
    return Image.open(uploaded_file).convert("RGB")


def open_image_from_path(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def main() -> None:
    st.title("Test Stage 1 Projector")

    try:
        cfg = load_train_config()
    except Exception as error:
        st.error(f"Không đọc được config.yaml: {error}")
        st.stop()

    with st.sidebar:
        st.header("Checkpoint")
        output_dir_text = st.text_input("Output dir", value=str(DEFAULT_OUTPUT_DIR))
        output_dir = Path(output_dir_text).expanduser().resolve()
        checkpoints = find_checkpoints(output_dir)
        if not checkpoints:
            st.error(f"Không tìm thấy checkpoint có projector.pt hoặc checkpoint-*.pt trong {output_dir}")
            st.stop()

        checkpoint = st.selectbox(
            "Checkpoint",
            checkpoints,
            index=default_checkpoint_index(output_dir, checkpoints),
            format_func=lambda p: f"{p.name} (step {checkpoint_step(p)})",
        )

        effective_cfg = merge_checkpoint_config(cfg, checkpoint)

        st.header("Model")
        vision_model = st.text_input("Vision model", value=str(effective_cfg["vision_model"]))
        llm_model = st.text_input("LLM model", value=str(effective_cfg["llm_model"]))
        model_dtype = st.selectbox(
            "Model dtype",
            ["bfloat16", "float16", "float32", "auto"],
            index=["bfloat16", "float16", "float32", "auto"].index(
                str(effective_cfg.get("model_dtype", "bfloat16"))
            )
            if str(effective_cfg.get("model_dtype", "bfloat16")) in ["bfloat16", "float16", "float32", "auto"]
            else 0,
        )
        projector_dtype = st.selectbox(
            "Projector dtype",
            ["bfloat16", "float16", "float32"],
            index=["bfloat16", "float16", "float32"].index(
                str(effective_cfg.get("projector_dtype", "float32"))
            )
            if str(effective_cfg.get("projector_dtype", "float32")) in ["bfloat16", "float16", "float32"]
            else 0,
        )
        devices = detect_devices()
        device_name = st.selectbox(
            "Device", devices, index=default_device_index(devices), format_func=device_label
        )
        if device_name == "cuda:2":
            st.caption("Đang chọn card vật lý cuda:2. Nếu chạy bằng CUDA_VISIBLE_DEVICES=2 thì chọn cuda:0.")

        st.header("Generate")
        max_new_tokens = st.slider("Max new tokens", 8, 256, 64, step=8)
        min_new_tokens = st.slider("Min new tokens", 0, 64, 5, step=1)
        temperature = st.slider("Temperature", 0.0, 1.5, 0.0, step=0.05)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)
        st.header("Diagnostics")
        run_black_baseline = st.checkbox("Black-image baseline", value=True)
        run_scale_diagnostics = st.checkbox("Projector scale", value=True)

        if st.button("Clear cache"):
            load_model_resource.clear()
            st.rerun()

    samples = load_eval_samples(cfg.get("eval_jsonl", []), limit=200)
    uploaded_file = st.file_uploader("Upload ảnh", type=["jpg", "jpeg", "png", "webp"])

    left, right = st.columns([0.45, 0.55], gap="large")
    with left:
        sample = None
        if uploaded_file is None and samples:
            sample = st.selectbox(
                "Ảnh mẫu từ eval",
                samples,
                format_func=lambda x: (
                    f"{x.get('source', 'eval')} | {Path(x['image']).name} | {x['caption'][:80]}"
                ),
            )

        try:
            if uploaded_file is not None:
                image = open_image_from_upload(uploaded_file)
                reference = ""
                image_path = uploaded_file.name
            elif sample is not None:
                image = open_image_from_path(sample["image"])
                reference = sample["caption"]
                image_path = sample["image"]
            else:
                st.info("Upload một ảnh để test.")
                st.stop()
        except Exception as error:
            st.error(f"Không đọc được ảnh: {error}")
            st.stop()

        st.image(image, caption=str(image_path), use_container_width=True)
        if reference:
            st.caption(f"Reference: {reference}")

    with right:
        prompt_text = st.text_area("Prompt", value=DEFAULT_PROMPT, height=90)
        run = st.button("Sinh mô tả", type="primary", use_container_width=True)

        if run:
            try:
                model, processor, step = load_model_resource(
                    str(checkpoint), vision_model, llm_model, model_dtype, projector_dtype, device_name
                )
                text, raw_text = generate_caption(
                    model=model,
                    processor=processor,
                    image=image,
                    prompt=prompt_text,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            except Exception as error:
                st.error(f"Không chạy được inference: {error}")
                st.exception(error)
                st.stop()

            st.subheader(f"Prediction - step {step}")
            st.write(text or "<empty>")
            with st.expander("Raw decode"):
                st.code(raw_text or "<empty>")

            if run_black_baseline:
                black_image = Image.new("RGB", image.size, (0, 0, 0))
                black_text, black_raw_text = generate_caption(
                    model=model,
                    processor=processor,
                    image=black_image,
                    prompt=prompt_text,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                st.subheader("Black-image baseline")
                st.write(black_text or "<empty>")
                with st.expander("Black raw decode"):
                    st.code(black_raw_text or "<empty>")

            if run_scale_diagnostics:
                rows = [
                    format_diag_row(
                        "real",
                        projector_scale_diagnostics(
                            model=model, processor=processor, image=image, prompt=prompt_text
                        ),
                    )
                ]
                if run_black_baseline:
                    rows.append(
                        format_diag_row(
                            "black",
                            projector_scale_diagnostics(
                                model=model, processor=processor, image=black_image, prompt=prompt_text
                            ),
                        )
                    )
                st.subheader("Projector Scale")
                st.table(rows)
                max_ratio = max(float(row["norm_ratio"].rstrip("x")) for row in rows)
                if max_ratio > 50:
                    st.warning(
                        "Projector norm đang lớn hơn text embedding rất nhiều. "
                        "Checkpoint này có nguy cơ bị visual soft-prompt collapse."
                    )


if __name__ == "__main__":
    main()
