"""Evaluate a Stage-1 projector checkpoint on KTVIC-style caption data."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.io import read_json_or_jsonl, write_json, write_jsonl  # noqa: E402
from src.evaluation.metrics import caption_metrics  # noqa: E402
from src.modeling import build_model, build_processor  # noqa: E402
from src.runtime import load_config  # noqa: E402
from src.training.checkpoint import load_projector_checkpoint  # noqa: E402


PROMPT_TEMPLATE = "<image>\nMô tả hình ảnh này: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage-1 captioning on KTVIC.")
    parser.add_argument("--annotations", default=None, help="KTVIC annotation JSON/JSONL path.")
    parser.add_argument("--image-root", default=None, help="Directory containing benchmark images.")
    parser.add_argument("--checkpoint", default=None, help="Stage-1 checkpoint directory.")
    parser.add_argument("--predictions-jsonl", default=None, help="Existing predictions JSONL; skips generation.")
    parser.add_argument("--output-dir", default="outputs/benchmarks/ktvic")
    parser.add_argument("--config-section", default="train")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt", default=PROMPT_TEMPLATE)
    return parser.parse_args()


def load_caption_records(
    annotation_path: str | Path, image_root: str | Path | None = None, limit: int | None = None
) -> list[dict]:
    path = Path(annotation_path).expanduser().resolve()
    root = Path(image_root).expanduser().resolve() if image_root else path.parent
    payload = read_json_or_jsonl(path)
    if not isinstance(payload, dict) or "images" not in payload or "annotations" not in payload:
        raise ValueError("KTVIC annotations must use COCO-style images/annotations JSON.")

    records = _load_coco_style(payload, root)
    if limit is not None:
        return records[:limit]
    return records


def _load_coco_style(payload: dict, image_root: Path) -> list[dict]:
    images = {item["id"]: item for item in payload["images"]}
    grouped: OrderedDict[str, dict] = OrderedDict()
    for annotation in payload["annotations"]:
        image_id = annotation["image_id"]
        image_path = _resolve_image_path(images[image_id]["file_name"], image_root)
        record = grouped.setdefault(image_path, {"id": image_id, "image": image_path, "references": []})
        record["references"].append(str(annotation["caption"]).strip())
    return list(grouped.values())


def _resolve_image_path(image_value: str, image_root: Path) -> str:
    raw = Path(str(image_value).strip()).expanduser()
    if raw.is_absolute():
        return str(raw.resolve())
    return str((image_root / raw).resolve())


def resolve_device(device_name: str):
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_tokenizer_source(checkpoint_path: str | Path) -> str:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    tokenizer_dir = checkpoint / "tokenizer"
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Missing Stage-1 tokenizer: {tokenizer_dir}")
    return str(tokenizer_dir)


def load_stage1_model(checkpoint_path: str | Path, config_section: str, device_name: str):
    cfg = load_config(config_section)
    device = resolve_device(device_name)
    tokenizer_source = resolve_tokenizer_source(checkpoint_path)
    processor = build_processor(cfg["vision_model"], tokenizer_source)
    model = build_model(
        cfg["vision_model"],
        cfg["llm_model"],
        tokenizer_name_or_path=tokenizer_source,
        model_dtype=cfg.get("model_dtype"),
        projector_dtype=cfg.get("projector_dtype", "float32"),
        image_token_id=processor.tokenizer.convert_tokens_to_ids("<image>"),
        vocab_size=len(processor.tokenizer),
    )
    state = load_projector_checkpoint(checkpoint_path, model)
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return model, processor, state


def generate_caption(model, processor, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    vision_dtype = next(model.vision_tower.parameters()).dtype
    tokenizer = processor.tokenizer
    inputs = processor(text=prompt, images=image.convert("RGB"), return_tensors="pt")
    moved = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif key == "pixel_values":
            moved[key] = value.to(device=device, dtype=vision_dtype)
        else:
            moved[key] = value.to(device=device)

    eos_ids = {tokenizer.eos_token_id}
    for token in ("<|eot_id|>", "<|end_of_text|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            eos_ids.add(token_id)

    autocast_context = nullcontext()
    if device.type == "cuda" and vision_dtype in (torch.float16, torch.bfloat16):
        autocast_context = torch.autocast(device_type="cuda", dtype=vision_dtype)

    with torch.inference_mode(), autocast_context:
        generated_ids = model.generate(
            **moved,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=False,
            eos_token_id=sorted(i for i in eos_ids if i is not None),
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = moved["input_ids"].shape[1]
    return tokenizer.decode(generated_ids[0, input_len:], skip_special_tokens=True).strip()


def evaluate_records(args: argparse.Namespace) -> tuple[list[dict], dict]:
    if args.predictions_jsonl:
        rows = read_json_or_jsonl(args.predictions_jsonl)
        if args.max_samples is not None:
            rows = rows[: args.max_samples]
    else:
        if not args.annotations:
            raise ValueError("--annotations is required unless --predictions-jsonl is provided.")
        if not args.checkpoint:
            raise ValueError("--checkpoint is required unless --predictions-jsonl is provided.")
        records = load_caption_records(args.annotations, image_root=args.image_root, limit=args.max_samples)
        model, processor, _state = load_stage1_model(args.checkpoint, args.config_section, args.device)
        rows = []
        for record in tqdm(records, desc="ktvic", dynamic_ncols=True):
            with Image.open(record["image"]) as image:
                prediction = generate_caption(model, processor, image, args.prompt, args.max_new_tokens)
            rows.append(
                {
                    "id": record["id"],
                    "image": record["image"],
                    "prediction": prediction,
                    "references": record["references"],
                }
            )

    scores = caption_metrics([row.get("prediction", "") for row in rows], [row.get("references", []) for row in rows])
    return rows, scores


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    rows, scores = evaluate_records(args)
    write_jsonl(output_dir / "predictions.jsonl", rows)
    write_json(output_dir / "metrics.json", scores)
    print(json.dumps(scores, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
