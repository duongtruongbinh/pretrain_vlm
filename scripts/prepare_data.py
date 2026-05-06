from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

import gdown

from src.config import load_config


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_valid_download(output_path: Path) -> bool:
    if not output_path.exists():
        return False

    if output_path.suffix.lower() == ".json":
        try:
            with output_path.open("r", encoding="utf-8") as handle:
                json.load(handle)
            return True
        except Exception:
            return False

    if output_path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(output_path, "r") as archive:
                return archive.testzip() is None
        except zipfile.BadZipFile:
            return False

    return output_path.stat().st_size > 0


def download_file(file_id: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if is_valid_download(output_path):
        print(f"Skip existing download: {output_path}")
        return
    if output_path.exists():
        output_path.unlink()

    gdown.download(id=file_id, output=str(output_path), quiet=False, resume=True)
    if not output_path.exists():
        raise FileNotFoundError(f"Download did not create expected file: {output_path}")


def extract_images_zip(images_zip_path: Path, images_root: Path) -> None:
    marker_path = images_root / ".extracted"
    if marker_path.exists():
        print(f"Skip extraction, marker found: {marker_path}")
        return

    temp_extract_dir = images_root.parent / "_images_extract_tmp"
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)
    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(images_zip_path, "r") as archive:
        archive.extractall(temp_extract_dir)

    extracted_entries = list(temp_extract_dir.iterdir())
    source_root = (
        extracted_entries[0]
        if len(extracted_entries) == 1 and extracted_entries[0].is_dir()
        else temp_extract_dir
    )

    images_root.mkdir(parents=True, exist_ok=True)
    for child in source_root.iterdir():
        destination = images_root / child.name
        if destination.exists():
            continue
        shutil.move(str(child), str(destination))

    shutil.rmtree(temp_extract_dir, ignore_errors=True)
    marker_path.write_text("ok\n", encoding="utf-8")


def ensure_symlink(source_path: Path, output_path: Path) -> None:
    source_path = source_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_symlink():
        if output_path.resolve() == source_path:
            return
        raise ValueError(f"{output_path} already exists and points somewhere else.")

    if output_path.exists():
        raise ValueError(f"{output_path} already exists and is not a symlink.")

    os.symlink(source_path, output_path, target_is_directory=source_path.is_dir())


def build_image_index(images_root: Path) -> dict[str, Path]:
    image_index = {}
    for image_path in images_root.rglob("*"):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if image_path.name in image_index:
            raise ValueError(f"Duplicate image filename found in extracted images: {image_path.name}")
        image_index[image_path.name] = image_path

    if not image_index:
        raise RuntimeError(f"No images found under {images_root}")
    return image_index


def iter_annotation_records(annotation_path: Path):
    with annotation_path.open("r", encoding="utf-8") as handle:
        annotation_data = json.load(handle)

    if isinstance(annotation_data, dict) and "images" in annotation_data and "annotations" in annotation_data:
        image_id_to_file_name = {
            image_info["id"]: image_info["file_name"] for image_info in annotation_data["images"]
        }
        for annotation in annotation_data["annotations"]:
            image_name = image_id_to_file_name[annotation["image_id"]]
            captions = [annotation["caption"]]
            yield image_name, captions
        return

    if isinstance(annotation_data, dict):
        for image_name, metadata in annotation_data.items():
            if not isinstance(metadata, dict) or "captions" not in metadata:
                raise ValueError(
                    f"Unsupported annotation entry for image '{image_name}' in {annotation_path}"
                )
            yield image_name, metadata["captions"]
        return

    raise ValueError(f"Unsupported annotation format in {annotation_path}")


def maybe_download_dataset(config: dict, raw_dir: Path) -> None:
    download_config = config.get("download", {})
    if not download_config.get("enabled", False):
        return

    downloads_dir = raw_dir / "downloads"
    annotations_dir = raw_dir / "annotations"
    images_root = raw_dir / "images"

    images_download = download_config["images"]
    images_zip_path = downloads_dir / images_download["filename"]
    download_file(images_download["file_id"], images_zip_path)
    extract_images_zip(images_zip_path, images_root)

    annotation_name_map = {"train": "train.json", "val": "val.json", "test": "test.json"}
    for split, target_name in annotation_name_map.items():
        annotation_download = download_config["annotations"][split]
        downloaded_path = downloads_dir / annotation_download["filename"]
        target_path = annotations_dir / target_name
        download_file(annotation_download["file_id"], downloaded_path)
        annotations_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(downloaded_path, target_path)


def main() -> None:
    config = load_config("prepare_data")
    raw_dir = Path(config["raw_dir"]).expanduser().resolve()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    maybe_download_dataset(config, raw_dir)

    source_annotations_dir = raw_dir / "annotations"
    source_images_root = raw_dir / "images"
    if not source_annotations_dir.exists():
        raise FileNotFoundError(f"Missing annotations directory: {source_annotations_dir}")
    if not source_images_root.exists():
        raise FileNotFoundError(f"Missing images directory: {source_images_root}")

    image_index = build_image_index(source_images_root)
    output_images_root = output_dir / "images"
    ensure_symlink(source_images_root, output_images_root)

    for split in ("train", "val", "test"):
        annotation_path = source_annotations_dir / f"{split}.json"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {annotation_path}")
        output_jsonl_path = output_dir / f"{split}.jsonl"
        skipped_empty_captions = 0
        written_rows = 0
        with output_jsonl_path.open("w", encoding="utf-8") as handle:
            for image_name, captions in iter_annotation_records(annotation_path):
                if image_name not in image_index:
                    raise FileNotFoundError(f"Missing image referenced by annotations: {image_name}")

                source_image_path = image_index[image_name]
                relative_image_path = source_image_path.relative_to(source_images_root)
                output_image_path = output_images_root / relative_image_path

                for caption in captions:
                    caption = str(caption).strip()
                    if not caption:
                        skipped_empty_captions += 1
                        continue

                    json.dump(
                        {"image": str(output_image_path.resolve()), "caption": caption},
                        handle,
                        ensure_ascii=False,
                    )
                    handle.write("\n")
                    written_rows += 1

        print(
            f"Wrote {output_jsonl_path} (rows={written_rows}, skipped_empty_captions={skipped_empty_captions})"
        )


if __name__ == "__main__":
    main()
