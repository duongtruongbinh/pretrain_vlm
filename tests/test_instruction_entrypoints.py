import importlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


class InstructionEntrypointTest(unittest.TestCase):
    def test_dataset_specific_entrypoints_have_separate_config_sections(self):
        gpt = importlib.import_module("scripts.prepare_instruction_viet_sharegpt")
        five_cd = importlib.import_module("scripts.prepare_instruction_5cd_localization")

        self.assertEqual(gpt.DEFAULT_CONFIG_SECTION, "instruction_data_gpt")
        self.assertEqual(five_cd.DEFAULT_CONFIG_SECTION, "instruction_data_5cd")

    def test_generic_entrypoint_keeps_config_section_override(self):
        generic = importlib.import_module("scripts.prepare_instruction_data")

        args = generic.parse_args(["--config-section", "instruction_data_5cd"])
        self.assertEqual(args.config_section, "instruction_data_5cd")

    def test_streamed_cmyk_images_are_saved_as_rgb_jpeg(self):
        common = importlib.import_module("scripts.prepare_instruction_common")
        image = Image.new("CMYK", (8, 8), color=(0, 128, 128, 0))

        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "sample.jpg"
            common.save_image_asset(image, destination)

            with Image.open(destination) as saved:
                self.assertEqual(saved.mode, "RGB")
                self.assertEqual(saved.format, "JPEG")

    def test_streamed_pil_images_use_jpeg_extension(self):
        common = importlib.import_module("scripts.prepare_instruction_common")
        image = Image.new("CMYK", (8, 8), color=(0, 128, 128, 0))

        self.assertEqual(common.infer_image_extension(image, image_key="sample"), ".jpg")

    def test_instruction_streamlit_prefers_checkpoint_llm_and_tokenizer_sources(self):
        app = importlib.import_module("streamlit_instruction_test")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint-10"
            (checkpoint / "llm").mkdir(parents=True)
            (checkpoint / "tokenizer").mkdir()

            llm_source, tokenizer_source = app.resolve_checkpoint_sources(checkpoint, "base-llm")

            self.assertEqual(llm_source, str((checkpoint / "llm").resolve()))
            self.assertEqual(tokenizer_source, str((checkpoint / "tokenizer").resolve()))

    def test_instruction_dataset_reports_jsonl_paths_when_all_images_are_unreadable(self):
        data = importlib.import_module("src.data")

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "samples.jsonl"
            record = {
                "id": "bad-image",
                "image": "missing.jpg",
                "messages": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Ảnh có gì?"},
                    {"role": "assistant", "content": "Không đọc được ảnh."},
                ],
            }
            jsonl_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = data.ImageInstructionDataset(jsonl_path)

            with self.assertRaisesRegex(RuntimeError, "samples.jsonl"):
                dataset[0]


if __name__ == "__main__":
    unittest.main()
