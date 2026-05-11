"""Prepare Viet-ShareGPT-4o-Text-VQA for instruction tuning."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_instruction_common import main  # noqa: E402


DEFAULT_CONFIG_SECTION = "instruction_data_gpt"


if __name__ == "__main__":
    main(default_config_section=DEFAULT_CONFIG_SECTION)
