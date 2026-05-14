"""Backward-compatible instruction-data entrypoint.

Prefer dataset-specific scripts:
  - scripts/prepare_instruction_viet_sharegpt.py
  - scripts/prepare_instruction_5cd_localization.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_instruction_common import DEFAULT_CONFIG_SECTION, main, parse_args  # noqa: E402

__all__ = ["DEFAULT_CONFIG_SECTION", "main", "parse_args"]


if __name__ == "__main__":
    main(default_config_section=DEFAULT_CONFIG_SECTION)
