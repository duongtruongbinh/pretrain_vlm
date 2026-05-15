"""Data preparation entry point.

Usage:
    python scripts/prepare_data.py uit
    python scripts/prepare_data.py coco [--config-section prepare_coco] [--output-dir ...] [--overwrite]
    python scripts/prepare_data.py sharegpt [--config-section instruction_data_gpt]
    python scripts/prepare_data.py 5cd [--config-section instruction_data_5cd]
    python scripts/prepare_data.py vietnamtourism-crawl
    python scripts/prepare_data.py vietnamtourism-qa
    python scripts/prepare_data.py vietnamtourism-prepare
"""

from __future__ import annotations

import argparse


def cmd_uit(_args: argparse.Namespace) -> None:
    from prepare_uit_openviic import main
    main()


def cmd_coco(args: argparse.Namespace) -> None:
    from prepare_coco_data import run
    run(args)


def cmd_sharegpt(args: argparse.Namespace) -> None:
    from prepare_instruction_common import run
    run(args.config_section)


def cmd_5cd(args: argparse.Namespace) -> None:
    from prepare_instruction_common import run
    run(args.config_section)


def cmd_vietnamtourism_crawl(_args: argparse.Namespace) -> None:
    from crawl_vietnamtourism import main
    main()


def cmd_vietnamtourism_qa(_args: argparse.Namespace) -> None:
    from generate_qa_vietnamtourism import main
    main()


def cmd_vietnamtourism_prepare(_args: argparse.Namespace) -> None:
    from prepare_vietnamtourism_data import main
    main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare training data.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    subparsers.add_parser("uit", help="Download and prepare UIT-OpenViIC captions (Stage 1).")

    coco_parser = subparsers.add_parser("coco", help="Prepare COCO 2017 Vietnamese captions (Stage 1).")
    coco_parser.add_argument("--config-section", default="prepare_coco")
    coco_parser.add_argument("--output-dir", default=None)
    coco_parser.add_argument("--max-rows-per-split", type=int, default=None)
    coco_parser.add_argument("--overwrite", action="store_true")
    coco_parser.add_argument("--inspect-only", action="store_true")

    sharegpt_parser = subparsers.add_parser(
        "sharegpt", help="Prepare Viet-ShareGPT-4o instruction data (Stage 2)."
    )
    sharegpt_parser.add_argument("--config-section", default="instruction_data_gpt")

    fivecd_parser = subparsers.add_parser(
        "5cd", help="Prepare 5CD-AI Viet-Localization-VQA data (Stage 2, requires HF approval)."
    )
    fivecd_parser.add_argument("--config-section", default="instruction_data_5cd")

    subparsers.add_parser(
        "vietnamtourism-crawl", help="Crawl images from vietnamtourism.gov.vn."
    )
    subparsers.add_parser(
        "vietnamtourism-qa",
        help="Generate QA conversations via OpenAI Batch API (requires OPENAI_API_KEY).",
    )
    subparsers.add_parser(
        "vietnamtourism-prepare", help="Convert crawled+generated data to instruction JSONL."
    )

    args = parser.parse_args()
    dispatch = {
        "uit": cmd_uit,
        "coco": cmd_coco,
        "sharegpt": cmd_sharegpt,
        "5cd": cmd_5cd,
        "vietnamtourism-crawl": cmd_vietnamtourism_crawl,
        "vietnamtourism-qa": cmd_vietnamtourism_qa,
        "vietnamtourism-prepare": cmd_vietnamtourism_prepare,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
