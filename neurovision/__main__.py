"""Inspect the pipeline architecture from a terminal."""

import argparse
from collections.abc import Sequence

from neurovision.pipeline import RESEARCH_NOTICE, Stage


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List the NeuroVision Intelligence Pipeline stages.",
        epilog=RESEARCH_NOTICE,
    )
    parser.parse_args(argv)
    print(RESEARCH_NOTICE)
    print("NeuroVision Intelligence Pipeline")
    for number, stage in enumerate(Stage, start=1):
        print(f"{number:02d}. {stage.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
