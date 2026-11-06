"""Inspect the pipeline architecture from a terminal."""

import argparse
from collections.abc import Sequence

from neurovision.core.errors import ConfigurationError
from neurovision.core.logging import create_cli_logger
from neurovision.core.settings import LogLevel, Settings
from neurovision.pipeline import RESEARCH_NOTICE, Stage


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List the NeuroVision Intelligence Pipeline stages.",
        epilog=RESEARCH_NOTICE,
    )
    parser.parse_args(argv)
    try:
        settings = Settings.from_env()
    except ConfigurationError as error:
        create_cli_logger(LogLevel.ERROR).error("%s", error)
        return 2

    create_cli_logger(settings.log_level).info("pipeline_catalog_listed")
    print(RESEARCH_NOTICE)
    print("NeuroVision Intelligence Pipeline")
    for number, stage in enumerate(Stage, start=1):
        print(f"{number:02d}. {stage.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
