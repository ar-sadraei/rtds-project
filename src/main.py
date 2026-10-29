#!/usr/bin/env python3
"""
Run the whole benchmark, one pipeline after the other.

Each pipeline prints its own progress and writes its own log file in logs/. Remove a
call to skip a step.

    venv/bin/python src/main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipelines import (
    prepare_medical,
    prepare_financial,
    tune_medical,
    tune_financial,
    benchmark_medical,
    benchmark_financial,
    analyze,
    summarize,
)


def main() -> None:
    prepare_medical()
    prepare_financial()
    tune_medical()
    tune_financial()
    benchmark_medical()
    benchmark_financial()
    analyze()
    summarize()


if __name__ == "__main__":
    main()
