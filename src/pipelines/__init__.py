"""
The pipelines, one function each, so they can be called from another file.

The file names start with a number (the run order), so they are loaded by name here.
"""

from importlib import import_module

prepare_medical = import_module(".0_prepare_medical", __package__).prepare_medical
prepare_financial = import_module(".0_prepare_financial", __package__).prepare_financial
tune_medical = import_module(".1_tune_medical", __package__).tune_medical
tune_financial = import_module(".1_tune_financial", __package__).tune_financial
benchmark_medical = import_module(".2_benchmark_medical", __package__).benchmark_medical
benchmark_financial = import_module(
    ".2_benchmark_financial", __package__
).benchmark_financial
analyze = import_module(".3_analyze", __package__).analyze
