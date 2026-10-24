#!/usr/bin/env python3
"""
Shared console output: a two level title, a settings string and a table.

Every pipeline prints progress as log lines while it works and a table at the end.
"""

import numpy as np
import pandas as pd

# True when the last printed line was empty, so no gap is printed twice
_at_blank = True


def _gap() -> None:
    global _at_blank
    if not _at_blank:
        print()
    _at_blank = True


def h1(text: str) -> None:
    """Top title, upper case, between two rules."""
    text = f" {text.upper()} "
    rule = "=" * len(text)
    _gap()
    print(f"{rule}\n{text}\n{rule}\n")


SMALL_WORDS = {"a", "an", "and", "by", "in", "of", "on", "or", "per", "the", "to"}


def title_case(text: str) -> str:
    """Capitalize each word except small ones, and keep words like SQ1 or IBS as is."""
    words, out = text.split(" "), []
    for i, word in enumerate(words):
        starts_part = i == 0 or words[i - 1].endswith(":")
        if word != word.lower():
            out.append(word)
        elif word in SMALL_WORDS and not starts_part:
            out.append(word)
        else:
            out.append(word[:1].upper() + word[1:])
    return " ".join(out)


def h2(text: str) -> None:
    """Sub title in title case, with a rule under it."""
    text = f" {title_case(text)} "
    _gap()
    print(f"{text}\n{'-' * len(text)}\n")


def log(text: str) -> None:
    """One info line, marked with a star so it stands out from tables."""
    global _at_blank
    _at_blank = False
    print(f"* {text}", flush=True)


def settings(params: dict, sep: str = ", ") -> str:
    """The key=value pairs joined by sep, with no spaces inside a value."""
    return sep.join(f"{k}={str(v).replace(' ', '')}" for k, v in params.items())


def _cell(value, decimals: int) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{decimals}f}"
    return str(value)


def table(df: pd.DataFrame, decimals: int = 5, right: tuple = ()) -> None:
    """Print a frame as an aligned table. A cell with line breaks spans several lines."""
    cols = list(df.columns)
    cells = [[_cell(v, decimals) for v in row] for row in df.itertuples(index=False)]
    widths = [
        max([len(str(c))] + [len(x) for r in cells for x in r[i].split("\n")])
        for i, c in enumerate(cols)
    ]
    align_right = [pd.api.types.is_numeric_dtype(df[c]) or c in right for c in cols]

    def fmt(parts):
        return "  ".join(
            p.rjust(w) if r else p.ljust(w)
            for p, w, r in zip(parts, widths, align_right)
        ).rstrip()

    def fmt_row(row):
        split = [c.split("\n") for c in row]
        height = max(len(c) for c in split)
        return [fmt([c[i] if i < len(c) else "" for c in split]) for i in range(height)]

    _gap()
    head = fmt([str(c) for c in cols])
    body = [fmt_row(row) for row in cells]
    width = max([len(head)] + [len(x) for lines in body for x in lines])
    print(head)
    print("=" * width)
    for lines in body:
        print("\n".join(lines))
        print("-" * width)
    print()
