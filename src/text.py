"""Strict and normalized views of a transcription."""

from __future__ import annotations

import re

QUOTE_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
    }
)


# What: Checks whether a string has no characters.
# Needs: A string.
# Returns: True when the string is empty. Spaces and newlines still count as text.
def is_empty(text: str) -> bool:
    return text == ""


# What: Prepares a transcription for the strict character and word rates.
# Needs: Raw transcription text.
# Returns: The same spelling, case, and line breaks, with unified endings and no edge newlines.
def strict_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")


# What: Prepares a transcription for the normalized character and word rates.
# Needs: Raw transcription text.
# Returns: Folded case, curly quotes, dashes, and extra spaces. Spelling and line breaks stay.
def normalized_text(text: str) -> str:
    text = strict_text(text).translate(QUOTE_MAP).casefold()
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# What: Collapses whitespace so a line break separates words.
# Needs: Text that is already strict or already normalized.
# Returns: The same words, with every run of whitespace turned into one space.
def for_words(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# What: Builds the four strings the error rates are computed on.
# Needs: Raw transcription text.
# Returns: strict_cer, strict_wer, normalized_cer, and normalized_wer.
def scoring_views(text: str) -> dict[str, str]:
    strict = strict_text(text)
    normalized = normalized_text(text)
    return {
        "strict_cer": strict,
        "strict_wer": for_words(strict),
        "normalized_cer": normalized,
        "normalized_wer": for_words(normalized),
    }
