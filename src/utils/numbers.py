"""Deterministic number parsing used by both extraction and the financial engine.

Supports Indian and Western digit grouping, decimal amounts, lakh/k suffixes, and
English number words. All results are Decimal; floats are never used for money.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
SMALL_SCALE_WORDS = {"hundred": 100}
LARGE_SCALE_WORDS = {
    "thousand": 1_000,
    "lakh": 100_000,
    "lakhs": 100_000,
    "lac": 100_000,
    "lacs": 100_000,
    "million": 1_000_000,
    "crore": 10_000_000,
    "crores": 10_000_000,
    "billion": 1_000_000_000,
}
ALL_WORD_NUMBERS = set(NUMBER_WORDS) | set(SMALL_SCALE_WORDS) | set(LARGE_SCALE_WORDS) | {"and"}

SUFFIX_MULTIPLIERS = {
    "k": Decimal(1_000),
    "thousand": Decimal(1_000),
    "lakh": Decimal(100_000),
    "lakhs": Decimal(100_000),
    "lac": Decimal(100_000),
    "lacs": Decimal(100_000),
    "million": Decimal(1_000_000),
    "crore": Decimal(10_000_000),
    "crores": Decimal(10_000_000),
    "cr": Decimal(10_000_000),
}

_NUM = r"(?:\d{1,3}(?:,\d{2,3})*|\d+)(?:\.\d+)?"


def parse_decimal_number(raw: str, suffix: str | None = None) -> Decimal:
    """Parse a numeric token such as '44,07,800', '1,104.94', '58.3'."""
    text = raw.strip().replace(" ", "")
    if not text:
        raise InvalidOperation("empty number")
    if "," in text:
        # Accept both Indian and Western grouping; groups are validated loosely.
        if not re.fullmatch(r"\d{1,3}(?:,\d{2,3})*(?:\.\d+)?", text):
            raise InvalidOperation(f"invalid grouped number: {raw!r}")
        text = text.replace(",", "")
    if not re.fullmatch(r"\d+(?:\.\d+)?", text):
        raise InvalidOperation(f"invalid number: {raw!r}")
    value = Decimal(text)
    if suffix:
        key = suffix.strip().lower()
        if key in SUFFIX_MULTIPLIERS:
            value *= SUFFIX_MULTIPLIERS[key]
        else:
            raise InvalidOperation(f"unknown numeric suffix: {suffix!r}")
    return value


def words_to_decimal(text: str) -> Decimal:
    """Convert an English number phrase to Decimal.

    Examples:
        'two hundred sixty-six' -> 266
        'one lakh seven thousand three hundred' -> 107300
        'twenty-five thousand two hundred' -> 25200
    """
    cleaned = re.sub(r"[-,]", " ", text.lower())
    tokens = [t for t in cleaned.split() if t and t != "and"]
    if not tokens:
        raise ValueError("empty number phrase")
    total = 0
    current = 0
    saw_number = False
    for token in tokens:
        if token in NUMBER_WORDS:
            current += NUMBER_WORDS[token]
            saw_number = True
        elif token in SMALL_SCALE_WORDS:
            current = max(current, 1) * SMALL_SCALE_WORDS[token]
            saw_number = True
        elif token in LARGE_SCALE_WORDS:
            if current == 0:
                current = 1
            total += current * LARGE_SCALE_WORDS[token]
            current = 0
            saw_number = True
        else:
            raise ValueError(f"not a number word: {token!r}")
    if not saw_number:
        raise ValueError(f"no number words in {text!r}")
    return Decimal(total + current)


def round_half_up(value: Decimal) -> int:
    """Round to the nearest whole rupee, x.5 up, without binary floating point."""
    from decimal import ROUND_HALF_UP

    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def format_inr(value: Decimal | None) -> str | None:
    if value is None:
        return None
    rounded = value.quantize(Decimal("0.01"))
    text = f"{rounded:,.2f}"
    return text.rstrip("0").rstrip(".")


def parse_percentage(raw: str) -> Decimal:
    return parse_decimal_number(raw)


def normalise_rupee_text(text: str) -> str:
    """Normalise unicode rupee variants without altering the amount."""
    return text.replace("रु", "₹").replace("₨", "₹").replace("Rs.", "Rs. ")


def extract_first_number(text: str) -> Decimal | None:
    match = re.search(_NUM, text)
    if not match:
        return None
    try:
        return parse_decimal_number(match.group(0))
    except InvalidOperation:
        return None
