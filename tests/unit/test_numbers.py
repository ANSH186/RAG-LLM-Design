from decimal import Decimal

import pytest

from src.utils.numbers import (
    parse_decimal_number,
    round_half_up,
    words_to_decimal,
)


@pytest.mark.parametrize(
    ("raw", "suffix", "expected"),
    [
        ("44,07,800", None, Decimal(4407800)),
        ("1,104.94", None, Decimal("1104.94")),
        ("58.3", "k", Decimal("58300.0")),
        ("1.2", "lakh", Decimal("120000.0")),
        ("118.5", "k", Decimal("118500.0")),
        ("1,10,000", None, Decimal(110000)),
    ],
)
def test_numeric_parsing(raw, suffix, expected):
    assert parse_decimal_number(raw, suffix) == expected


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        ("two hundred sixty-six", Decimal(266)),
        ("one thousand six hundred", Decimal(1600)),
        ("twenty-five thousand two hundred", Decimal(25200)),
        ("one lakh seven thousand three hundred", Decimal(107300)),
        ("three thousand four hundred", Decimal(3400)),
        ("one lakh thirty-six thousand seven hundred", Decimal(136700)),
    ],
)
def test_word_numbers(words, expected):
    assert words_to_decimal(words) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Decimal("2.5"), 3),
        (Decimal("2.49"), 2),
        (Decimal("0.5"), 1),
        (Decimal("62888.888888"), 62889),
        (Decimal("85000.000"), 85000),
    ],
)
def test_round_half_up(value, expected):
    assert round_half_up(value) == expected


def test_invalid_numbers_rejected():
    with pytest.raises(Exception):
        parse_decimal_number("12,34,56,789.1.2")
    with pytest.raises(Exception):
        words_to_decimal("many thousands")
