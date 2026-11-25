"""GST arithmetic. Normalized comparison is always INR excluding GST."""
from __future__ import annotations

from decimal import Decimal


def gst_factor(rate: Decimal) -> Decimal:
    if not (Decimal(0) <= rate < Decimal(1)):
        raise ValueError(f"invalid GST rate: {rate}")
    return Decimal(1) + rate


def exclude_gst(amount_inclusive: Decimal, rate: Decimal) -> Decimal:
    return amount_inclusive / gst_factor(rate)


def include_gst(amount_exclusive: Decimal, rate: Decimal) -> Decimal:
    return amount_exclusive * gst_factor(rate)
