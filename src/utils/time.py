"""Date parsing and policy-date comparisons."""
from __future__ import annotations

from datetime import date, datetime


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value.strip(), "%Y-%m-%d").date()


def is_effective(effective_from: str | date | None, negotiation_date: str | date) -> bool:
    if effective_from is None:
        return True
    start = parse_iso_date(effective_from) if isinstance(effective_from, str) else effective_from
    as_of = parse_iso_date(negotiation_date) if isinstance(negotiation_date, str) else negotiation_date
    return start <= as_of
