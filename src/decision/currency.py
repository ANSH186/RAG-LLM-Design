"""Currency conversion driven exclusively by the resolved policy snapshot."""
from __future__ import annotations

from decimal import Decimal

from src.knowledge.policy_snapshot import PolicySnapshot


def rate_for(currency: str, snapshot: PolicySnapshot) -> Decimal:
    code = currency.strip().upper()
    try:
        return snapshot.effective_rate(code)
    except KeyError as exc:
        raise ValueError(f"no official exchange rate for {code}") from exc


def to_inr(amount: Decimal, currency: str, snapshot: PolicySnapshot) -> Decimal:
    return amount * rate_for(currency, snapshot)


def from_inr(amount_inr: Decimal, currency: str, snapshot: PolicySnapshot) -> Decimal:
    rate = rate_for(currency, snapshot)
    if rate <= 0:
        raise ValueError("exchange rate must be positive")
    return amount_inr / rate


def supported_currencies(snapshot: PolicySnapshot) -> set[str]:
    return {"INR", *snapshot.currency_rates.keys(), *snapshot.supplier_exchange_overrides.keys()}
