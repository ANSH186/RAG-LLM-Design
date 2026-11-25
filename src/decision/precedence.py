"""The official decision precedence.

The numeric order is executable: the first applicable rule wins.
"""
from __future__ import annotations

from enum import Enum


class RuleName(str, Enum):
    MISSING_INFORMATION = "MISSING_INFORMATION"
    BLACKLISTED_SUPPLIER = "BLACKLISTED_SUPPLIER"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    BANNED_TERM = "BANNED_TERM"
    PRICE_DECISION = "PRICE_DECISION"
    APPROVAL_LIMIT = "APPROVAL_LIMIT"


RULE_ORDER: tuple[RuleName, ...] = (
    RuleName.MISSING_INFORMATION,
    RuleName.BLACKLISTED_SUPPLIER,
    RuleName.PROMPT_INJECTION,
    RuleName.BANNED_TERM,
    RuleName.PRICE_DECISION,
    RuleName.APPROVAL_LIMIT,
)

RULE_PRECEDENCE = {rule: index + 1 for index, rule in enumerate(RULE_ORDER)}


class DecisionAction(str, Enum):
    ACCEPT = "ACCEPT"
    COUNTER = "COUNTER"
    REJECT = "REJECT"
    ESCALATE = "ESCALATE"


class DecisionReason(str, Enum):
    ACCEPTABLE_PRICE = "ACCEPTABLE_PRICE"
    COUNTER_OFFER = "COUNTER_OFFER"
    ABOVE_WALKAWAY = "ABOVE_WALKAWAY"
    BLACKLISTED_SUPPLIER = "BLACKLISTED_SUPPLIER"
    MISSING_INFO = "MISSING_INFO"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    BANNED_TERM = "BANNED_TERM"
    OVER_APPROVAL_LIMIT = "OVER_APPROVAL_LIMIT"


ACTION_FOR_REASON = {
    DecisionReason.ACCEPTABLE_PRICE: DecisionAction.ACCEPT,
    DecisionReason.COUNTER_OFFER: DecisionAction.COUNTER,
    DecisionReason.ABOVE_WALKAWAY: DecisionAction.REJECT,
    DecisionReason.BLACKLISTED_SUPPLIER: DecisionAction.REJECT,
    DecisionReason.MISSING_INFO: DecisionAction.ESCALATE,
    DecisionReason.PROMPT_INJECTION: DecisionAction.ESCALATE,
    DecisionReason.BANNED_TERM: DecisionAction.ESCALATE,
    DecisionReason.OVER_APPROVAL_LIMIT: DecisionAction.ESCALATE,
}
