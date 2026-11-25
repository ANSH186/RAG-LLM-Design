"""Typed application errors.

Failures are never swallowed: callers either recover explicitly or route to a safe,
auditable escalation. Monetary errors are distinguished from infrastructure errors.
"""
from __future__ import annotations


class ProcurementError(Exception):
    """Base class for all application errors."""


class KnowledgeIngestionError(ProcurementError):
    """A source document could not be parsed faithfully."""


class KnowledgeValidationError(ProcurementError):
    """The compiled knowledge base violates a structural invariant."""


class TemporalResolutionError(ProcurementError):
    """A valid policy snapshot cannot be constructed for a negotiation."""


class ExtractionError(ProcurementError):
    """The supplier offer could not be interpreted."""


class UnsupportedCurrencyError(ExtractionError):
    """The offer uses a currency with no official conversion rate."""


class MissingInformationError(ExtractionError):
    """The latest supplier offer lacks mandatory information."""


class DecisionError(ProcurementError):
    """The deterministic decision procedure cannot be applied."""


class VerificationError(ProcurementError):
    """Independent verification failed."""


class LLMProviderError(ProcurementError):
    """The configured language-model provider failed."""
