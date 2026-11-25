from decimal import Decimal

import pytest

from src.errors import ExtractionError, LLMProviderError
from src.extraction.extractor import ExtractionOrchestrator
from src.extraction.llm_extractor import LLMExtractor
from src.extraction.normalizer import OfferNormalizer
from src.llm.cache import LLMCache
from src.llm.provider import LLMConfig
from tests.helpers import snapshot_for

SNAPSHOT = None


class FakeProvider:
    name = "fake"

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0

    def complete_json(self, *, system_prompt, user_prompt, schema):
        self.calls += 1
        if self.error:
            raise self.error
        return self.response


def valid_response(source_text, price="3000", currency="INR"):
    return {
        "usable_price": True,
        "offered_price": price,
        "currency": currency,
        "is_total": False,
        "total_quantity": None,
        "unit": "headset",
        "billing_period": None,
        "gst_included": False,
        "discount_percent": None,
        "discount_base": None,
        "list_price": None,
        "is_correction": False,
        "superseded_price": None,
        "free_units": None,
        "paid_units": None,
        "volume_discount_percent": None,
        "volume_threshold": None,
        "tier_threshold": None,
        "tier_prices": [],
        "ambiguities": [],
        "injection_signals": [],
        "source_text": source_text,
        "confidence": 0.95,
    }


def test_llm_extraction_valid_and_grounded(knowledge, tmp_path):
    message = "For the 10 headsets you requested, we can do ₹3,000 per headset."
    snapshot = snapshot_for(knowledge, category="headsets", supplier="Monsoon Foods", date="2026-11-10")
    provider = FakeProvider(valid_response(message))
    extractor = LLMExtractor(
        provider=provider,
        config=LLMConfig(provider="fake", base_url=None, api_key=None, model="fake"),
        cache=LLMCache(tmp_path),
    )
    extraction = extractor.extract(thread_id="T", source_message=message, snapshot=snapshot)
    assert extraction.extraction_method.value == "llm"
    assert extraction.selected_candidate.amount == Decimal(3000)
    normalized = OfferNormalizer(snapshot).normalize(extraction, quantity=10)
    assert normalized.normalized_price_inr_excl_gst == Decimal(3000)


def test_llm_schema_validation_failure(knowledge, tmp_path):
    message = "For the 10 headsets you requested, we can do ₹3,000 per headset."
    snapshot = snapshot_for(knowledge, category="headsets", supplier="Monsoon Foods", date="2026-11-10")
    response = valid_response(message)
    response["offered_price"] = "not-a-number"
    extractor = LLMExtractor(
        provider=FakeProvider(response),
        config=LLMConfig(provider="fake", base_url=None, api_key=None, model="fake"),
        cache=LLMCache(tmp_path),
    )
    with pytest.raises(ExtractionError):
        extractor.extract(thread_id="T", source_message=message, snapshot=snapshot)


def test_llm_ungrounded_source_is_rejected(knowledge, tmp_path):
    message = "For the 10 headsets you requested, we can do ₹3,000 per headset."
    snapshot = snapshot_for(knowledge, category="headsets", supplier="Monsoon Foods", date="2026-11-10")
    extractor = LLMExtractor(
        provider=FakeProvider(valid_response("A sentence that is not in the thread.")),
        config=LLMConfig(provider="fake", base_url=None, api_key=None, model="fake"),
        cache=LLMCache(tmp_path),
    )
    with pytest.raises(ExtractionError, match="source_text"):
        extractor.extract(thread_id="T", source_message=message, snapshot=snapshot)


def test_llm_invented_price_is_rejected(knowledge, tmp_path):
    message = "For the 10 headsets you requested, we can do ₹3,000 per headset."
    snapshot = snapshot_for(knowledge, category="headsets", supplier="Monsoon Foods", date="2026-11-10")
    extractor = LLMExtractor(
        provider=FakeProvider(valid_response(message, price="9999")),
        config=LLMConfig(provider="fake", base_url=None, api_key=None, model="fake"),
        cache=LLMCache(tmp_path),
    )
    with pytest.raises(ExtractionError, match="not present"):
        extractor.extract(thread_id="T", source_message=message, snapshot=snapshot)


def test_llm_currency_mismatch_is_rejected(knowledge, tmp_path):
    message = "For the 10 headsets you requested, we can do ₹3,000 per headset."
    snapshot = snapshot_for(knowledge, category="headsets", supplier="Monsoon Foods", date="2026-11-10")
    extractor = LLMExtractor(
        provider=FakeProvider(valid_response(message, currency="USD")),
        config=LLMConfig(provider="fake", base_url=None, api_key=None, model="fake"),
        cache=LLMCache(tmp_path),
    )
    with pytest.raises(ExtractionError, match="currency"):
        extractor.extract(thread_id="T", source_message=message, snapshot=snapshot)


def test_llm_cache_prevents_duplicate_call(knowledge, tmp_path):
    message = "For the 10 headsets you requested, we can do ₹3,000 per headset."
    snapshot = snapshot_for(knowledge, category="headsets", supplier="Monsoon Foods", date="2026-11-10")
    provider = FakeProvider(valid_response(message))
    extractor = LLMExtractor(
        provider=provider,
        config=LLMConfig(provider="fake", base_url=None, api_key=None, model="fake"),
        cache=LLMCache(tmp_path),
    )
    extractor.extract(thread_id="T", source_message=message, snapshot=snapshot)
    extractor.extract(thread_id="T", source_message=message, snapshot=snapshot)
    assert provider.calls == 1


def test_orchestrator_falls_back_without_provider(knowledge):
    message = "For the 10 headsets you requested, we can do ₹3,000 per headset."
    snapshot = snapshot_for(knowledge, category="headsets", supplier="Monsoon Foods", date="2026-11-10")
    orchestrator = ExtractionOrchestrator(
        category_factor_resolver=lambda label: knowledge.categories["headsets"].subunit_factors.get(label, Decimal(1)),
        mode="llm",
        llm_config=LLMConfig(provider="none", base_url=None, api_key=None, model=None),
    )
    outcome = orchestrator.extract(thread_id="T", source_message=message, snapshot=snapshot)
    assert outcome.fallback_used
    assert outcome.used_method.value == "deterministic"


def test_orchestrator_falls_back_on_provider_error(knowledge, tmp_path):
    message = "For the 10 headsets you requested, we can do ₹3,000 per headset."
    snapshot = snapshot_for(knowledge, category="headsets", supplier="Monsoon Foods", date="2026-11-10")
    provider = FakeProvider(error=LLMProviderError("timeout"))
    orchestrator = ExtractionOrchestrator(
        category_factor_resolver=lambda label: knowledge.categories["headsets"].subunit_factors.get(label, Decimal(1)),
        mode="llm",
        provider=provider,
        llm_config=LLMConfig(provider="fake", base_url="x", api_key="x", model="fake"),
    )
    outcome = orchestrator.extract(thread_id="T", source_message=message, snapshot=snapshot)
    assert outcome.fallback_used
    assert outcome.used_method.value == "deterministic"


def test_auto_mode_does_not_call_llm_for_complete_deterministic_parse(knowledge):
    message = "For the 10 headsets you requested, we can do ₹3,000 per headset."
    snapshot = snapshot_for(knowledge, category="headsets", supplier="Monsoon Foods", date="2026-11-10")
    provider = FakeProvider(valid_response(message))
    orchestrator = ExtractionOrchestrator(
        category_factor_resolver=lambda label: knowledge.categories["headsets"].subunit_factors.get(label, Decimal(1)),
        mode="auto",
        provider=provider,
        llm_config=LLMConfig(provider="fake", base_url="x", api_key="x", model="fake"),
    )
    outcome = orchestrator.extract(thread_id="T", source_message=message, snapshot=snapshot)
    assert provider.calls == 0
    assert outcome.used_method.value == "deterministic"
