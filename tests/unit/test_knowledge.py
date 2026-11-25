from decimal import Decimal
from pathlib import Path

import pytest

from src.errors import KnowledgeIngestionError
from src.knowledge.ingestion import verify_source_hashes
from src.knowledge.policy_compiler import compile_knowledge
from src.knowledge.validation import assert_mandatory_rule_fragments

ROOT = Path(__file__).resolve().parents[2]


def test_compiled_counts_and_core_values(knowledge):
    assert len(knowledge.categories) == 15
    assert len(knowledge.suppliers) == 40
    assert len(knowledge.amendments) == 9
    assert len(knowledge.precedents) == 8
    assert len(knowledge.documents) == 80
    catering = knowledge.categories["catering"]
    assert catering.target_price == 280
    assert catering.walkaway_price == 340
    assert catering.gst_rate == Decimal("0.05")
    assert catering.negotiation_rounds == 3
    assert catering.approval_limit == 600000
    assert knowledge.categories["led-lighting"].subunit_factors["pack"] == 4
    assert knowledge.categories["printer-paper"].subunit_factors["ream"] == Decimal("0.1")
    assert knowledge.currency_rates["USD"].inr_per_unit == 84.5
    assert knowledge.currency_rates["EUR"].inr_per_unit == 91
    assert knowledge.suppliers["greenleaf-facility-care"].status.value == "BLACKLISTED"
    assert knowledge.suppliers["banyan-guarding"].status.value == "PROBATION"
    assert knowledge.suppliers["ivory-events-catering"].price_caps[0].category_slug == "headsets"
    assert knowledge.suppliers["arvind-office-supplies"].exchange_rate_overrides[0].inr_per_unit == 104.5


def test_source_hashes_and_rule_fragments(knowledge):
    assert verify_source_hashes(knowledge, ROOT) == []
    assert_mandatory_rule_fragments(knowledge)


def test_missing_category_field_fails_loudly(tmp_path):
    root = tmp_path / "playbook"
    (root / "categories").mkdir(parents=True)
    (root / "categories" / "broken.md").write_text(
        "# Category: Broken\n\n- **Normalized unit:** INR excl. GST per unit\n"
        "- **GST rate:** 18%\n- **Target price:** ₹10\n",
        encoding="utf-8",
    )
    with pytest.raises(KnowledgeIngestionError, match="missing category fields"):
        compile_knowledge(knowledge_root=root, environment="broken")


def test_unknown_contract_category_fails(tmp_path):
    root = tmp_path / "playbook"
    (root / "categories").mkdir(parents=True)
    (root / "suppliers").mkdir(parents=True)
    (root / "categories" / "widgets.md").write_text(
        "# Category: Widgets\n\n- **Normalized unit:** INR excl. GST per widget\n"
        "- **GST rate:** 18%\n- **Target price:** ₹100\n- **Walk-away price:** ₹120\n"
        "- **Negotiation rounds (R):** 3\n- **Concession exponent (β):** 1\n"
        "- **Approval limit:** ₹1,00,000\n",
        encoding="utf-8",
    )
    (root / "suppliers" / "supplier.md").write_text(
        "# Supplier: Supplier\n\n- **Status:** Approved\n"
        "- **Contract:** price cap for Unknown of ₹100 per widget (replaces the walk-away price)\n",
        encoding="utf-8",
    )
    with pytest.raises(KnowledgeIngestionError, match="unknown category title"):
        compile_knowledge(knowledge_root=root, environment="broken")


def test_validation_rejects_missing_policy(tmp_path):
    root = tmp_path / "playbook"
    (root / "categories").mkdir(parents=True)
    (root / "suppliers").mkdir(parents=True)
    (root / "categories" / "widgets.md").write_text(
        "# Category: Widgets\n\n- **Normalized unit:** INR excl. GST per widget\n"
        "- **GST rate:** 18%\n- **Target price:** ₹100\n- **Walk-away price:** ₹120\n"
        "- **Negotiation rounds (R):** 3\n- **Concession exponent (β):** 1\n"
        "- **Approval limit:** ₹1,00,000\n",
        encoding="utf-8",
    )
    (root / "suppliers" / "supplier.md").write_text(
        "# Supplier: Supplier\n\n- **Status:** Approved\n", encoding="utf-8"
    )
    with pytest.raises(KnowledgeIngestionError, match="missing mandatory policy documents"):
        compile_knowledge(knowledge_root=root, environment="broken")
