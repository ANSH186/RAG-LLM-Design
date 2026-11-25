import pytest

from src.errors import TemporalResolutionError
from src.knowledge.schema import SupplierStatus
from src.knowledge.temporal_resolver import resolve_policy_snapshot


def snapshot(knowledge, category, supplier, date, thread_id="T"):
    return resolve_policy_snapshot(
        knowledge,
        thread_id=thread_id,
        category_slug=category,
        supplier_name=supplier,
        negotiation_date=date,
    )


def test_amendment_effective_on_date_and_after(knowledge):
    before = snapshot(knowledge, "courier", "Yamuna Traders", "2026-09-21")
    on_date = snapshot(knowledge, "courier", "Yamuna Traders", "2026-09-22")
    after = snapshot(knowledge, "courier", "Yamuna Traders", "2026-09-30")
    assert before.walkaway_price == 200
    assert on_date.walkaway_price == 210
    assert after.walkaway_price == 210
    assert before.amendments_applied == ()
    assert "AMD-01" in on_date.amendments_applied


def test_multiple_amendments_latest_effective_wins(knowledge):
    dec3 = snapshot(knowledge, "led-lighting", "Everbright Lighting", "2026-12-03")
    dec4 = snapshot(knowledge, "led-lighting", "Everbright Lighting", "2026-12-04")
    dec5 = snapshot(knowledge, "led-lighting", "Everbright Lighting", "2026-12-05")
    assert dec3.walkaway_price == 2200
    assert dec4.walkaway_price == 2330
    assert dec5.walkaway_price == 2360
    assert dec4.amendments_applied == ("AMD-08",)
    assert dec5.amendments_applied == ("AMD-08", "AMD-07")


def test_supplier_cap_overrides_amended_walkaway(knowledge):
    snap = snapshot(knowledge, "led-lighting", "Xenon Displays", "2026-12-10")
    assert snap.supplier_price_cap == 2300
    assert snap.walkaway_price == 2300
    assert "AMD-08" in snap.amendments_applied
    assert "AMD-07" in snap.amendments_applied


def test_headsets_approval_amendment(knowledge):
    before = snapshot(knowledge, "headsets", "Monsoon Foods", "2026-11-08")
    on_date = snapshot(knowledge, "headsets", "Monsoon Foods", "2026-11-09")
    assert before.approval_limit == 400000
    assert on_date.approval_limit == 320000
    assert before.walkaway_price == 4500
    assert before.supplier_price_cap == 4500


def test_catering_target_and_supplier_cap(knowledge):
    snap = snapshot(knowledge, "catering", "Helix CRM", "2026-12-10")
    assert snap.target_price == 270
    assert snap.walkaway_price == 400
    assert snap.supplier_status == SupplierStatus.APPROVED


def test_currency_overrides(knowledge):
    arvind = snapshot(knowledge, "laptops", "Arvind Office Supplies", "2026-10-10")
    assert arvind.supplier_exchange_overrides["GBP"] == 104.5
    assert arvind.currency_rates["GBP"] == 106
    assert arvind.effective_rate("GBP") == 104.5
    assert arvind.effective_rate("USD") == 84.5
    kestrel = snapshot(knowledge, "training", "Kestrel Learning", "2026-10-10")
    assert kestrel.effective_rate("EUR") == 90.5


def test_snapshot_version_is_deterministic(knowledge):
    first = snapshot(knowledge, "monitors", "Jupiter Devices", "2026-12-15")
    second = snapshot(knowledge, "monitors", "Jupiter Devices", "2026-12-15")
    assert first.snapshot_id == second.snapshot_id
    assert first.policy_version == second.policy_version
    assert "2026-12-15" in first.policy_version
    assert first.approval_limit == 960000


def test_unknown_supplier_and_category(knowledge):
    with pytest.raises(TemporalResolutionError):
        snapshot(knowledge, "laptops", "Missing Supplier", "2026-10-10")
    with pytest.raises(TemporalResolutionError):
        snapshot(knowledge, "missing-category", "Yamuna Traders", "2026-10-10")


def test_future_amendment_not_in_snapshot(knowledge):
    snap = snapshot(knowledge, "monitors", "Jupiter Devices", "2026-12-13")
    assert "AMD-09" not in snap.amendments_applied
    assert snap.approval_limit == 1200000
