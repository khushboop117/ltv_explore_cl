"""Tests for the billing-rule reconstruction and censoring in pipeline/prepare.py."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from prepare import DUNNING_DAYS, payment_counts  # noqa: E402

MONTH = 365.25 / 12


def frame(rows):
    cols = ["subscription_id", "created_at", "channel", "utm_campaign", "plan",
            "canceled_at", "ended_at", "end_reason"]
    d = pd.DataFrame(rows, columns=cols)
    for c in ["created_at", "canceled_at", "ended_at"]:
        d[c] = pd.to_datetime(d[c])
    return d


def day(start, offset):
    return (pd.Timestamp(start) + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")


def test_voluntary_cancel_pays_full_periods():
    # Three monthly payments, access ends on the 4th billing date.
    d = payment_counts(frame([
        ["a", "2024-01-01", "direct", None, "monthly", "2024-03-15",
         day("2024-01-01", round(3 * MONTH)), "voluntary"],
    ]))
    row = d.iloc[0]
    assert row.n == 3
    assert row.status == "vol" and row.n_conf == 3


def test_payment_failure_ends_after_dunning():
    # Paid twice, third charge failed, access ends 28 days after it.
    d = payment_counts(frame([
        ["b", "2024-01-01", "email", None, "monthly", None,
         day("2024-01-01", round(2 * MONTH) + DUNNING_DAYS), "payment_failed"],
    ]))
    row = d.iloc[0]
    assert row.n == 2
    assert row.status == "pf" and row.n_conf == 2


def test_active_annual_is_censored():
    d = payment_counts(frame([
        ["c", "2026-01-15", "organic_search", None, "annual", None, None, None],
    ]))
    row = d.iloc[0]
    assert row.n == 1 and row.status == "cen" and row.n_conf == 1
    assert bool(row.active_now)


def test_recent_renewal_in_retry_window_is_not_trusted():
    # Monthly sub whose latest renewal is 10 days before the cutoff:
    # that charge could still fail, so only the earlier payment is confirmed.
    created = (pd.Timestamp("2026-06-30") - pd.Timedelta(days=round(MONTH) + 10)).strftime("%Y-%m-%d")
    d = payment_counts(frame([
        ["d", created, "direct", None, "monthly", None, None, None],
    ]))
    row = d.iloc[0]
    assert row.n == 2
    assert row.status == "cen" and row.n_conf == 1


def test_pending_cancel_not_counted_before_outcome_window():
    # Asked to cancel 5 days before cutoff; renewal outcome not yet resolvable.
    d = payment_counts(frame([
        ["e", "2026-06-10", "direct", None, "monthly", "2026-06-25", None, None],
    ]))
    row = d.iloc[0]
    assert row.status == "cen" and not bool(row.active_now)


def test_end_date_that_breaks_billing_rule_raises():
    with pytest.raises(AssertionError):
        payment_counts(frame([
            ["f", "2024-01-01", "direct", None, "monthly", "2024-01-05", "2024-01-20", "voluntary"],
        ]))


def test_unknown_source_raises():
    with pytest.raises(AssertionError):
        payment_counts(frame([
            ["g", "2026-01-01", "tiktok", None, "monthly", None, None, None],
        ]))
