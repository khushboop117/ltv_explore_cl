"""
Turn subscriptions.csv into the compact summary the dashboard reads (web/data.js).

Each subscription becomes:
  n         - number of charges actually paid
  n_conf    - the last payment index whose *next* renewal outcome we can trust
  status    - 'vol' (voluntary churn after n_conf payments),
              'pf'  (payment failure after n_conf payments),
              'cen' (censored: still paying, or outcome not yet knowable)

Billing rules recovered from the data (they reproduce every ended row exactly):
  * period = 365.25/12 days (monthly) or 365 days (annual)
  * renewal k is charged on day round(k * period) after created_at
  * voluntary cancel: access ends on the next renewal date  -> paid n full periods
  * payment failure:  access ends DUNNING_DAYS after the failed renewal
                      -> the failed charge is not revenue

Censoring rule: a renewal's outcome only counts once (renewal date + DUNNING_DAYS)
is on or before the data cutoff. This keeps voluntary churn (visible on the
renewal date) and payment failure (visible 28 days later) on equal footing,
and stops cards still in retry from being counted as payers.

Usage:
  python pipeline/prepare.py --csv subscriptions.csv --out web/data.js
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp("2026-06-30")
DUNNING_DAYS = 28
PRICE = {"monthly": 15, "annual": 150}
PERIOD_DAYS = {"monthly": 365.25 / 12, "annual": 365.0}

# Order matters: the dashboard refers to sources by index.
SOURCES = [
    "direct",
    "email",
    "organic_search",
    "organic_social",
    "paid_social:prospecting_broad",
    "paid_social:lookalike_subscribers",
]
STATUS_CODE = {"cen": 0, "vol": 1, "pf": 2}
PLAN_CODE = {"monthly": 0, "annual": 1}


def load(csv_path: str) -> pd.DataFrame:
    d = pd.read_csv(csv_path, parse_dates=["created_at", "canceled_at", "ended_at"])
    assert d.subscription_id.is_unique, "duplicate subscription ids"
    assert d.plan.isin(PRICE).all(), "unexpected plan values"
    return d


def payment_counts(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    period = d.plan.map(PERIOD_DAYS)
    dur = (d.ended_at - d.created_at).dt.days          # days of access, if ended
    age = (CUTOFF - d.created_at).dt.days              # days observed

    vol = d.end_reason.eq("voluntary")
    pf = d.end_reason.eq("payment_failed")
    active = d.ended_at.isna()
    pending_cancel = active & d.canceled_at.notna()    # asked to cancel, not yet effective

    n = pd.Series(np.nan, index=d.index)
    n[vol] = np.round(dur[vol] / period[vol])
    n[pf] = np.round((dur[pf] - DUNNING_DAYS) / period[pf])

    # Active: count billing dates round(k*period) <= age, k = 0, 1, 2, ...
    nb = np.floor(age / period) + 1
    nb = np.where(np.round(nb * period) <= age, nb + 1, nb)
    nb = np.where(np.round((nb - 1) * period) > age, nb - 1, nb)
    n[active] = nb[active]
    d["n"] = n.astype(int)

    # Sanity check: rebuilding end dates from n must match the file exactly.
    rebuilt_vol = np.round(d.n * period) - dur
    rebuilt_pf = np.round(d.n * period) + DUNNING_DAYS - dur
    assert (rebuilt_vol[vol] == 0).all(), "voluntary end dates don't follow the billing rule"
    assert (rebuilt_pf[pf] == 0).all(), "payment-failure end dates don't follow the billing rule"

    # k = number of renewal transitions whose outcome is fully observable.
    k = np.floor((age - DUNNING_DAYS) / period)
    k = np.where(np.round((k + 1) * period) + DUNNING_DAYS <= age, k + 1, k)
    k = np.where((k >= 1) & (np.round(k * period) + DUNNING_DAYS > age), k - 1, k)
    k = np.maximum(k, 0)

    churned = vol | pf | pending_cancel
    event = churned & (d.n <= k)
    d["status"] = np.where(event, np.where(pf, "pf", "vol"), "cen")
    d["n_conf"] = np.where(event, d.n, k + 1).astype(int)
    assert (d.n_conf <= d.n).all()

    d["cohort"] = d.created_at.dt.strftime("%Y-%m")
    d["src"] = np.where(
        d.channel.eq("paid_social"),
        "paid_social:" + d.utm_campaign.fillna("none"),
        d.channel,
    )
    unknown = set(d.src) - set(SOURCES)
    assert not unknown, f"new sources need adding to SOURCES: {unknown}"
    d["active_now"] = active & ~pending_cancel
    return d


def export(d: pd.DataFrame, out_path: str) -> None:
    cohorts = sorted(d.cohort.unique())
    ci = {c: i for i, c in enumerate(cohorts)}
    si = {s: i for i, s in enumerate(SOURCES)}

    # Survival rows: [cohort, source, plan, n_conf, status, count]
    agg = d.groupby(["cohort", "src", "plan", "n_conf", "status"]).size().reset_index(name="c")
    rows = [
        [ci[r.cohort], si[r.src], PLAN_CODE[r.plan], int(r.n_conf), STATUS_CODE[r.status], int(r.c)]
        for r in agg.itertuples()
    ]

    # Revenue rows: [cohort, source, plan, signups, payments_to_date, active_now]
    rv = d.groupby(["cohort", "src", "plan"]).agg(
        s=("n", "size"), p=("n", "sum"), a=("active_now", "sum")
    ).reset_index()
    rev = [
        [ci[r.cohort], si[r.src], PLAN_CODE[r.plan], int(r.s), int(r.p), int(r.a)]
        for r in rv.itertuples()
    ]

    payload = {"cohorts": cohorts, "srcs": SOURCES, "rows": rows, "rev": rev}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        "window.LTV_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="subscriptions.csv")
    ap.add_argument("--out", default="web/data.js")
    args = ap.parse_args()

    d = payment_counts(load(args.csv))
    export(d, args.out)

    revenue = (d.n * d.plan.map(PRICE)).sum()
    print(f"{len(d):,} subscriptions | {d.active_now.sum():,} active | "
          f"${revenue:,.0f} collected to date")
    print(d.groupby(["plan", "status"]).size().unstack(fill_value=0))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
