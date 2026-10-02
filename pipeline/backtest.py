"""
Backtest the tail model on monthly plans.

Fit on the first TRAIN renewals only, predict the rest, and compare with the
observed (Kaplan-Meier style) survival curve. Three candidate models:

  sbg       one shifted-beta-geometric curve for all churn
  sbg+flat  sBG for voluntary churn + a constant payment-failure rate
  dual      sBG for voluntary churn + sBG for payment failure (used in the app)

Usage:
  python pipeline/backtest.py --csv subscriptions.csv --train 12
"""
import argparse
import warnings

import numpy as np
from scipy.optimize import minimize

from prepare import load, payment_counts

warnings.filterwarnings("ignore")
J = 35  # renewals to evaluate


def tables(g):
    """Per renewal j: voluntary churn, payment-failure churn, and survivors."""
    vol = np.zeros(J + 2); pf = np.zeros(J + 2); tot = np.zeros(J + 2)
    for (n, s), c in g.groupby(["n_conf", "status"]).size().items():
        n = min(n, J + 1)
        tot[n] += c
        if s == "vol": vol[n] += c
        if s == "pf": pf[n] += c
    reached = tot[::-1].cumsum()[::-1]
    surv = np.append(reached[1:], 0)
    return vol[1:J + 1], pf[1:J + 1], surv[1:J + 1]


def fit_sbg(events, non_events):
    t = np.arange(1, len(events) + 1)

    def nll(p):
        a, b = np.exp(p)
        h = a / (a + b + t - 1)
        return -(events * np.log(h) + non_events * np.log1p(-h)).sum()

    return np.exp(minimize(nll, [0, 1], method="Nelder-Mead").x)


def survival(h):
    return np.concatenate([[1], np.cumprod(1 - h)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="subscriptions.csv")
    ap.add_argument("--train", type=int, default=12)
    args = ap.parse_args()
    T = args.train

    d = payment_counts(load(args.csv))
    m = d[d.plan == "monthly"]
    t = np.arange(1, J + 1)
    print(f"Train on renewals 1-{T}; retention at month 30 and 36-month LTV error\n")
    print(f"{'source':36s} {'actual':>7s} {'sbg':>7s} {'sbg+flat':>9s} {'dual':>7s} {'dual LTV err':>13s}")
    for name, g in [("ALL", m)] + list(m.groupby("src")):
        v, p, s = tables(g)
        actual = survival((v + p) / np.maximum(v + p + s, 1))

        a, b = fit_sbg(v[:T] + p[:T], s[:T])
        S1 = survival(a / (a + b + t - 1))

        a, b = fit_sbg(v[:T], p[:T] + s[:T])
        flat = p[:T].sum() / (p[:T] + s[:T]).sum()
        S2 = survival(1 - (1 - a / (a + b + t - 1)) * (1 - flat))

        c, e = fit_sbg(p[:T], s[:T])
        S3 = survival(1 - (1 - a / (a + b + t - 1)) * (1 - c / (c + e + t - 1)))

        err = S3[:36].sum() / actual[:36].sum() - 1
        print(f"{name:36s} {actual[30]:7.1%} {S1[30]:7.1%} {S2[30]:9.1%} {S3[30]:7.1%} {err:+13.1%}")


if __name__ == "__main__":
    main()
