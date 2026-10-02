# LTV Explorer

Compares subscriber lifetime value (LTV) by acquisition source for a $15/month, $150/year subscription product.

## Quick start

```bash
pip install -r requirements-dev.txt

# 1. Put subscriptions.csv in data/ (git-ignored), then build the summary the dashboard reads
python pipeline/prepare.py --csv data/subscriptions.csv --out web/data.js

# 2. Open the dashboard: either double-click web/index.html, or serve it
python -m http.server 8000 --directory web    # then visit http://localhost:8000

# 3. Optional: check the tail model against held-out data
python pipeline/backtest.py --csv data/subscriptions.csv --train 12

# 4. Run the tests
pytest -q
```

`web/data.js` is included already, so the dashboard opens before you run step 1. Chart.js and the fonts load from public CDNs, so you need to be online.

## Publish on GitHub

1. Create an empty repository on GitHub (no README, .gitignore or license, so the first push doesn't conflict).
2. Push this folder:
   ```bash
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
3. In the repository, go to **Settings → Pages** and set **Source** to **GitHub Actions**.
4. The *Deploy dashboard to GitHub Pages* workflow publishes `web/` to `https://<you>.github.io/<repo>/`. It runs on every push that changes `web/`, and you can also start it from the Actions tab.

The *CI* workflow runs the tests and checks that the dashboard script parses, on every push and pull request.

A GitHub Pages site is public even when the repository is private (unless you're on GitHub Enterprise). The page shows aggregated results only, and the raw CSV is never committed.

## Layout

```
.github/workflows/
  ci.yml        tests + script syntax check
  pages.yml     deploys web/ to GitHub Pages
data/           put subscriptions.csv here (git-ignored)
tests/          unit tests for the billing rules and censoring
pipeline/
  prepare.py    CSV -> payment counts -> censored survival table -> web/data.js
  backtest.py   fits the tail models on early renewals, compares with what happened
web/
  index.html    page structure and the methodology text
  styles.css    light and dark themes as CSS variables
  app.js        model fitting, LTV curves, charts, table, controls (runs in the browser)
  data.js       generated, about 7,000 rows: [cohort, source, plan, payments, status, count]
```

## How it works

**1. Payment counts (prepare.py).** Each subscription becomes the number of charges it paid. The end dates follow fixed rules, and the script asserts that they hold for every row:

| Rule | Value |
|---|---|
| Billing period | 365.25/12 days (monthly), 365 days (annual) |
| Renewal *k* charged on | day `round(k × period)` after signup |
| Voluntary cancel | access ends on the next renewal date (last period fully paid) |
| Payment failure | access ends 28 days after the failed renewal (failed charge is not revenue) |

**2. Censoring.** A renewal's outcome counts only once `renewal date + 28 days ≤ 2026-06-30`. Voluntary churn shows up on the renewal date, but payment failure only shows up 28 days later. Without this rule the two would be counted unevenly and cards still in retry would look like payers. Pending cancellations follow the same rule.

**3. Survival curve (app.js, `planCurve`).** For each renewal *j* the model takes churners ÷ subscribers at risk. That observed hazard is used while at least 100 subscribers are at risk (`MINRISK`).

**4. Tail model.** Past the observed data, the curve continues with two shifted-beta-geometric (sBG) processes, one for voluntary churn and one for payment failure. They are combined as competing risks:

```
h_vol(j) = α_v / (α_v + β_v + j − 1)
h_pf(j)  = α_f / (α_f + β_f + j − 1)
S(j+1)   = S(j) · (1 − h_vol(j)) · (1 − h_pf(j))
```

Parameters are fit by maximum likelihood (Nelder–Mead, written out in `app.js`), so they refit whenever filters change. A segment with fewer than 200 at-risk renewals (`THIN`) borrows the curve shape from the whole selection and gets a "thin" flag.

**5. LTV.** `LTV(H) = Σ price × margin × S(j) × (1 + r)^(−t_j/12)`, summed over payments *j* charged before month *H*. With both plans selected, each group is a plan-mix-weighted average of its monthly and annual curves.

**Backtest (monthly plans, trained on renewals 1–12):**

| Source | Actual retention at month 30 | One sBG | sBG + flat failures | Dual sBG (used) | 36-month LTV error |
|---|---|---|---|---|---|
| All | 36.6% | 41.2% | 33.6% | 37.7% | +0.8% |
| Direct | 40.3% | 45.1% | 37.3% | 41.3% | +0.5% |
| Email | 52.3% | 57.1% | 48.9% | 53.2% | +0.2% |
| Organic search | 39.0% | 43.4% | 35.8% | 39.6% | +0.4% |
| Organic social | 25.2% | 29.2% | 23.1% | 26.7% | +1.7% |
| Paid social · Lookalike | 38.1% | 43.3% | 36.1% | 40.9% | +2.1% |
| Paid social · Prospecting | 23.5% | 27.4% | 20.9% | 24.7% | +1.5% |

## Settings you can change

| Where | Setting | Default |
|---|---|---|
| `prepare.py` | `CUTOFF`, `DUNNING_DAYS`, `PRICE`, `SOURCES` | 2026-06-30, 28, 15/150 |
| `app.js` | `MINRISK` (at-risk count needed to trust observed data) | 100 |
| `app.js` | `THIN` (at-risk total below which a segment borrows the pooled curve) | 200 |
| `app.js` | `JMAX` (payments modeled per plan, sets the 60-month cap) | 61 monthly / 6 annual |
| `app.js` | `SRC_LABEL`, `COLOR_SLOT` (display names, chart colors) | n/a |

A new channel or campaign means adding it to `SOURCES` in `prepare.py`, plus a label and color in `app.js`. The script stops with an error if it finds a source that isn't listed.

## Taking it further

- **Real CAC.** CAC is typed in by hand and kept in the browser's localStorage. Add a spend-by-source file to `prepare.py` and pass CAC through `data.js`.
- **Scheduled refresh.** Add a scheduled workflow that pulls fresh data, runs `prepare.py` and commits `web/data.js`. The Pages workflow then redeploys.
- **Larger data.** Move `prepare.py` into SQL or dbt (the output is just a `GROUP BY cohort, source, plan, n_conf, status`) and refresh `data.js` on a schedule.
- **Uncertainty.** Bootstrap subscribers within each segment and refit to get LTV intervals.
