const D = window.LTV_DATA;
if (!D) {
  document.body.innerHTML =
    '<p style="padding:24px">Missing data.js. Run: python pipeline/prepare.py --csv subscriptions.csv</p>';
  throw new Error("data.js not loaded");
}
const PRICE = [15, 150],
  PERIOD = [1, 12],
  JMAX = [61, 6],
  MINRISK = 100,
  THIN = 200;
const SRC_LABEL = {
  direct: "Direct",
  email: "Email",
  organic_search: "Organic search",
  organic_social: "Organic social",
  "paid_social:prospecting_broad": "Paid social · Prospecting",
  "paid_social:lookalike_subscribers": "Paid social · Lookalike",
  paid_social: "Paid social",
  monthly: "Monthly plan",
  annual: "Annual plan",
  all: "All selected",
};
const COLOR_SLOT = {
  direct: 1,
  email: 2,
  organic_search: 3,
  organic_social: 4,
  paid_social: 5,
  "paid_social:prospecting_broad": 5,
  "paid_social:lookalike_subscribers": 6,
  monthly: 1,
  annual: 2,
};
const CHANNELS = ["direct", "email", "organic_search", "organic_social", "paid_social"];
const srcChannel = D.srcs.map((s) => s.split(":")[0]);
const S = {
  groupBy: "channel",
  plan: "all",
  hz: 24,
  c0: 0,
  c1: D.cohorts.length - 1,
  chans: new Set(CHANNELS),
  tail: "model",
  disc: 0,
  margin: 100,
  target: 3,
  cac: {},
};
try {
  const c = JSON.parse(localStorage.getItem("ltv-cac") || "{}");
  if (c && typeof c === "object") S.cac = c;
} catch (e) {}

// ---------- model ----------
function nm(f, x0) {
  let s = [x0, [x0[0] + 0.6, x0[1]], [x0[0], x0[1] + 0.6]].map((x) => ({ x, v: f(x) }));
  for (let it = 0; it < 300; it++) {
    s.sort((a, b) => a.v - b.v);
    const [b, g, w] = s;
    const c = [(b.x[0] + g.x[0]) / 2, (b.x[1] + g.x[1]) / 2];
    const pt = (t) => [c[0] + t * (w.x[0] - c[0]), c[1] + t * (w.x[1] - c[1])];
    const r = pt(-1),
      fr = f(r);
    if (fr < b.v) {
      const e = pt(-2),
        fe = f(e);
      s[2] = fe < fr ? { x: e, v: fe } : { x: r, v: fr };
    } else if (fr < g.v) {
      s[2] = { x: r, v: fr };
    } else {
      const k = pt(0.5),
        fk = f(k);
      if (fk < w.v) s[2] = { x: k, v: fk };
      else {
        s = s.map((p) => {
          const x = [(p.x[0] + b.x[0]) / 2, (p.x[1] + b.x[1]) / 2];
          return { x, v: f(x) };
        });
      }
    }
    if (Math.abs(s[2].v - s[0].v) < 1e-9) break;
  }
  s.sort((a, b) => a.v - b.v);
  return s[0].x;
}
function fitSBG(ev, non) {
  let tot = 0;
  for (let j = 1; j < ev.length; j++) tot += ev[j];
  if (tot === 0) return [Math.exp(-8), 1];
  const f = (p) => {
    const a = Math.exp(Math.max(-8, Math.min(8, p[0]))),
      b = Math.exp(Math.max(-8, Math.min(8, p[1])));
    let L = 0;
    for (let j = 1; j < ev.length; j++) {
      if (!ev[j] && !non[j]) continue;
      const h = a / (a + b + j - 1);
      L -= ev[j] * Math.log(h) + non[j] * Math.log1p(-h);
    }
    return L;
  };
  const x = nm(f, [0, 1]);
  return [Math.exp(Math.max(-8, Math.min(8, x[0]))), Math.exp(Math.max(-8, Math.min(8, x[1])))];
}
function tally(rows, p) {
  const J = JMAX[p] + 1,
    vol = new Float64Array(J + 2),
    pf = new Float64Array(J + 2),
    cen = new Float64Array(J + 2);
  let n = 0;
  for (const r of rows) {
    if (r[2] !== p) continue;
    const j = Math.min(r[3], J);
    if (r[4] === 1) vol[j] += r[5];
    else if (r[4] === 2) pf[j] += r[5];
    else cen[j] += r[5];
    n += r[5];
  }
  const reached = new Float64Array(J + 3);
  for (let j = J; j >= 1; j--) reached[j] = reached[j + 1] + vol[j] + pf[j] + cen[j];
  const surv = new Float64Array(J + 1),
    risk = new Float64Array(J + 1);
  for (let j = 1; j <= J; j++) {
    surv[j] = reached[j + 1];
    risk[j] = vol[j] + pf[j] + surv[j];
  }
  return { vol, pf, surv, risk, n, J };
}
function fitModel(t) {
  const ev1 = [],
    non1 = [],
    ev2 = [],
    non2 = [];
  for (let j = 0; j <= t.J; j++) {
    ev1[j] = t.vol[j] || 0;
    non1[j] = (t.pf[j] || 0) + (t.surv[j] || 0);
    ev2[j] = t.pf[j] || 0;
    non2[j] = t.surv[j] || 0;
  }
  return { v: fitSBG(ev1, non1), f: fitSBG(ev2, non2) };
}
function hazModel(m, j) {
  const hv = m.v[0] / (m.v[0] + m.v[1] + j - 1),
    hf = m.f[0] / (m.f[0] + m.f[1] + j - 1);
  return 1 - (1 - hv) * (1 - hf);
}
// survival over payments: Sv[j]=P(make payment j), j=1..JMAX ; obsTo = last payment index supported by data
function planCurve(rows, p, fallback) {
  const t = tally(rows, p);
  if (!t.n) return null;
  let riskSum = 0;
  for (let j = 1; j <= t.J; j++) riskSum += t.risk[j];
  const thin = riskSum < THIN && fallback;
  const m = thin ? fallback.model : fitModel(t);
  const Jm = JMAX[p];
  const Sv = new Float64Array(Jm + 1);
  Sv[1] = 1;
  let obsTo = 1,
    emp = true;
  for (let j = 1; j < Jm; j++) {
    let h;
    if (emp && !thin && t.risk[j] >= MINRISK) {
      h = (t.vol[j] + t.pf[j]) / t.risk[j];
      obsTo = j + 1;
    } else {
      emp = false;
      h = hazModel(m, j);
    }
    Sv[j + 1] = Sv[j] * (1 - h);
  }
  if (thin) obsTo = 1;
  const v1 = t.risk[1] ? (t.vol[1] + t.pf[1]) / t.risk[1] : null;
  let pfT = 0,
    vT = 0;
  for (let j = 1; j <= t.J; j++) {
    pfT += t.pf[j];
    vT += t.vol[j];
  }
  return {
    Sv,
    obsTo,
    n: t.n,
    model: m,
    thin,
    r1: v1 == null ? null : 1 - v1,
    risk1: t.risk[1],
    pfShare: pfT + vT ? pfT / (pfT + vT) : null,
  };
}
function curves(c, p) {
  // cumulative LTV by month 0..60, observed-only part, retention
  const L = new Float64Array(61),
    Lo = new Float64Array(61),
    R = new Float64Array(61);
  const per = PERIOD[p],
    pr = (PRICE[p] * S.margin) / 100,
    r = S.disc / 100;
  for (let j = 1; j < c.Sv.length; j++) {
    const tj = (j - 1) * per;
    if (tj >= 60) break;
    const v = pr * c.Sv[j] * Math.pow(1 + r, -tj / 12);
    const obs = j <= c.obsTo;
    for (let m = tj + 1; m <= 60; m++) {
      L[m] += obs || S.tail === "model" ? v : 0;
      if (obs) Lo[m] += v;
    }
  }
  for (let m = 0; m <= 60; m++) {
    const j = Math.floor(m / per) + 1;
    R[m] = j < c.Sv.length ? c.Sv[j] : 0;
  }
  const obsMonth = c.obsTo * per; // months fully supported
  return { L, Lo, R, obsMonth };
}

// ---------- data selection ----------
function keyOf(r) {
  const s = D.srcs[r[1]];
  if (S.groupBy === "plan") return r[2] ? "annual" : "monthly";
  if (S.groupBy === "channel") return srcChannel[r[1]];
  return s;
}
function inSel(r) {
  return (
    r[0] >= S.c0 &&
    r[0] <= S.c1 &&
    S.chans.has(srcChannel[r[1]]) &&
    (S.plan === "all" || r[2] === (S.plan === "annual" ? 1 : 0))
  );
}
function planList() {
  return S.plan === "all" ? [0, 1] : [S.plan === "annual" ? 1 : 0];
}
function evalGroup(rows, fb) {
  const out = { plans: {}, n: 0 };
  for (const p of planList()) {
    const c = planCurve(rows, p, fb && fb[p]);
    if (c) {
      out.plans[p] = { c, cv: curves(c, p) };
      out.n += c.n;
    }
  }
  const L = new Float64Array(61),
    Lo = new Float64Array(61),
    R = new Float64Array(61);
  let obsMonth = 60,
    thin = false,
    annual = 0;
  for (const p in out.plans) {
    const { c, cv } = out.plans[p];
    const w = c.n / out.n;
    if (+p === 1) annual = w;
    thin = thin || c.thin;
    obsMonth = Math.min(obsMonth, cv.obsMonth);
    for (let m = 0; m <= 60; m++) {
      L[m] += w * cv.L[m];
      Lo[m] += w * cv.Lo[m];
      R[m] += w * cv.R[m];
    }
  }
  Object.assign(out, { L, Lo, R, obsMonth, thin, annual });
  const m = out.plans[0]?.c,
    a = out.plans[1]?.c;
  out.r1m = m?.r1;
  out.r1a = a?.r1;
  let pv = 0,
    pw = 0;
  for (const p in out.plans) {
    const c = out.plans[p].c;
    if (c.pfShare != null) {
      pv += c.pfShare * c.n;
      pw += c.n;
    }
  }
  out.pfShare = pw ? pv / pw : null;
  return out;
}
function compute() {
  const sel = D.rows.filter(inSel);
  const all = evalGroup(sel, null);
  const fb = {};
  for (const p in all.plans) fb[p] = all.plans[p].c;
  const by = new Map();
  for (const r of sel) {
    const k = keyOf(r);
    if (!by.has(k)) by.set(k, []);
    by.get(k).push(r);
  }
  const groups = [];
  for (const [k, rs] of by) {
    const g = evalGroup(rs, fb);
    if (g.n) groups.push(Object.assign(g, { key: k }));
  }
  // realized revenue
  for (const g of [all, ...groups]) {
    g.real = 0;
    g.active = 0;
  }
  const gm = new Map(groups.map((g) => [g.key, g]));
  for (const r of D.rev) {
    if (!inSel(r)) continue;
    const g = gm.get(keyOf(r));
    const v = r[4] * PRICE[r[2]];
    if (g) {
      g.real += v;
      g.active += r[5];
    }
    all.real += v;
    all.active += r[5];
  }
  all.key = "all";
  return { all, groups, sel };
}

// ---------- rendering ----------
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const col = (k) => css("--s" + (COLOR_SLOT[k] || 1));
const fmt$ = (v) =>
  v == null || !isFinite(v)
    ? "–"
    : "$" + (v >= 1000 ? Math.round(v).toLocaleString() : v.toFixed(v < 100 ? 2 : 0));
const fmt0 = (v) => "$" + Math.round(v).toLocaleString();
const pct = (v) => (v == null ? "–" : (v * 100).toFixed(1) + "%");
const charts = {};
function baseOpts() {
  const ink2 = css("--ink-2"),
    grid = css("--grid");
  Chart.defaults.font.family = css("--f-body") || "sans-serif";
  Chart.defaults.color = ink2;
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 250 },
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: css("--panel"),
        titleColor: css("--ink"),
        bodyColor: css("--ink-2"),
        borderColor: css("--rule"),
        borderWidth: 1,
        padding: 10,
        boxPadding: 4,
        usePointStyle: true,
      },
    },
    scales: {
      x: {
        grid: { color: grid, drawTicks: false },
        border: { display: false },
        ticks: { padding: 6 },
      },
      y: {
        grid: { color: grid, drawTicks: false },
        border: { display: false },
        ticks: { padding: 6 },
      },
    },
  };
}
function upsert(id, cfg) {
  if (charts[id]) {
    charts[id].destroy();
  }
  charts[id] = new Chart(document.getElementById(id), cfg);
}
function legend(id, items, extra = "") {
  document.getElementById(id).innerHTML =
    items
      .map(
        (g) =>
          `<span><i class="sw" style="background:${col(g.key)}"></i>${SRC_LABEL[g.key]}</span>`,
      )
      .join("") + extra;
}
const hzLine = {
  id: "hz",
  afterDatasetsDraw(ch) {
    const x = ch.scales.x;
    if (!x || ch.config.options._hz == null) return;
    const px = x.getPixelForValue(ch.config.options._hz);
    const c = ch.ctx;
    c.save();
    c.strokeStyle = css("--ink-3");
    c.setLineDash([2, 3]);
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(px, ch.chartArea.top);
    c.lineTo(px, ch.chartArea.bottom);
    c.stroke();
    c.fillStyle = css("--ink-2");
    c.font = "11px " + css("--f-mono");
    c.fillText(ch.config.options._hz + " mo", px + 4, ch.chartArea.top + 10);
    c.restore();
  },
};
function render() {
  const { all, groups, sel } = compute();
  const H = S.hz;
  const order = [...groups].sort((a, b) => b.L[H] - a.L[H]);
  const stable = [...groups].sort(
    (a, b) => COLOR_SLOT[a.key] - COLOR_SLOT[b.key] || a.key.localeCompare(b.key),
  );
  // tiles
  const best = order[0],
    worst = order[order.length - 1];
  document.getElementById("tiles").innerHTML = `
   <div class="panel tile"><span class="eyebrow">Blended LTV · ${H} mo</span><span class="v">${fmt$(all.L[H])}</span><span class="s">${all.n.toLocaleString()} signups selected · ${pct(all.annual)} annual</span></div>
   <div class="panel tile"><span class="eyebrow">Highest value</span><span class="v">${best ? fmt$(best.L[H]) : "–"}</span><span class="s">${best ? SRC_LABEL[best.key] : ""}${best ? ` · ${(best.L[H] / all.L[H]).toFixed(2)}× blended` : ""}</span></div>
   <div class="panel tile"><span class="eyebrow">Lowest value</span><span class="v">${worst ? fmt$(worst.L[H]) : "–"}</span><span class="s">${worst ? SRC_LABEL[worst.key] : ""}${worst ? ` · ${(worst.L[H] / all.L[H]).toFixed(2)}× blended` : ""}</span></div>
   <div class="panel tile"><span class="eyebrow">Modeled share of blended LTV</span><span class="v">${S.tail === "none" ? "0%" : pct(1 - all.Lo[H] / all.L[H])}</span><span class="s">${S.tail === "none" ? "Tail switched off; LTV is a floor" : "Revenue the data has not yet seen"}</span></div>`;
  document.getElementById("barT").textContent = `LTV at ${H} month${H > 1 ? "s" : ""}`;
  // bar
  const ob = baseOpts();
  ob.indexAxis = "y";
  ob.interaction = { mode: "nearest", axis: "y", intersect: false };
  ob.scales.x.stacked = true;
  ob.scales.y.stacked = true;
  ob.scales.y.grid.display = false;
  ob.scales.x.ticks.callback = (v) => "$" + v;
  ob.plugins.tooltip.callbacks = {
    title: (i) => SRC_LABEL[order[i[0].dataIndex].key],
    label: (c) => {
      const g = order[c.dataIndex];
      return c.datasetIndex
        ? ` Modeled tail: ${fmt$(g.L[H] - g.Lo[H])}`
        : ` Observed: ${fmt$(Math.min(g.Lo[H], g.L[H]))}`;
    },
    footer: (i) => {
      const g = order[i[0].dataIndex];
      return `Total ${fmt$(g.L[H])} · n=${g.n.toLocaleString()}`;
    },
  };
  upsert("barC", {
    type: "bar",
    options: ob,
    data: {
      labels: order.map((g) => SRC_LABEL[g.key]),
      datasets: [
        {
          data: order.map((g) => Math.min(g.Lo[H], g.L[H])),
          backgroundColor: order.map((g) => col(g.key)),
          borderColor: css("--panel"),
          borderWidth: { right: 2 },
          borderSkipped: false,
          borderRadius: 0,
          barThickness: "flex",
          maxBarThickness: 26,
        },
        {
          data: order.map((g) => Math.max(0, g.L[H] - g.Lo[H])),
          backgroundColor: order.map((g) => col(g.key) + "55"),
          borderColor: order.map((g) => col(g.key)),
          borderWidth: { top: 1, right: 1, bottom: 1 },
          borderSkipped: "left",
          borderRadius: { topRight: 4, bottomRight: 4 },
          maxBarThickness: 26,
        },
      ],
    },
  });
  document.getElementById("barL").innerHTML =
    `<span><i class="sw box" style="background:var(--ink-3)"></i>Observed</span><span><i class="sw box" style="background:transparent;border:1px solid var(--ink-3)"></i>Modeled tail</span>`;
  // cumulative
  const months = [...Array(61).keys()];
  const mk = (g, arr) => ({
    label: SRC_LABEL[g.key],
    data: Array.from(arr),
    borderColor: col(g.key),
    backgroundColor: col(g.key),
    borderWidth: 2,
    pointRadius: 0,
    pointHoverRadius: 4,
    tension: 0,
    segment: { borderDash: (c) => (c.p1DataIndex > g.obsMonth ? [5, 4] : undefined) },
  });
  const oc = baseOpts();
  oc.scales.x.type = "linear";
  oc.scales.x.min = 0;
  oc.scales.x.max = 60;
  oc.scales.x.ticks.stepSize = 12;
  oc.scales.x.title = { display: true, text: "Months since signup" };
  oc.scales.y.ticks.callback = (v) => "$" + v;
  oc._hz = H;
  oc.plugins.tooltip.callbacks = {
    title: (i) => "Month " + i[0].parsed.x,
    label: (c) => ` ${c.dataset.label}: ${fmt$(c.parsed.y)}`,
  };
  oc.plugins.tooltip.itemSort = (a, b) => b.parsed.y - a.parsed.y;
  upsert("cumC", {
    type: "line",
    options: oc,
    plugins: [hzLine],
    data: {
      datasets: stable.map((g) => {
        const d = mk(g, g.L);
        d.data = months.map((m) => ({ x: m, y: g.L[m] }));
        return d;
      }),
    },
  });
  legend("cumL", stable, `<span><i class="sw dash"></i>Modeled</span>`);
  // retention
  const or = baseOpts();
  or.scales.x.type = "linear";
  or.scales.x.min = 0;
  or.scales.x.max = 60;
  or.scales.x.ticks.stepSize = 12;
  or.scales.x.title = { display: true, text: "Months since signup" };
  or.scales.y.min = 0;
  or.scales.y.max = 1;
  or.scales.y.ticks.callback = (v) => Math.round(v * 100) + "%";
  or._hz = H;
  or.plugins.tooltip.callbacks = {
    title: (i) => "Month " + i[0].parsed.x,
    label: (c) => ` ${c.dataset.label}: ${pct(c.parsed.y)}`,
  };
  or.plugins.tooltip.itemSort = (a, b) => b.parsed.y - a.parsed.y;
  upsert("retC", {
    type: "line",
    options: or,
    plugins: [hzLine],
    data: {
      datasets: stable.map((g) => {
        const d = mk(g, g.R);
        d.stepped = S.plan === "annual" ? "before" : false;
        d.data = months.map((m) => ({ x: m, y: g.R[m] }));
        return d;
      }),
    },
  });
  legend("retL", stable, `<span><i class="sw dash"></i>Modeled</span>`);
  renderCohort(sel, stable);
  renderTable(order, all, H);
}
function renderCohort(sel, stable) {
  const p = S.plan === "annual" ? 1 : 0;
  document.getElementById("cohT").textContent =
    `First-renewal rate by signup quarter · ${p ? "annual" : "monthly"} plan`;
  document.getElementById("cohP").textContent = p
    ? "Share of annual subscribers who renewed for a second year. Readable ~13 months after signup, so recent quarters are blank."
    : "Share of monthly subscribers who paid a second time. An early read on cohort quality, readable ~2 months after signup.";
  const qs = [
    ...new Set(
      D.cohorts.map((c) => c.slice(0, 4) + " Q" + (Math.floor((+c.slice(5) - 1) / 3) + 1)),
    ),
  ];
  const qOf = (i) => {
    const c = D.cohorts[i];
    return qs.indexOf(c.slice(0, 4) + " Q" + (Math.floor((+c.slice(5) - 1) / 3) + 1));
  };
  const acc = new Map();
  for (const r of sel) {
    if (r[2] !== p) continue;
    const k = keyOf(r),
      q = qOf(r[0]);
    if (!acc.has(k))
      acc.set(
        k,
        qs.map(() => ({ e: 0, s: 0 })),
      );
    const a = acc.get(k)[q];
    if (r[3] === 1 && r[4] !== 0) a.e += r[5];
    else if (r[3] >= 2) a.s += r[5];
  }
  const qmin = qOf(S.c0),
    qmax = qOf(S.c1);
  const labels = qs.slice(qmin, qmax + 1);
  const o = baseOpts();
  o.scales.y.ticks.callback = (v) => Math.round(v * 100) + "%";
  o.plugins.tooltip.callbacks = {
    label: (c) => ` ${c.dataset.label}: ${pct(c.parsed.y)} (n=${c.dataset._n[c.dataIndex]})`,
  };
  o.plugins.tooltip.itemSort = (a, b) => b.parsed.y - a.parsed.y;
  o.scales.x.ticks.maxRotation = 0;
  o.scales.x.ticks.autoSkip = true;
  const ds = stable
    .filter((g) => acc.has(g.key))
    .map((g) => {
      const arr = acc.get(g.key).slice(qmin, qmax + 1);
      return {
        label: SRC_LABEL[g.key],
        _n: arr.map((a) => a.e + a.s),
        data: arr.map((a) => (a.e + a.s >= 30 ? a.s / (a.e + a.s) : null)),
        borderColor: col(g.key),
        backgroundColor: col(g.key),
        borderWidth: 2,
        pointRadius: 2.5,
        pointHoverRadius: 5,
        spanGaps: false,
      };
    });
  upsert("cohC", { type: "line", options: o, data: { labels, datasets: ds } });
  legend(
    "cohL",
    stable.filter((g) => acc.has(g.key)),
  );
}
function renderTable(order, all, H) {
  const t = S.target;
  const showPlanCols = S.groupBy !== "plan";
  const head = `<thead><tr><th>${S.groupBy === "plan" ? "Plan" : "Source"}</th><th>Signups</th>${showPlanCols && S.plan === "all" ? "<th>Annual mix</th>" : ""}<th>Monthly: 2nd payment</th><th>Annual: 2nd year</th><th>Churn from failed payment</th><th>LTV ${H} mo</th><th>vs blended</th><th>Modeled share</th><th>Revenue collected to date</th><th>CAC (enter)</th><th>LTV : CAC</th><th>Payback</th><th>Max CPA at ${t}:1</th></tr></thead>`;
  const row = (g, isTotal) => {
    const L = g.L[H];
    const cac = +S.cac[g.key] || 0;
    const ratio = cac ? L / cac : null;
    let pb = "–";
    if (cac) {
      let m = 0;
      while (m <= 60 && g.L[m] < cac) m++;
      pb = m > 60 ? "> 60 mo" : m + " mo";
    }
    const rc = ratio == null ? "" : ratio >= t ? "good" : ratio >= 1 ? "warn" : "bad";
    const idx = L / all.L[H];
    return `<tr class="${isTotal ? "total" : ""}"><td>${isTotal ? "" : `<i class="dot" style="background:${col(g.key)}"></i>`}${SRC_LABEL[g.key]}${g.thin ? '<span class="pill warn">thin</span>' : ""}</td>
     <td>${g.n.toLocaleString()}</td>${showPlanCols && S.plan === "all" ? `<td>${pct(g.annual)}</td>` : ""}<td>${pct(g.r1m)}</td><td>${pct(g.r1a)}</td><td>${pct(g.pfShare)}</td>
     <td><b>${fmt$(L)}</b></td><td>${isTotal ? "1.00×" : idx.toFixed(2) + "×"}<i class="bar" style="width:${Math.min(60, idx * 30)}px"></i></td><td>${S.tail === "none" ? "0%" : pct(1 - g.Lo[H] / L)}</td><td>${fmt0(g.real)}</td>
     <td>${isTotal ? "" : `<input type="number" min="0" step="1" id="cac-${g.key.replace(/[^a-z_]/g, "-")}" data-k="${g.key}" value="${cac || ""}" placeholder="$" aria-label="CAC for ${SRC_LABEL[g.key]}">`}</td>
     <td>${ratio == null ? "–" : ratio.toFixed(2) + "×"}${ratio == null ? "" : `<span class="pill ${rc}">${ratio >= t ? "above target" : ratio >= 1 ? "below target" : "loses money"}</span>`}</td><td>${pb}</td><td>${isTotal ? "–" : fmt$(L / t)}</td></tr>`;
  };
  document.getElementById("tbl").innerHTML =
    head + "<tbody>" + order.map((g) => row(g)).join("") + row(all, true) + "</tbody>";
  document.getElementById("tblNote").textContent =
    `2nd payment and 2nd year rates count only subscribers whose renewal had fully resolved by 30 Jun 2026. Churn from failed payment = share of observed cancellations caused by a failed charge. Revenue collected to date is actual cash in the file and depends on cohort age, so compare sources with LTV instead.`;
  document.querySelectorAll("#tbl input[data-k]").forEach((inp) =>
    inp.addEventListener("change", (e) => {
      const k = e.target.dataset.k;
      const v = parseFloat(e.target.value);
      if (v > 0) S.cac[k] = v;
      else delete S.cac[k];
      try {
        localStorage.setItem("ltv-cac", JSON.stringify(S.cac));
      } catch (_) {}
      render();
    }),
  );
}

// ---------- controls ----------
function seg(id, key) {
  const el = document.getElementById(id);
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    S[key] = b.dataset.v;
    el.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x === b));
    render();
  });
}
seg("groupBy", "groupBy");
seg("plan", "plan");
seg("tail", "tail");
const hz = document.getElementById("hz"),
  hzOut = document.getElementById("hzOut");
function setHz(v) {
  S.hz = +v;
  hz.value = v;
  hzOut.textContent = v;
  document
    .querySelectorAll("#hzPre button")
    .forEach((b) => b.classList.toggle("on", +b.dataset.v === +v));
}
hz.addEventListener("input", (e) => {
  setHz(e.target.value);
  render();
});
document.getElementById("hzPre").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (b) {
    setHz(b.dataset.v);
    render();
  }
});
const c0 = document.getElementById("c0"),
  c1 = document.getElementById("c1");
const mname = (c) =>
  new Date(c + "-15").toLocaleString("en-US", { month: "short", year: "numeric" });
D.cohorts.forEach((c, i) => {
  c0.add(new Option(mname(c), i));
  c1.add(new Option(mname(c), i));
});
c0.value = 0;
c1.value = D.cohorts.length - 1;
c0.addEventListener("change", () => {
  S.c0 = +c0.value;
  if (S.c0 > S.c1) {
    S.c1 = S.c0;
    c1.value = S.c1;
  }
  render();
});
c1.addEventListener("change", () => {
  S.c1 = +c1.value;
  if (S.c1 < S.c0) {
    S.c0 = S.c1;
    c0.value = S.c0;
  }
  render();
});
document.getElementById("chans").innerHTML = CHANNELS.map(
  (c) =>
    `<label class="chk"><input type="checkbox" id="ch-${c}" value="${c}" checked><i class="dot sw box" style="background:var(--s${COLOR_SLOT[c]})"></i>${SRC_LABEL[c]}</label>`,
).join("");
document.getElementById("chans").addEventListener("change", (e) => {
  const v = e.target.value;
  if (e.target.checked) S.chans.add(v);
  else {
    if (S.chans.size === 1) {
      e.target.checked = true;
      return;
    }
    S.chans.delete(v);
  }
  render();
});
for (const [id, key] of [
  ["disc", "disc"],
  ["margin", "margin"],
  ["target", "target"],
])
  document.getElementById(id).addEventListener("change", (e) => {
    let v = parseFloat(e.target.value);
    if (!isFinite(v) || v < 0) {
      v = key === "margin" ? 100 : key === "target" ? 3 : 0;
      e.target.value = v;
    }
    S[key] = v;
    render();
  });
setHz(24);
render();
const rerender = () => render();
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rerender);
new MutationObserver(rerender).observe(document.documentElement, {
  attributes: true,
  attributeFilter: ["data-theme"],
});
