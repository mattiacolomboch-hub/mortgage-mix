"""First run on real SNB data. Prints estimates and results; saves figure."""
import csv, numpy as np, model as m
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = list(csv.DictReader(open("../data/clean/saron_monthly.csv")))
dates = [r["month"] for r in S if r["saron_pct"]]; r = np.array([float(r["saron_pct"]) for r in S if r["saron_pct"]]) / 100
M = {row["month"]: row for row in csv.DictReader(open("../data/clean/mortgage_rates_monthly.csv"))}

# 1. AR(1) estimates: full SARON sample and post-2008 sample
for lab, sl in [("SARON 1999-06..2026-10", slice(None)), ("SARON 2008-12..2026-10", slice(dates.index("2008-12"), None))]:
    p = m.estimate_ar1(r[sl]); print(lab, {k: round(float(v), 5) for k, v in p.items()},
        "half-life (months):", round(np.log(2) / p["kappa"], 1) if p["kappa"] > 0 else "no reversion")
p = m.estimate_ar1(r)

# 2. Historical SARON-mortgage margin over SARON (from 2021)
marg = [(d, float(M[d]["saron_unlimited_pct"]) - 100 * r[dates.index(d)]) for d in M if M[d]["saron_unlimited_pct"] and d in dates]
mg = np.array([x for _, x in marg]); print("SARON-mortgage minus SARON: mean %.2f%%, min %.2f, max %.2f (n=%d)" % (mg.mean(), mg.min(), mg.max(), len(mg)))
fl = [(d, float(M[d]["saron_unlimited_pct"]) - 100 * max(r[dates.index(d)], 0)) for d in M if M[d]["saron_unlimited_pct"] and d in dates]
print("SARON-mortgage minus max(SARON,0): mean %.2f%%" % np.mean([x for _, x in fl]))

# 3. Today's inputs from SNB (latest month 2026-09): margin = SARON-unlimited - max(SARON,0)
last = "2026-09"; r0 = r[-1]
offered = [float(M[last]["saron_unlimited_pct"]) / 100 - max(r[dates.index(last)], 0),
           float(M[last]["fixed_2y_pct"]) / 100, float(M[last]["fixed_5y_pct"]) / 100, float(M[last]["fixed_10y_pct"]) / 100]
print("r0 = %.3f%%, offered (margin, 2y, 5y, 10y) =" % (100 * r0), [round(100 * x, 3) for x in offered])

loan = 800_000
for name, P in [("model", m.simulate_paths(r0, p)), ("bootstrap", m.bootstrap_paths(r, r0))]:
    rates = m.tranche_rates(P, offered)
    print("\n==", name, "== SARON in 10y: p5 %.2f%% median %.2f%% p95 %.2f%%" % tuple(100 * np.percentile(P[:, -1], [5, 50, 95])))
    for nm, w in [("100% SARON", [1, 0, 0, 0]), ("100% 2y", [0, 1, 0, 0]), ("100% 5y", [0, 0, 1, 0]), ("100% 10y", [0, 0, 0, 1]), ("50/50 SARON-10y", [.5, 0, 0, .5])]:
        s = m.mix_stats(np.array(w, float), rates, loan, yearly_budget=25_000)
        print("  %-16s mean %7.0f  std %6.0f  p95 %7.0f  P(year>25k) %.3f" % (nm, s["mean"], s["std"], s["p95"], s["p_trouble"]))
    if name == "model":
        R_model = rates
        for lam in [1e-6, 1e-5, 5e-5, 2e-4]:
            print("  lambda", lam, "grid w>=0:", m.grid_mix(rates, lam, loan), "closed form:", m.closed_form_mix(rates, lam, loan).round(2))

# Figure: SARON history + fan of simulated paths
P = m.simulate_paths(r0, p, n=2000)
fig, ax = plt.subplots(figsize=(11, 4.5))
x = np.arange(len(r)); ax.plot(x, 100 * r, c="k", lw=1.5, label="SARON (SNB, monthly)")
xf = np.arange(len(r) - 1, len(r) + 120)
for q, a in [((5, 95), .15), ((25, 75), .3)]:
    lo, hi = np.percentile(P, q, axis=0); ax.fill_between(xf, 100 * lo, 100 * hi, color="#1f5fbf", alpha=a, lw=0)
ax.plot(xf, 100 * np.median(P, axis=0), c="#1f5fbf", label="simulated median, 25-75% and 5-95% bands")
ax.axhline(0, c="red", ls="--", lw=1, label="floor applied to SARON mortgages")
tk = list(range(0, len(r) + 120, 60)); ax.set_xticks(tk, [(dates[i] if i < len(r) else "%d" % (2026 + (i - len(r)) // 12 + 1)) for i in tk])
ax.set_ylabel("% per year"); ax.set_title("SARON history 1999-2026 and 10-year Monte Carlo (AR(1) estimated by OLS)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("../saron_history_and_simulation.png", dpi=150)
