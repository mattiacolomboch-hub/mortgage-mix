"""Answers the project question for one example buyer, with real SNB inputs."""
import csv, numpy as np, model as m
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = [x for x in csv.DictReader(open("../data/clean/saron_monthly.csv")) if x["saron_pct"]]
r = np.array([float(x["saron_pct"]) for x in S]) / 100; r0 = r[-1]
Mo = {x["month"]: x for x in csv.DictReader(open("../data/clean/mortgage_rates_monthly.csv"))}["2026-09"]
offered = [float(Mo["saron_unlimited_pct"]) / 100, float(Mo["fixed_2y_pct"]) / 100, float(Mo["fixed_5y_pct"]) / 100, float(Mo["fixed_10y_pct"]) / 100]
price, own, income = 1_000_000, 200_000, 180_000
loan = price - own
budget = m.interest_budget(price, loan, income)
print("Bank check:", m.bank_affordability(price, own, income), "| yearly interest budget %.0f" % budget)
P, info = m.policy_move_paths(r, r0, n=5000)
rates = m.tranche_rates(P, offered)
W, mu, sd, p95, ptr = m.frontier_table(rates, loan, budget)
ok = ptr <= 0.05
eff = []                                                   # efficient = no other mix cheaper AND less risky
for i in range(len(W)):
    if not np.any((mu < mu[i] - 1e-9) & (p95 <= p95[i] + 1e-9)): eff.append(i)
eff = sorted(eff, key=lambda i: p95[i])
names = ["SARON", "2y", "5y", "10y"]
lab = lambda w: " / ".join(f"{int(round(100*x))}% {n}" for x, n in zip(w, names) if x > 0)
picks = {}
for tag, lam in [("Cheapest acceptable", 0.0), ("Balanced", 2e-5), ("Safest", 1e-2)]:
    obj = np.where(ok, mu + lam * sd**2, np.inf); i = int(np.argmin(obj)); picks[tag] = i
    print(f"{tag:20s} {lab(W[i]):35s} mean {mu[i]:7.0f}  std {sd[i]:6.0f}  p95 {p95[i]:7.0f}  P(trouble) {ptr[i]:.3f}")
for nm, w in [("Bank 50/50", [.5, 0, 0, .5]), ("100% SARON", [1, 0, 0, 0]), ("100% 10y", [0, 0, 0, 1])]:
    i = int(np.argmin(np.abs(W - np.array(w)).sum(axis=1)))
    print(f"{nm:20s} {lab(W[i]):35s} mean {mu[i]:7.0f}  std {sd[i]:6.0f}  p95 {p95[i]:7.0f}  P(trouble) {ptr[i]:.3f}")
fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.3, 1]})
a = ax[0]; a.scatter(p95[~ok] / 1e3, mu[~ok] / 1e3, s=6, c="#f4b4b4", label="mixes with P(unaffordable) > 5%")
a.scatter(p95[ok] / 1e3, mu[ok] / 1e3, s=6, c="#cfd6e0", label="all other mixes (5% grid)")
a.plot(p95[eff] / 1e3, mu[eff] / 1e3, c="#1f5fbf", lw=2.5, label="efficient frontier")
cols = {"Cheapest acceptable": "#16a34a", "Balanced": "#7c3aed", "Safest": "#dc2626"}
for t, i in picks.items(): a.scatter(p95[i] / 1e3, mu[i] / 1e3, s=130, c=cols[t], edgecolor="w", zorder=5); a.annotate(f"{t}\n{lab(W[i])}", (p95[i] / 1e3, mu[i] / 1e3), xytext=(8, -4), textcoords="offset points", fontsize=8, color=cols[t])
i5 = int(np.argmin(np.abs(W - np.array([.5, 0, 0, .5])).sum(axis=1))); a.scatter(p95[i5] / 1e3, mu[i5] / 1e3, marker="X", s=90, c="#d97706", zorder=5); a.annotate("bank 50/50", (p95[i5] / 1e3, mu[i5] / 1e3), xytext=(6, 6), textcoords="offset points", fontsize=8, color="#d97706")
a.set_xlabel("Bad case: 95th percentile of 10-year interest (kCHF)"); a.set_ylabel("Expected 10-year interest (kCHF)"); a.grid(alpha=.3); a.legend(fontsize=8, loc="upper right")
a.set_title("Which mix? CHF 800k mortgage, income CHF 180k (real SNB inputs, Sep 2026)", fontsize=10)
b = ax[1]
for j, (t, i) in enumerate(picks.items()):
    b.bar(j, mu[i] / 1e3, color=cols[t], alpha=.85, width=.55); b.errorbar(j, mu[i] / 1e3, yerr=[[max(0,(mu[i] - np.percentile(loan * m.total_cost_per_chf(rates) @ W[i], 5)) / 1e3)], [max(0,(p95[i] - mu[i]) / 1e3)]], fmt="none", ecolor="k", capsize=8, lw=2)
    b.text(j, 5, lab(W[i]).replace(" / ", "\n"), ha="center", fontsize=8, color="w", fontweight="bold")
b.set_xticks(range(3), list(picks), fontsize=9); b.set_ylabel("10-year interest (kCHF)"); b.set_title("Expected cost and 5%-95% range", fontsize=10); b.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig("../decision_example.png", dpi=150)
print("efficient mixes with 2y or 5y > 0:", [lab(W[i]) for i in eff if W[i][1] + W[i][2] > 0])
print("n efficient:", len(eff))
