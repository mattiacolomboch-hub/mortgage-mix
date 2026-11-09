import csv, numpy as np, model as m
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = [x for x in csv.DictReader(open("../data/clean/saron_monthly.csv")) if x["saron_pct"]]
r = np.array([float(x["saron_pct"]) for x in S]) / 100; r0 = r[-1]
p = m.estimate_ar1(r)
P_boot = m.residual_bootstrap_paths(r, r0, p)
P_step, info = m.policy_move_paths(r, r0)
print("policy-step params:", {k: round(v, 4) for k, v in info.items()})
P_step_mkt, info2 = m.policy_move_paths(r, r0, theta=0.0)   # scenario: SNB keeps rates near 0
for nm, P in [("AR1 normal", m.simulate_paths(r0, p)), ("AR1 resid-bootstrap", P_boot), ("policy steps (hist. theta)", P_step), ("policy steps (theta 0%)", P_step_mkt)]:
    e = P[:, -1]; print("%-28s 10y: p5 %.2f median %.2f p95 %.2f | P(SARON>1.5%% at some point) %.2f" % (nm, *(100*np.percentile(e,[5,50,95])), (P.max(axis=1) > .015).mean()))
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
n = len(r); xh = np.arange(n); xf = np.arange(n - 1, n + 120)
for ax, P, title in [(axs[0], P_step, "Historical long-run level (%.2f%%)" % (100*info["theta"])), (axs[1], P_step_mkt, "Scenario: long-run level 0% (rates stay low)")]:
    ax.plot(xh[-120:], 100 * r[-120:], c="k", lw=1.6, label="SARON 2016-2026 (SNB)")
    lo, hi = np.percentile(P, [5, 95], axis=0); ax.fill_between(xf, 100*lo, 100*hi, color="#1f5fbf", alpha=.12, lw=0, label="5-95% of 10,000 paths")
    ax.plot(xf, 100 * P.mean(axis=0), c="#1f5fbf", lw=2.5, label="expected SARON")
    for i, c in zip([3, 11, 27, 40], ["#16a34a", "#dc2626", "#d97706", "#7c3aed"]): ax.step(xf, 100 * P[i], where="post", c=c, lw=1.2, alpha=.9)
    ax.axhline(0, c="red", ls="--", lw=.8); ax.set_title(title, fontsize=11); ax.grid(alpha=.3)
    tk = [n-120, n-60, n, n+60, n+119]; ax.set_xticks(tk, ["2016", "2021", "2026", "2031", "2036"])
axs[0].set_ylabel("SARON, % per year"); axs[0].legend(fontsize=8, loc="upper left")
fig.suptitle("SNB-style simulation: SARON moves only at quarterly SNB meetings (moves in ~35% of quarters, 0.25% steps) (4 example paths in colour)", fontsize=11)
plt.tight_layout(); plt.savefig("../saron_policy_step_simulation.png", dpi=150)
