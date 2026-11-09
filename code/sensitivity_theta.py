"""Sensitivity of the decision to the key assumption: long-run SARON level theta."""
import csv, numpy as np, model as m
S = [x for x in csv.DictReader(open("../data/clean/saron_monthly.csv")) if x["saron_pct"]]
r = np.array([float(x["saron_pct"]) for x in S]) / 100; r0 = r[-1]
Mo = {x["month"]: x for x in csv.DictReader(open("../data/clean/mortgage_rates_monthly.csv"))}["2026-09"]
offered = [float(Mo[k]) / 100 for k in ["saron_unlimited_pct", "fixed_2y_pct", "fixed_5y_pct", "fixed_10y_pct"]]
price, own, income = 1_000_000, 200_000, 180_000; loan = price - own
budget = m.interest_budget(price, loan, income)
names = ["SARON", "2y", "5y", "10y"]; lab = lambda w: " / ".join(f"{int(round(100*x))}% {n}" for x, n in zip(w, names) if x > 0)
print("theta | E[SARON] in 10y | 100% SARON mean, p95 | 10y | optimal mix (lambda 2e-5) | breakeven")
for th in [0.0, 0.005, None, 0.02]:
    P, info = m.policy_move_paths(r, r0, n=5000, theta=th)
    rates = m.tranche_rates(P, offered)
    W, mu, sd, p95, ptr = m.frontier_table(rates, loan, budget)
    i = int(np.argmin(np.where(ptr <= .05, mu + 2e-5 * sd**2, np.inf)))
    s = 0; f = 3
    iS = int(np.argmin(np.abs(W - [1,0,0,0]).sum(1))); iF = int(np.argmin(np.abs(W - [0,0,0,1]).sum(1)))
    print("%5.2f%% | %5.2f%% | %7.0f %7.0f | %7.0f | %-26s | SARON cheaper on average: %s" % (100*info["theta"], 100*P[:, -1].mean(), mu[iS], p95[iS], mu[iF], lab(W[i]), mu[iS] < mu[iF]))
