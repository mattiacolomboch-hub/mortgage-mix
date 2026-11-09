"""Validation of the model and the optimiser (proposal section 8). Prints one block per test."""
import csv, numpy as np, model as m
S = [x for x in csv.DictReader(open("../data/clean/saron_monthly.csv")) if x["saron_pct"]]
months_s = [x["month"] for x in S]
r = np.array([float(x["saron_pct"]) for x in S]) / 100; r0 = r[-1]
MR = {x["month"]: x for x in csv.DictReader(open("../data/clean/mortgage_rates_monthly.csv"))}
Mo = MR["2026-09"]
offered = [float(Mo["saron_unlimited_pct"]) / 100] + [float(Mo[f"fixed_{k}y_pct"]) / 100 for k in (2, 5, 10)]
price, own, income = 1_000_000, 200_000, 180_000; loan = price - own
budget = m.interest_budget(price, loan, income)
names = ["SARON", "2y", "5y", "10y"]
lab = lambda w: " / ".join(f"{round(100*x)}% {n}" for x, n in zip(w, names) if x > 1e-9)
LAM = 2e-5

def best_mix(rates, lam=LAM, bud=budget, ln=loan):
    W, mu, sd, p95, ptr = m.frontier_table(rates, ln, bud)
    obj = np.where(ptr <= 0.05, mu + lam * sd**2, np.inf); i = int(np.argmin(obj))
    return W[i], mu[i], sd[i], p95[i], ptr[i]

P, info = m.policy_move_paths(r, r0, n=10_000)
rates = m.tranche_rates(P, offered)
C = loan * m.total_cost_per_chf(rates)                     # (paths, 4) CHF

print("== V1 Optimiser: hand formula vs fine grid (SARON + 10y only) ==")
muS, sdS, cf = C[:, 0].mean(), C[:, 0].std(ddof=1), C[:, 3].mean()
for lam in (1e-5, 2e-5, 5e-5):
    w_hand = min(1, max(0, (cf - muS) / (2 * lam * sdS**2)))
    ws = np.linspace(0, 1, 1001); obj = [(C[:, 0] * w + C[:, 3] * (1 - w)).mean() + lam * (C[:, 0] * w + C[:, 3] * (1 - w)).var(ddof=1) for w in ws]
    print(f"lambda {lam:.0e}: formula w* = {w_hand:.3f}   grid (0.1% steps) = {ws[int(np.argmin(obj))]:.3f}")

print("\n== V2 Lagrange closed form (shorting allowed) vs constrained optimum, lambda 2e-5 ==")
print("closed form:", np.round(m.closed_form_mix(rates, LAM, loan), 2), "(negative weights -> need w >= 0)")
print("constrained:", lab(best_mix(rates)[0]))

print("\n== V3 Monte Carlo precision ==")
w, mu, sd, p95, ptr = best_mix(rates)
print(f"recommended {lab(w)}: mean {mu:,.0f}, SE of mean = sd/sqrt(N) = {sd:,.0f}/sqrt(10000) = {sd/100:,.0f} CHF")
p = 0.006; print(f"SE of P(trouble) for p = {p}: sqrt(p(1-p)/N) = {np.sqrt(p*(1-p)/10_000):.4f}")
picks = []
for seed in range(5):
    Ps, _ = m.policy_move_paths(r, r0, n=10_000, seed=seed + 1)
    ws_, mus, *_ = best_mix(m.tranche_rates(Ps, offered)); picks.append(f"seed {seed+1}: {lab(ws_)} (mean {mus:,.0f})")
print("\n".join(picks))

print("\n== V4 Second rate model: same question with monthly AR(1) (normal and bootstrap) ==")
par = m.estimate_ar1(r)
for nm, R in [("policy move-or-hold", P), ("AR(1) normal", m.simulate_paths(r0, par, n=10_000)), ("AR(1) residual bootstrap", m.residual_bootstrap_paths(r, r0, par, n=10_000))]:
    rt = m.tranche_rates(R, offered); Cc = loan * m.total_cost_per_chf(rt)
    print(f"{nm:26s} 100% SARON mean {Cc[:,0].mean():8,.0f} p95 {np.percentile(Cc[:,0],95):8,.0f} | best mix: {lab(best_mix(rt)[0])}")
Pa, _ = m.policy_move_paths(r, r0, n=10_000, theta=par["theta"])
print(f"{'policy, theta = AR(1) ' + format(100*par['theta'], '.2f') + '%':26s} best mix: {lab(best_mix(m.tranche_rates(Pa, offered))[0])}")

print("\n== V5 Benchmark: UBS CIO table (CHF 1M, 10y), with UBS-like rates 10y 1.71%, SARON margin 0.75% ==")
rt = m.tranche_rates(P, [0.0075, offered[1], offered[2], 0.0171]); Cu = 1_000_000 * m.total_cost_per_chf(rt)
q = np.percentile(Cu[:, 0], [5, 50, 95])
print(f"10y fixed: ours {Cu[:,3].mean()/1e3:.0f}k vs UBS 171k")
print(f"SARON: ours p5 / median / mean / p95 = {q[0]/1e3:.0f} / {q[1]/1e3:.0f} / {Cu[:,0].mean()/1e3:.0f} / {q[2]/1e3:.0f}k  vs UBS low / base / high = 98 / 167 / 257k")
print(f"share of our paths below UBS low (98k): {(Cu[:,0] < 98e3).mean():.1%}, above UBS high (257k): {(Cu[:,0] > 257e3).mean():.1%}")

print("\n== V6 Backtest on history: realised 10y cost of CHF 800k, starts 2007-12 to 2016-09 ==")
idx = {mo: i for i, mo in enumerate(months_s)}; margin = 0.0106
rows = []
for mo in sorted(MR):
    if mo < "2007-12" or mo > "2016-09" or not MR[mo]["fixed_10y_pct"]: continue
    i = idx[mo]; path = r[i:i + 120]
    c_s = loan * (np.maximum(path, 0) + margin).sum() / 12
    c_f = loan * float(MR[mo]["fixed_10y_pct"]) / 100 * 10
    rows.append((mo, c_s, c_f))
cs, cf_ = np.array([x[1] for x in rows]), np.array([x[2] for x in rows])
print(f"{len(rows)} start months. SARON cheaper in {np.mean(cs < cf_):.0%} of them. Average saving {np.mean(cf_ - cs):,.0f} CHF")
for mo, a, b in rows[::24]: print(f"  start {mo}: SARON {a:9,.0f}  10y fixed {b:9,.0f}  50/50 {(a+b)/2:9,.0f}")

print("\n== V7 Extreme and invalid inputs ==")
for nm, pr, ow, inc in [("own funds 10%", 1e6, 1e5, 180e3), ("income 120k", 1e6, 2e5, 120e3), ("own funds 100%", 1e6, 1e6, 180e3), ("income 0", 1e6, 2e5, 0)]:
    b = m.bank_affordability(pr, ow, inc) if inc > 0 else None
    msgs = []
    if ow < 0.2 * pr: msgs.append("own funds < 20%")
    if pr - ow <= 0: msgs.append("no loan needed")
    if inc <= 0: msgs.append("income must be positive")
    elif b["ratio"] > 1/3: msgs.append(f"bank rule fails ({100*b['ratio']:.0f}% > 33%)")
    print(f"{nm:16s} -> app stops with: {', '.join(msgs) or 'NO ERROR'}")
for nm, kw, offs in [("long-run level 3%", dict(theta=0.03), offered), ("long-run level -0.75%", dict(theta=-0.0075), offered), ("start at 1.75% (2023)", dict(), offered)]:
    start = 0.0175 if "2023" in nm else r0
    Pe, _ = m.policy_move_paths(r, start, n=5000, **kw); we, mue, sde, p95e, pte = best_mix(m.tranche_rates(Pe, offs))
    print(f"{nm:22s} -> {lab(we):22s} mean {mue:,.0f} p95 {p95e:,.0f} P(trouble) {pte:.1%}")
tight = m.interest_budget(1e6, 8e5, 150e3)        # income that only just passes a lower budget
for inc in (150e3, 140e3):
    bud = m.interest_budget(1e6, 8e5, inc); W_, mu_, sd_, p95_, pt_ = m.frontier_table(rates, loan, bud)
    print(f"income {inc/1e3:.0f}k (budget {bud:,.0f}/yr): P(trouble) 100% SARON {pt_[np.argmin(np.abs(W_-[1,0,0,0]).sum(1))]:.1%} | cheapest mix allowed (lambda 0): {lab(best_mix(rates, 0, bud)[0])}")
