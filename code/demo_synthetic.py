"""Runs the whole pipeline on SYNTHETIC data until the real SNB file is loaded."""
import numpy as np, model as m
rng = np.random.default_rng(1)
r = [0.005]
for _ in range(300): r.append(r[-1] + 0.0002 - 0.01 * r[-1] + rng.normal(0, 0.0015))
r = np.array(r)
p = m.estimate_ar1(r); print("AR(1):", {k: round(v, 5) for k, v in p.items()})
R = m.simulate_paths(r0=0.0, params=p)
offered = [0.0075, 0.0123, 0.0146, 0.0166]          # Swissquote 6 Nov 2026 (SARON margin approx.)
rates = m.tranche_rates(R, offered)
loan = 800_000
for name, w in [("100% SARON", [1,0,0,0]), ("50/50", [.5,0,0,.5]), ("100% 10y", [0,0,0,1])]:
    print(name, {k: round(v) if k != "p_trouble" else round(v, 3) for k, v in m.mix_stats(np.array(w, float), rates, loan, yearly_budget=20_000).items()})
for lam in [1e-5, 5e-5, 2e-4, 1e-3]:
    print("lambda", lam, "| closed form:", m.closed_form_mix(rates, lam, loan).round(2), "| grid w>=0:", m.grid_mix(rates, lam, loan))
print("grid + tight budget (16k/yr):", m.grid_mix(rates, 1e-5, loan, yearly_budget=16_000))
print("bank check:", m.bank_affordability(1_000_000, 200_000, 180_000))
