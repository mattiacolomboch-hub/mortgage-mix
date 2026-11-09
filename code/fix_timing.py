"""SARON now, when do I fix? Backward induction on a rate lattice built from the SNB move-or-hold model.

State: SARON level s on a 0.25% grid, quarter t = 0..T-1.
Each quarter the borrower (still on SARON) chooses: stay, or lock the whole loan in a k-year fixed (k = 2, 5, 10).
Bellman equation (risk-neutral, a = 0):
    V(t,s) = min( stay:   L (max(s,0)+margin)/4           + sum_s' Q[s,s'] V(t+1,s'),
                  fix k:  L f_k(s)/4 x min(4k, T-t)        + sum_s' Q^(4k)[s,s'] V(t+4k,s') )
With risk aversion a > 0 the expectation E[V] is replaced by the certainty equivalent (1/a) ln E[exp(a V)].
"""
import csv, os, numpy as np
from math import erf, sqrt
import model as m
HERE = os.path.dirname(os.path.abspath(__file__))
STEP = 0.0025
Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))

def load():
    S = {x["month"]: float(x["saron_pct"]) / 100 for x in csv.DictReader(open(os.path.join(HERE, "../data/clean/saron_monthly.csv"))) if x["saron_pct"]}
    MR = [x for x in csv.DictReader(open(os.path.join(HERE, "../data/clean/mortgage_rates_monthly.csv"))) if x["fixed_10y_pct"] and x["month"] in S]
    return np.array(list(S.values())), MR, S

def fixed_rate_slopes(MR, S):
    """OLS of each published fixed rate on SARON (SNB, 2007-12 onwards): f_k = a_k + b_k SARON."""
    s = np.array([S[x["month"]] for x in MR]); X = np.column_stack([np.ones_like(s), s]); out = {}
    for k in (2, 5, 10):
        f = np.array([float(x[f"fixed_{k}y_pct"]) for x in MR]) / 100
        (a, b), *_ = np.linalg.lstsq(X, f, rcond=None); e = f - X @ [a, b]
        out[k] = dict(a=a, b=b, r2=1 - e.var() / f.var())
    return out

def lattice(r, lo=-0.0075, hi=0.05, theta=None):
    """Grid of SARON levels and the exact one-quarter transition matrix Q of the move-or-hold model."""
    _, info = m.policy_move_paths(r, r[-1], months=3, n=10)          # reuse the OLS estimates
    p, d, sig = info["p_move"], info["d"], info["sigma_move"]
    th = info["theta"] if theta is None else theta
    grid = np.round(np.arange(lo, hi + 1e-9, STEP), 6); n = len(grid)
    Q = np.zeros((n, n))
    for i, s in enumerate(grid):
        mu = d * (s - th)                                            # mean of x = d (s - theta) + e
        Q[i, i] += 1 - p                                             # hold
        for k in range(-n, n + 1):
            if k == 0: continue
            a, b = (k - 0.5) * STEP, (k + 0.5) * STEP
            if k == 1: a = 0.0                                       # a rounded-to-zero move becomes +-1 step
            if k == -1: b = 0.0
            pr = Phi((b - mu) / sig) - Phi((a - mu) / sig)
            j = min(max(i + k, 0), n - 1)                            # floor -0.75%, cap at hi
            Q[i, j] += p * pr
    return grid, Q / Q.sum(1, keepdims=True), dict(p_move=p, d=d, sigma=sig, theta=th)

def solve(grid, Q, s0, loan, margin, offers, slopes, years=10, a=0.0):
    """offers = {k: today's k-year rate}. Returns V (T+1, n) and policy (T, n): 0 stay, else k."""
    T, n = 4 * years, len(grid)
    Qp = {k: np.linalg.matrix_power(Q, 4 * k) for k in offers}
    ce = (lambda P, v: P @ v) if a == 0 else (lambda P, v: (np.log(P @ np.exp(a * (v - v.min()))) / a + v.min()))
    f = {k: np.maximum(offers[k] + slopes[k]["b"] * (grid - s0), 0) for k in offers}
    V = np.zeros((T + 1, n)); pol = np.zeros((T, n), int)
    for t in range(T - 1, -1, -1):
        stay = loan * (np.maximum(grid, 0) + margin) / 4 + ce(Q, V[t + 1])
        best, act = stay, np.zeros(n, int)
        for k in offers:
            q = 4 * k; nq = min(q, T - t)
            val = loan * f[k] / 4 * nq + (ce(Qp[k], V[t + q]) if t + q < T else 0)
            better = val < best - 1e-6; best = np.where(better, val, best); act = np.where(better, k, act)
        V[t], pol[t] = best, act
    return V, pol, f

def simulate_policy(grid, Q, pol, f, i0, loan, margin, years=10, n=20000, seed=0):
    """Monte Carlo check: follow the optimal rule on random lattice paths."""
    rng = np.random.default_rng(seed); T = 4 * years; C = np.cumsum(Q, 1)
    cost = np.zeros(n); first_fix = np.full(n, -1); first_k = np.zeros(n, int)
    for p in range(n):
        i, t = i0, 0
        while t < T:
            k = pol[t, i]
            if k == 0:
                cost[p] += loan * (max(grid[i], 0) + margin) / 4; steps = 1
            else:
                steps = min(4 * k, T - t); cost[p] += loan * f[k][i] / 4 * steps
                if first_fix[p] < 0: first_fix[p], first_k[p] = t, k
            for _ in range(steps): i = int(np.searchsorted(C[i], rng.random()))
            t += steps
    return cost, first_fix, first_k

if __name__ == "__main__":
    r, MR, S = load(); s0 = r[-1]; Mo = MR[-1]
    slopes = fixed_rate_slopes(MR, S)
    for k, v in slopes.items(): print(f"{k:2d}y fixed = {100*v['a']:.2f}% + {v['b']:.2f} x SARON   (R2 {v['r2']:.2f})")
    grid, Q, info = lattice(r)
    i0 = int(np.argmin(np.abs(grid - s0)))
    s0g = grid[i0]
    print(f"lattice: {len(grid)} levels, p_move {info['p_move']:.3f}, theta {100*info['theta']:.2f}%, start {100*s0g:.2f}%")
    loan = 800_000; margin = float(Mo["saron_unlimited_pct"]) / 100 - max(s0, 0)
    offers = {k: float(Mo[f"fixed_{k}y_pct"]) / 100 for k in (2, 5, 10)}
    print("offers today:", {k: f"{100*v:.2f}%" for k, v in offers.items()}, f"SARON margin {100*margin:.2f}%")
    for a in (0.0, 1e-5, 3e-5):
        V, pol, f = solve(grid, Q, s0g, loan, margin, offers, slopes, a=a)
        rule = []
        for t in (0, 4, 8, 16, 24, 32):
            row = pol[t]; fix_lv = [grid[j] for j in range(len(grid)) if row[j] > 0]
            rule.append(f"y{t//4}: fix if SARON >= {100*min(fix_lv):.2f}% ({row[[j for j in range(len(grid)) if row[j]>0][0]]}y)" if fix_lv else f"y{t//4}: never fix")
        print(f"\na = {a:g}: action today = {'stay on SARON' if pol[0,i0]==0 else 'fix '+str(pol[0,i0])+'y'}, value today {V[0,i0]:,.0f}")
        print("   " + " | ".join(rule))
        cost, ff, fk = simulate_policy(grid, Q, pol, f, i0, loan, margin, n=5000)
        print(f"   MC check: mean cost {cost.mean():,.0f} (DP {V[0,i0] if a==0 else float('nan'):,.0f}), p95 {np.percentile(cost,95):,.0f}, P(fix within 10y) {np.mean(ff>=0):.0%}, "
              f"median time to fix {np.median(ff[ff>=0])/4 if np.any(ff>=0) else float('nan'):.1f}y, tranche chosen: " +
              ", ".join(f"{k}y {np.mean(fk==k):.0%}" for k in (2,5,10)))
    # benchmarks (risk-neutral)
    print("\nBenchmarks: fix 10y today =", f"{loan*offers[10]*10:,.0f}")
