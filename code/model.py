"""
Optimal Swiss mortgage mix: core model (no UI).
Steps: load SARON -> estimate AR(1) by OLS -> Monte Carlo paths -> tranche costs
-> risk measures -> optimal mix (closed form + constrained grid search).
All rates in decimals per year (0.015 = 1.5%). Money in CHF.
"""
import csv, io, itertools
import numpy as np

TRANCHES = ["SARON", "2y fixed", "5y fixed", "10y fixed"]
TERMS_M = [None, 24, 60, 120]          # fixed-rate term in months (None = SARON)

# ---------- 1. Data ----------
def load_snb_csv(path, keyword="SARON"):
    """Read an SNB data-portal CSV (semicolon separated, header block first).
    Returns (dates, values in decimals) for rows whose dimension contains `keyword`."""
    txt = open(path, encoding="utf-8-sig").read()
    lines = txt.splitlines()
    start = next(i for i, l in enumerate(lines) if l.replace('"', '').startswith("Date"))
    rows = list(csv.reader(io.StringIO("\n".join(lines[start:])), delimiter=";"))
    header, data = rows[0], rows[1:]
    out = []
    for r in data:
        if len(r) < len(header) or r[-1] in ("", None):
            continue
        dims = " ".join(r[1:-1])
        if keyword.upper() in dims.upper() or len(header) == 2:
            out.append((r[0], float(r[-1]) / 100))
    out.sort()
    return [d for d, _ in out], np.array([v for _, v in out])

# ---------- 2. Rate model: r(t+1) - r(t) = a + b r(t) + e ----------
def estimate_ar1(r):
    """OLS of monthly changes on the level. Returns a, b, sigma, plus
    implied long-run mean theta = -a/b and monthly mean reversion kappa = -b."""
    y = np.diff(r)
    X = np.column_stack([np.ones(len(r) - 1), r[:-1]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)      # (X'X)^-1 X'y
    resid = y - X @ beta
    sigma = np.sqrt(resid @ resid / (len(y) - 2))
    a, b = beta
    return dict(a=a, b=b, sigma=sigma, kappa=-b, theta=-a / b if b != 0 else np.nan)

def simulate_paths(r0, params, months=120, n=10_000, seed=0):
    """Monte Carlo: n paths of monthly SARON, shape (n, months+1)."""
    rng = np.random.default_rng(seed)
    a, b, s = params["a"], params["b"], params["sigma"]
    R = np.empty((n, months + 1)); R[:, 0] = r0
    eps = rng.normal(0, s, (n, months))
    for t in range(months):
        R[:, t + 1] = R[:, t] + a + b * R[:, t] + eps[:, t]
    return R

def bootstrap_paths(r_hist, r0, months=120, n=10_000, seed=0):
    """Second method: resample historical monthly changes."""
    rng = np.random.default_rng(seed)
    d = np.diff(r_hist)
    steps = rng.choice(d, size=(n, months), replace=True)
    return np.column_stack([np.full(n, r0), r0 + np.cumsum(steps, axis=1)])

# ---------- 3. Tranche costs ----------
def tranche_rates(R, offered, months=120):
    """Monthly rate paid by each tranche on each path, shape (n, months, 4).
    offered = [saron_margin, fixed2y, fixed5y, fixed10y] (today's offers).
    SARON tranche: max(SARON,0) + margin (contractual floor).
    Fixed tranche renews at maturity at today's offer shifted by the change in SARON
    since today (parallel-shift assumption: a stated limitation)."""
    n = R.shape[0]
    floor = np.maximum(R[:, :months], 0)
    out = np.empty((n, months, 4))
    out[:, :, 0] = floor + offered[0]
    for k, term in enumerate(TERMS_M[1:], start=1):
        rate = np.empty((n, months))
        for start in range(0, months, term):
            shift = R[:, start] - R[:, 0]
            fixed = np.maximum(offered[k] + shift, 0.0)
            rate[:, start:start + term] = fixed[:, None]
        out[:, :, k] = rate
    return out

def total_cost_per_chf(rates):
    """Total interest over the horizon per CHF 1 of loan, shape (n, 4)."""
    return rates.sum(axis=1) / 12

# ---------- 4. Risk measures for a mix ----------
def mix_stats(w, rates, loan, yearly_budget=None):
    """w: weights (4,). Returns mean, std, 95th pct of total cost (CHF) and
    P(any year's interest > yearly_budget)."""
    cost = loan * total_cost_per_chf(rates) @ w
    res = dict(mean=cost.mean(), std=cost.std(ddof=1), p95=np.percentile(cost, 95))
    if yearly_budget is not None:
        n, m, _ = rates.shape
        yearly = loan * (rates @ w).reshape(n, m // 12, 12).sum(axis=2) / 12
        res["p_trouble"] = (yearly.max(axis=1) > yearly_budget).mean()
    return res

# ---------- 5. Optimisation ----------
def closed_form_mix(rates, lam, loan, riskless=3):
    """min mu'w + lam w'Sigma w  s.t. 1'w = 1 (shorting allowed).
    Over a 10y horizon the 10y fixed tranche has zero variance (riskless debt),
    so Sigma is singular; treat it like the risk-free asset in Markowitz:
    w_risky = Sigma_r^-1 (c_f 1 - mu_r) / (2 lam), w_f = 1 - sum(w_risky)."""
    C = loan * total_cost_per_chf(rates)      # CHF, same units as grid_mix
    risky = [j for j in range(C.shape[1]) if j != riskless]
    mu_r, c_f = C[:, risky].mean(axis=0), C[:, riskless].mean()
    S_r = np.cov(C[:, risky], rowvar=False)
    w_r = np.linalg.solve(S_r, c_f - mu_r) / (2 * lam)
    w = np.zeros(C.shape[1]); w[risky] = w_r; w[riskless] = 1 - w_r.sum()
    return w

def grid_mix(rates, lam, loan, yearly_budget=None, max_p=0.05, step=0.05):
    """Constrained version: w >= 0, sum 1, P(trouble) <= max_p. Brute-force grid
    (also used to validate the closed form when the constraints are not binding)."""
    C = loan * total_cost_per_chf(rates)
    k = int(round(1 / step)); best = None
    for a, b, c in itertools.product(range(k + 1), repeat=3):
        if a + b + c > k: continue
        w = np.array([a, b, c, k - a - b - c]) / k
        cost = C @ w
        obj = cost.mean() + lam * cost.var(ddof=1)
        if yearly_budget is not None:
            if mix_stats(w, rates, loan, yearly_budget)["p_trouble"] > max_p: continue
        if best is None or obj < best[0]: best = (obj, w)
    return None if best is None else best[1]

# ---------- 6. Affordability rule (Swiss bank practice, eligibility check) ----------
def bank_affordability(price, own_funds, income, theo_rate=0.05, maint=0.01, amort_years=15):
    loan = price - own_funds
    second = max(0.0, loan - 2 / 3 * price)
    yearly = loan * theo_rate + price * maint + second / amort_years
    return dict(loan=loan, ltv=loan / price, ratio=yearly / income,
                ok=(own_funds >= 0.2 * price) and (yearly / income <= 1 / 3))

# ---------- 2b. Better path generators ----------
def residual_bootstrap_paths(r, r0, params, months=120, n=10_000, seed=0):
    """Fixed bootstrap: keep the AR(1) mean reversion, but resample the model's
    historical residuals instead of drawing normal shocks (keeps fat tails/jumps)."""
    rng = np.random.default_rng(seed)
    a, b = params["a"], params["b"]
    resid = np.diff(r) - (a + b * r[:-1])
    R = np.empty((n, months + 1)); R[:, 0] = r0
    E = rng.choice(resid, size=(n, months), replace=True)
    for t in range(months):
        R[:, t + 1] = R[:, t] + a + b * R[:, t] + E[:, t]
    return R

def policy_step_paths(r, r0, months=120, n=10_000, seed=0, step=0.0025, theta=None, meetings=(2, 5, 8, 11)):
    """SNB-style model: SARON only moves at quarterly policy meetings, in steps of 0.25%.
    Quarterly AR(1) estimated by OLS on end-of-quarter SARON; the simulated change is
    rounded to the nearest 0.25% step, so paths look like real policy staircases.
    theta (long-run level) can be overridden, e.g. by a market-implied value."""
    rng = np.random.default_rng(seed)
    q = r[2::3]                                    # one observation per quarter
    p = estimate_ar1(q)
    kappa, sig = p["kappa"], p["sigma"]
    th = p["theta"] if theta is None else theta
    R = np.empty((n, months + 1)); R[:, 0] = r0
    level = np.full(n, r0)
    for t in range(months):
        if t % 3 == 2:                             # meeting month
            move = kappa * (th - level) + rng.normal(0, sig, n)
            level = level + step * np.round(move / step)
        R[:, t + 1] = level
    return R, dict(kappa_q=kappa, sigma_q=sig, theta=th)

def policy_move_paths(r, r0, months=120, n=10_000, seed=0, step=0.0025, theta=None, thr=0.00125, lower=-0.0075):
    """Improved SNB-style model ("move or hold"), estimated on end-of-quarter SARON:
    1) p_move = share of quarters with |change| > 0.125% (estimated);
    2) on move quarters, OLS of the change on the level: change = c + d*level + e
       (d < 0 = mean reversion; theta = -c/d unless overridden);
    3) simulated move rounded to 0.25% steps, never zero when a move happens."""
    rng = np.random.default_rng(seed)
    q = r[2::3]; dq = np.diff(q); lev = q[:-1]
    mv = np.abs(dq) > thr
    p_move = mv.mean()
    X = np.column_stack([np.ones(mv.sum()), lev[mv]])
    (c, d), *_ = np.linalg.lstsq(X, dq[mv], rcond=None)
    sig = np.std(dq[mv] - X @ np.array([c, d]), ddof=2)
    th = -c / d if theta is None else theta
    R = np.empty((n, months + 1)); R[:, 0] = r0; level = np.full(n, r0)
    for t in range(months):
        if t % 3 == 2:
            go = rng.random(n) < p_move
            x = d * (level - th) + rng.normal(0, sig, n)
            k = np.round(x / step); k[k == 0] = np.sign(x[k == 0]); k[k == 0] = 1
            level = np.maximum(level + go * step * k, lower)   # SNB never went below -0.75%
        R[:, t + 1] = level
    return R, dict(p_move=p_move, c=c, d=d, sigma_move=sig, theta=th)

def interest_budget(price, loan, income, maint=0.01, amort_years=15):
    """Yearly interest the household can pay while total housing costs stay <= 1/3 of income
    (same structure as the bank rule, but with the simulated actual rate instead of 5%)."""
    second = max(0.0, loan - 2 / 3 * price)
    return income / 3 - maint * price - second / amort_years

def all_mixes(step=0.05):
    k = int(round(1 / step))
    return np.array([[a, b, c, k - a - b - c] for a, b, c in itertools.product(range(k + 1), repeat=3) if a + b + c <= k]) / k

def frontier_table(rates, loan, yearly_budget, step=0.05):
    """Stats for every mix on a 5% grid: mean, std, p95 of total cost, P(trouble)."""
    W = all_mixes(step)
    C = loan * total_cost_per_chf(rates) @ W.T                       # (n_paths, n_mixes)
    n, mth, _ = rates.shape
    yearly_tr = loan * rates.reshape(n, mth // 12, 12, -1).sum(axis=2) / 12   # (paths, years, tranches)
    yearly = yearly_tr @ W.T                                                   # (paths, years, mixes), linear in w
    ptr = (yearly.max(axis=1) > yearly_budget).mean(axis=0)
    return W, C.mean(axis=0), C.std(axis=0, ddof=1), np.percentile(C, 95, axis=0), ptr
