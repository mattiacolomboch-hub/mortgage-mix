"""Streamlit app: optimal Swiss mortgage mix. Run: streamlit run app.py"""
import csv, os, numpy as np, streamlit as st, matplotlib.pyplot as plt
import model as m

st.set_page_config(page_title="Mortgage mix", layout="wide")
HERE = os.path.dirname(__file__)
NAMES = ["SARON", "2y fixed", "5y fixed", "10y fixed"]
lab = lambda w: " / ".join(f"{int(round(100*x))}% {n}" for x, n in zip(w, NAMES) if x > 0.001)

@st.cache_data
def load():
    S = [x for x in csv.DictReader(open(os.path.join(HERE, "../data/clean/saron_monthly.csv"))) if x["saron_pct"]]
    r = np.array([float(x["saron_pct"]) for x in S]) / 100
    M = {x["month"]: x for x in csv.DictReader(open(os.path.join(HERE, "../data/clean/mortgage_rates_monthly.csv")))}
    last = max(k for k in M if M[k]["fixed_10y_pct"])
    return r, S[-1]["month"], M[last], last

@st.cache_data
def simulate(theta_pct, n=3000):
    r, *_ = load()
    P, info = m.policy_move_paths(r, r[-1], n=n, theta=None if theta_pct is None else theta_pct / 100)
    return P, info

r, saron_month, Mlast, mort_month = load()
st.title("How should I split my mortgage?")
st.caption(f"SARON vs 2y / 5y / 10y fixed. Data: SNB (SARON to {saron_month}, mortgage rates to {mort_month}). "
           "Results are simulations of possible futures, not forecasts.")

with st.sidebar:
    st.header("Your situation")
    price = st.number_input("Property price (CHF)", 100_000, 10_000_000, 1_000_000, 50_000)
    own = st.number_input("Own funds (CHF)", 0, 10_000_000, 200_000, 10_000)
    income = st.number_input("Gross household income (CHF / year)", 0, 5_000_000, 180_000, 5_000)
    st.header("Rates offered to you (% / year)")
    st.caption(f"Defaults: SNB average published rates, {mort_month}")
    saron_total = st.number_input("SARON mortgage (today's total rate)", 0.0, 10.0, round(float(Mlast["saron_unlimited_pct"]), 2), 0.05)
    f2 = st.number_input("2y fixed", 0.0, 10.0, round(float(Mlast["fixed_2y_pct"]), 2), 0.05)
    f5 = st.number_input("5y fixed", 0.0, 10.0, round(float(Mlast["fixed_5y_pct"]), 2), 0.05)
    f10 = st.number_input("10y fixed", 0.0, 10.0, round(float(Mlast["fixed_10y_pct"]), 2), 0.05)
    st.header("Assumptions")
    use_est = st.checkbox("Use long-run SARON level estimated from data", True)
    theta = None if use_est else st.slider("Long-run SARON level (%)", -0.75, 3.0, 1.0, 0.25)
    risk = st.select_slider("Your attitude to risk", ["Cheapest", "Relaxed", "Balanced", "Careful", "Safest"], "Balanced")
LAM = {"Cheapest": 0.0, "Relaxed": 5e-6, "Balanced": 2e-5, "Careful": 1e-4, "Safest": 1e-2}[risk]

# ---- input checks ----
loan = price - own
errors = []
if own < 0.2 * price: errors.append(f"Own funds must be at least 20% of the price (CHF {0.2*price:,.0f}). Swiss banks will not lend more than 80%.")
if loan <= 0: errors.append("Own funds cover the full price: no mortgage needed.")
bank = m.bank_affordability(price, own, income) if income > 0 else None
if income <= 0: errors.append("Income must be positive.")
elif not bank["ok"] and own >= 0.2 * price:
    errors.append(f"Bank affordability rule fails: costs at a theoretical 5% rate = {100*bank['ratio']:.0f}% of income (max 33%).")
if errors:
    for e in errors: st.error(e)
    st.stop()

# ---- model ----
r0 = r[-1]
offered = [saron_total / 100 - max(r0, 0), f2 / 100, f5 / 100, f10 / 100]
P, info = simulate(theta)
rates = m.tranche_rates(P, offered)
budget = m.interest_budget(price, loan, income)
W, mu, sd, p95, ptr = m.frontier_table(rates, loan, budget, step=0.05)
ok = ptr <= 0.05
best = int(np.argmin(np.where(ok, mu + LAM * sd**2, np.inf)))
i5050 = int(np.argmin(np.abs(W - [.5, 0, 0, .5]).sum(1)))
eff = sorted([i for i in range(len(W)) if not np.any((mu < mu[i] - 1e-9) & (p95 <= p95[i] + 1e-9))], key=lambda i: p95[i])

st.subheader(f"Recommended mix: {lab(W[best])}")
c2, c3, c4 = st.columns(3)
c2.metric("Expected interest, 10 years", f"CHF {mu[best]:,.0f}")
c3.metric("Bad case (worst 5%)", f"CHF {p95[best]:,.0f}")
c4.metric("Chance a year becomes unaffordable", f"{100*ptr[best]:.1f}%")
st.caption(f"Mortgage CHF {loan:,.0f} (loan-to-value {100*loan/price:.0f}%). Yearly interest budget keeping housing costs ≤ 1/3 of income: CHF {budget:,.0f}. "
           f"Long-run SARON level used: {100*info['theta']:.2f}%.")

left, right = st.columns([1.3, 1])
with left:
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.scatter(p95[~ok] / 1e3, mu[~ok] / 1e3, s=5, c="#f4b4b4", label="too risky for your income")
    ax.scatter(p95[ok] / 1e3, mu[ok] / 1e3, s=5, c="#cfd6e0", label="other mixes")
    ax.plot(p95[eff] / 1e3, mu[eff] / 1e3, c="#1f5fbf", lw=2.5, label="efficient mixes")
    ax.scatter(p95[best] / 1e3, mu[best] / 1e3, s=150, c="#16a34a", edgecolor="w", zorder=5, label="your mix")
    ax.scatter(p95[i5050] / 1e3, mu[i5050] / 1e3, marker="X", s=90, c="#d97706", zorder=5, label="bank 50/50")
    ax.set_xlabel("Bad case: worst-5% interest over 10 years (kCHF)"); ax.set_ylabel("Expected interest over 10 years (kCHF)")
    ax.grid(alpha=.3); ax.legend(fontsize=8); st.pyplot(fig)
with right:
    rows = [("Your mix", best)] + [(n, int(np.argmin(np.abs(W - w).sum(1)))) for n, w in
            [("100% SARON", [1, 0, 0, 0]), ("Bank 50/50", [.5, 0, 0, .5]), ("100% 10y fixed", [0, 0, 0, 1])]]
    st.subheader("Compare")
    st.table({"Mix": [f"{n}: {lab(W[i])}" for n, i in rows],
              "Expected (CHF)": [f"{mu[i]:,.0f}" for _, i in rows],
              "Worst 5% (CHF)": [f"{p95[i]:,.0f}" for _, i in rows],
              "P(unaffordable year)": [f"{100*ptr[i]:.1f}%" for _, i in rows]})
    fig2, ax2 = plt.subplots(figsize=(6, 3.2))
    xs = np.arange(P.shape[1]) / 12
    lo, hi = np.percentile(P, [5, 95], axis=0)
    ax2.fill_between(xs, 100 * lo, 100 * hi, color="#1f5fbf", alpha=.15, lw=0, label="5-95% of paths")
    ax2.plot(xs, 100 * P.mean(axis=0), c="#1f5fbf", lw=2, label="expected SARON")
    for k, c in zip([3, 11, 27], ["#16a34a", "#dc2626", "#7c3aed"]): ax2.step(xs, 100 * P[k], where="post", c=c, lw=1)
    ax2.axhline(0, c="red", ls="--", lw=.8); ax2.set_xlabel("years from today"); ax2.set_ylabel("SARON %"); ax2.legend(fontsize=7); ax2.grid(alpha=.3)
    st.pyplot(fig2)

with st.expander("How this is calculated"):
    st.markdown(f"""
- **Rate model** (SNB data 1999-{saron_month[:4]}): SARON changes only at quarterly SNB meetings, in 0.25% steps.
  Probability of a move per quarter = {100*info['p_move']:.0f}%; size from an OLS regression of moves on the level (mean reversion); {P.shape[0]:,} simulated 10-year paths.
- **SARON tranche** pays max(SARON, 0) + margin. **Fixed tranches** renew at today's offer shifted by the change in SARON.
- **Optimisation**: minimise expected cost + λ·variance over all mixes on a 5% grid, with weights ≥ 0 and P(any year's interest > budget) ≤ 5%.
- Not modelled: switching strategies, amortisation, early-repayment penalties, taxes. Not financial advice.
""")
