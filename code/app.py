"""Streamlit app: optimal Swiss mortgage mix. Run from the repo root: streamlit run code/app.py"""
import csv, os, numpy as np, pandas as pd, streamlit as st, altair as alt
import model as m

st.set_page_config(page_title="Swiss Mortgage Mix", page_icon="🏠", layout="wide")
HERE = os.path.dirname(__file__)
NAMES = ["SARON", "2y fixed", "5y fixed", "10y fixed"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # fixed order, one colour per tranche
INK, MUTED = "#0b0b0b", "#6b6a66"
lab = lambda w: " + ".join(f"{int(round(100*x))}% {n}" for x, n in zip(w, NAMES) if x > 0.001)
chf = lambda x: f"CHF {x:,.0f}".replace(",", "'")

st.markdown("""<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {background: linear-gradient(135deg,#eef4fc 0%,#f7f9fc 100%); border:1px solid #dfe7f2; border-radius:16px; padding:22px 26px 10px;}
.hero .kicker {font-size:.8rem; letter-spacing:.08em; text-transform:uppercase; color:#2a78d6; font-weight:600;}
.hero .mix {font-size:2.1rem; font-weight:700; line-height:1.2; margin:.2rem 0 .3rem;}
.hero .say {color:#52514e; font-size:1rem;}
div[data-testid="stMetricValue"] {font-size:1.6rem;}
.small {color:#6b6a66; font-size:.85rem;}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load():
    S = [x for x in csv.DictReader(open(os.path.join(HERE, "../data/clean/saron_monthly.csv"))) if x["saron_pct"]]
    r = np.array([float(x["saron_pct"]) for x in S]) / 100
    M = {x["month"]: x for x in csv.DictReader(open(os.path.join(HERE, "../data/clean/mortgage_rates_monthly.csv")))}
    last = max(k for k in M if M[k]["fixed_10y_pct"])
    return r, [x["month"] for x in S], M[last], last

@st.cache_data
def simulate(theta_pct, n=3000):
    r, *_ = load()
    return m.policy_move_paths(r, r[-1], n=n, theta=None if theta_pct is None else theta_pct / 100)

r, s_months, Mlast, mort_month = load()

# ---------------- sidebar: inputs ----------------
with st.sidebar:
    st.markdown("### 🏠 Your purchase")
    price = st.number_input("Property price (CHF)", 100_000, 10_000_000, 1_000_000, 50_000, format="%d")
    own = st.number_input("Own funds (CHF)", 0, 10_000_000, 200_000, 10_000, format="%d")
    income = st.number_input("Gross household income (CHF / year)", 0, 5_000_000, 180_000, 5_000, format="%d")
    st.markdown("### ⚖️ Your attitude to risk")
    risk = st.select_slider("risk", ["Cheapest", "Relaxed", "Balanced", "Careful", "Safest"], "Balanced", label_visibility="collapsed")
    st.caption({"Cheapest": "Lowest expected cost, whatever the swings.", "Relaxed": "Mostly cost, a little safety.",
                "Balanced": "Trade cost against risk.", "Careful": "Pay more to sleep well.", "Safest": "No surprises at all."}[risk])
    with st.expander("📈 Rates offered to you (% / year)"):
        st.caption(f"Defaults: SNB average published rates, {mort_month}. Enter your bank's offers.")
        saron_total = st.number_input("SARON mortgage, today's total rate", 0.0, 10.0, round(float(Mlast["saron_unlimited_pct"]), 2), 0.05)
        f2 = st.number_input("2-year fixed", 0.0, 10.0, round(float(Mlast["fixed_2y_pct"]), 2), 0.05)
        f5 = st.number_input("5-year fixed", 0.0, 10.0, round(float(Mlast["fixed_5y_pct"]), 2), 0.05)
        f10 = st.number_input("10-year fixed", 0.0, 10.0, round(float(Mlast["fixed_10y_pct"]), 2), 0.05)
    with st.expander("🔮 Scenario: where SARON goes long-term"):
        use_est = st.toggle("Use the level estimated from SNB history", True)
        theta = None if use_est else st.slider("Long-run SARON level (%)", -0.75, 3.0, 1.0, 0.25)
LAM = {"Cheapest": 0.0, "Relaxed": 5e-6, "Balanced": 2e-5, "Careful": 1e-4, "Safest": 1e-2}[risk]

st.markdown("## How should I split my Swiss mortgage?")
st.markdown('<div class="small">SARON vs 2-, 5- and 10-year fixed · real SNB data · 3,000 simulated rate paths over 10 years</div>', unsafe_allow_html=True)
st.write("")

# ---------------- input checks ----------------
loan = price - own
errors = []
if own < 0.2 * price: errors.append(f"Own funds must be at least 20% of the price ({chf(0.2*price)}). Swiss banks do not lend more than 80%.")
if loan <= 0: errors.append("Your own funds cover the full price: no mortgage needed.")
bank = m.bank_affordability(price, own, income) if income > 0 else None
if income <= 0: errors.append("Income must be positive.")
elif not bank["ok"] and own >= 0.2 * price:
    errors.append(f"A bank would refuse this mortgage: at the 5% stress rate, housing costs are {100*bank['ratio']:.0f}% of income (max 33%). "
                  f"You need an income of at least {chf(bank['ratio']*income*3)} or more own funds.")
if errors:
    for e in errors: st.error(e, icon="🚫")
    st.stop()

# ---------------- model ----------------
r0 = r[-1]
offered = [saron_total / 100 - max(r0, 0), f2 / 100, f5 / 100, f10 / 100]
P, info = simulate(theta)
rates = m.tranche_rates(P, offered)
budget = m.interest_budget(price, loan, income)
W, mu, sd, p95, ptr = m.frontier_table(rates, loan, budget, step=0.05)
ok = ptr <= 0.05
best = int(np.argmin(np.where(ok, mu + LAM * sd**2, np.inf)))
find = lambda w: int(np.argmin(np.abs(W - np.array(w)).sum(1)))
i5050, iS, i10 = find([.5, 0, 0, .5]), find([1, 0, 0, 0]), find([0, 0, 0, 1])
Cpaths = loan * m.total_cost_per_chf(rates)                      # (paths, 4)
p5 = lambda i: np.percentile(Cpaths @ W[i], 5)
wb = W[best]

# ---------------- hero ----------------
parts = [f"<b>{chf(loan*x)}</b> in {n}" for x, n in zip(wb, NAMES) if x > 0.001]
segs = "".join(f'<div style="flex:{x};background:{c};color:white;font-weight:600;font-size:.85rem;padding:6px 10px;white-space:nowrap;overflow:hidden">{int(round(100*x))}% {n}</div>'
               for x, n, c in zip(wb, NAMES, COLORS) if x > 0.001)
st.markdown(f"""<div class="hero"><div class="kicker">Recommended for you · {risk}</div>
<div class="mix">{lab(wb)}</div>
<div class="say">Of your {chf(loan)} mortgage, put {' and '.join(parts)}.</div>
<div style="display:flex;gap:2px;border-radius:8px;overflow:hidden;margin:14px 0 12px">{segs}</div></div>""", unsafe_allow_html=True)
st.write("")
c1, c2, c3 = st.columns(3)
c1.metric("Expected interest, 10 years", chf(mu[best]), f"{mu[best]-mu[i5050]:+,.0f} CHF vs bank 50/50".replace(",", "'"), delta_color="inverse")
c2.metric("Bad case (worst 5% of futures)", chf(p95[best]), f"{p95[best]-p95[i5050]:+,.0f} CHF vs bank 50/50".replace(",", "'"), delta_color="inverse")
c3.metric("Chance a year exceeds your budget", f"{100*ptr[best]:.1f}%", help=f"Budget = income/3 − maintenance − amortisation = {chf(budget)} of interest per year")
st.caption(f"Mortgage {chf(loan)} · loan-to-value {100*loan/price:.0f}% · bank stress test {100*bank['ratio']:.0f}% of income ✓ · long-run SARON level {100*info['theta']:.2f}%")

# ---------------- tabs ----------------
t1, t2, t3, t4 = st.tabs(["📊 Compare strategies", "🎯 Cost vs risk", "📈 Where could SARON go?", "🧮 Formulas"])

with t1:
    rows = [("Your mix", best), ("All SARON", iS), ("Bank 50/50", i5050), ("All 10y fixed", i10)]
    df = pd.DataFrame({"Strategy": [n for n, _ in rows], "Mix": [lab(W[i]) for _, i in rows],
                       "Expected": [mu[i] for _, i in rows], "Good case (5%)": [p5(i) for _, i in rows],
                       "Bad case (95%)": [p95[i] for _, i in rows], "P(over budget)": [ptr[i] for _, i in rows]})
    df["is_you"] = df.Strategy == "Your mix"
    base = alt.Chart(df).encode(y=alt.Y("Strategy:N", sort=None, title=None, axis=alt.Axis(labelFontSize=13, labelColor=INK)))
    rng = base.mark_rule(strokeWidth=2, color="#9aa4b2").encode(x=alt.X("Good case (5%):Q", title="10-year interest (CHF)", scale=alt.Scale(zero=False), axis=alt.Axis(format=",.0f", gridOpacity=.3)), x2="Bad case (95%):Q")
    ends = base.mark_tick(thickness=2, size=14, color="#9aa4b2").encode(x="Good case (5%):Q") + base.mark_tick(thickness=2, size=14, color="#9aa4b2").encode(x="Bad case (95%):Q")
    dot = base.mark_point(filled=True, size=220, stroke="#fcfcfb", strokeWidth=2).encode(
        x="Expected:Q", color=alt.condition("datum.is_you", alt.value("#2a78d6"), alt.value("#52514e")),
        tooltip=["Strategy", "Mix", alt.Tooltip("Expected:Q", format=",.0f"), alt.Tooltip("Good case (5%):Q", format=",.0f"),
                 alt.Tooltip("Bad case (95%):Q", format=",.0f"), alt.Tooltip("P(over budget):Q", format=".1%")])
    txt = base.mark_text(dy=-16, fontSize=11, color=MUTED).encode(x="Expected:Q", text=alt.Text("Expected:Q", format=",.0f"))
    st.markdown("**Dot = expected cost. Line = range from the best 5% to the worst 5% of futures.**")
    st.altair_chart((rng + ends + dot + txt).properties(height=260), use_container_width=True)
    show = df.drop(columns="is_you").copy()
    for c in ["Expected", "Good case (5%)", "Bad case (95%)"]: show[c] = show[c].map(chf)
    show["P(over budget)"] = df["P(over budget)"].map(lambda x: f"{100*x:.1f}%")
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.caption("Over 10 years the 10y fixed tranche never renews, so its cost is certain: it plays the role of the risk-free asset.")

with t2:
    eff = sorted([i for i in range(len(W)) if ok[i] and not np.any(ok & (mu < mu[i] - 1e-9) & (p95 <= p95[i] + 1e-9))], key=lambda i: p95[i])
    cloud = pd.DataFrame({"Bad case": p95, "Expected": mu, "Mix": [lab(w) for w in W], "Affordable": np.where(ok, "other mixes", "too risky for your income")})
    front = cloud.iloc[eff]
    marks = pd.DataFrame({"Bad case": [p95[i] for _, i in rows], "Expected": [mu[i] for _, i in rows], "Strategy": [n for n, _ in rows], "Mix": [lab(W[i]) for _, i in rows]})
    sc = alt.Scale(zero=False)
    ch = (alt.Chart(cloud).mark_circle(size=18, opacity=.35).encode(x=alt.X("Bad case:Q", scale=sc, title="Bad case: worst-5% interest over 10 years (CHF)", axis=alt.Axis(format=",.0f", gridOpacity=.3)),
            y=alt.Y("Expected:Q", scale=sc, title="Expected interest over 10 years (CHF)", axis=alt.Axis(format=",.0f", gridOpacity=.3)),
            color=alt.Color("Affordable:N", scale=alt.Scale(domain=["other mixes", "too risky for your income"], range=["#b8c2d0", "#e34948"]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["Mix", alt.Tooltip("Expected:Q", format=",.0f"), alt.Tooltip("Bad case:Q", format=",.0f")])
          + alt.Chart(front).mark_line(strokeWidth=3, color="#2a78d6").encode(x="Bad case:Q", y="Expected:Q")
          + alt.Chart(marks).mark_point(filled=True, size=200, stroke="#fcfcfb", strokeWidth=2).encode(x="Bad case:Q", y="Expected:Q",
              color=alt.condition("datum.Strategy == 'Your mix'", alt.value("#2a78d6"), alt.value("#0b0b0b")), tooltip=["Strategy", "Mix", alt.Tooltip("Expected:Q", format=",.0f"), alt.Tooltip("Bad case:Q", format=",.0f")])
          + alt.Chart(marks).mark_text(align="left", dx=10, dy=-8, fontSize=12, fontWeight="bold", color=INK).encode(x="Bad case:Q", y="Expected:Q", text="Strategy:N"))
    st.markdown("**Every grey dot is one possible mix (5% steps). The blue line is the efficient frontier: nothing is both cheaper and safer.**")
    st.altair_chart(ch.properties(height=430), use_container_width=True)
    has25 = any(W[i][1] + W[i][2] > 0 for i in eff)
    st.caption(("2y and 5y fixed appear on the frontier with these rates." if has25 else
                "2y and 5y fixed are never efficient here: a mix of SARON and 10y fixed always beats them.") + " Move the risk slider to slide along the blue line.")

with t3:
    xs = np.arange(P.shape[1])
    lo, hi = np.percentile(P, [5, 95], axis=0)
    months = pd.period_range(s_months[-1], periods=P.shape[1], freq="M").to_timestamp()
    hist_n = 12 * 8
    hist = pd.DataFrame({"date": pd.PeriodIndex(s_months[-hist_n:], freq="M").to_timestamp(), "SARON": 100 * r[-hist_n:]})
    band = pd.DataFrame({"date": months, "lo": 100 * lo, "hi": 100 * hi, "mean": 100 * P.mean(0)})
    paths = pd.concat([pd.DataFrame({"date": months, "SARON": 100 * P[k], "path": f"example {j+1}"}) for j, k in enumerate([3, 11, 27])])
    g = alt.Axis(gridOpacity=.3)
    ch = (alt.Chart(band).mark_area(color="#2a78d6", opacity=.15).encode(x=alt.X("date:T", title=None, axis=g), y=alt.Y("lo:Q", title="SARON (%)", axis=g), y2="hi:Q")
          + alt.Chart(hist).mark_line(color=INK, strokeWidth=2, interpolate="step-after").encode(x="date:T", y="SARON:Q", tooltip=[alt.Tooltip("date:T", format="%b %Y"), alt.Tooltip("SARON:Q", format=".2f")])
          + alt.Chart(paths).mark_line(strokeWidth=1.2, opacity=.7, interpolate="step-after").encode(x="date:T", y="SARON:Q", detail="path:N", color=alt.value("#9085e9"))
          + alt.Chart(band).mark_line(color="#2a78d6", strokeWidth=2.5, strokeDash=[6, 3]).encode(x="date:T", y="mean:Q", tooltip=[alt.Tooltip("date:T", format="%b %Y"), alt.Tooltip("mean:Q", title="expected", format=".2f"), alt.Tooltip("lo:Q", title="5%", format=".2f"), alt.Tooltip("hi:Q", title="95%", format=".2f")])
          + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#e34948", strokeDash=[3, 3]).encode(y="y:Q"))
    st.markdown("**Black = actual SARON. Blue dashed = expected path. Shaded = 90% of simulated futures. Purple = three example futures.**")
    st.altair_chart(ch.properties(height=380), use_container_width=True)
    st.caption(f"The SNB moves in 0.25% steps at quarterly meetings. Estimated from 1999-{s_months[-1][:4]}: a move happens in {100*info['p_move']:.0f}% of quarters, "
               f"and rates drift towards {100*info['theta']:.2f}%. The SARON mortgage never goes below its margin (floor at 0).")

with t4:
    cf, muS, sdS = Cpaths[:, 3].mean(), Cpaths[:, 0].mean(), Cpaths[:, 0].std(ddof=1)
    wstar = (cf - muS) / (2 * LAM * sdS**2) if LAM > 0 else float("inf")
    st.markdown("**1. Rate model: SNB move or hold** (OLS on quarterly SARON)")
    st.code(f"P(move in a quarter) = {info['p_move']:.3f}\nmove = c + d x level + e     c = {info['c']:.4f}, d = {info['d']:.3f}, sd(e) = {100*info['sigma_move']:.2f}%\n"
            f"long-run level theta = -c/d = {100*info['theta']:.2f}%   (moves rounded to 0.25%, floor -0.75%)", language=None)
    st.markdown("**2. Cost of each tranche on each simulated path**")
    st.code(f"SARON:  rate = max(SARON, 0) + margin ({100*offered[0]:.2f}%)\nFixed:  rate = offer + (SARON at renewal - SARON today)\n"
            f"Cost   = loan x sum of monthly rates / 12   ->  10y fixed = {chf(loan)} x {f10:.2f}% x 10 = {chf(cf)} (certain)", language=None)
    st.markdown("**3. Optimisation**")
    st.code(f"minimise   expected cost + lambda x variance      lambda = {LAM:g} ({risk})\nsubject to weights >= 0, sum = 1, P(a year's interest > {chf(budget)}) <= 5%", language=None)
    st.markdown("**4. Check by hand** (SARON + 10y fixed only)")
    st.code(f"w* = (c_f - mu_S) / (2 lambda sd_S^2)\n   = ({cf:,.0f} - {muS:,.0f}) / (2 x {LAM:g} x {sdS:,.0f}^2)\n   = {min(max(wstar,0),1):.2f}  ->  {100*min(max(wstar,0),1):.0f}% SARON"
            + ("" if LAM > 0 else "  (lambda = 0: pick the cheapest acceptable mix)"), language=None)
    st.caption("Not modelled: switching strategies, amortisation, early-repayment penalties, taxes. Simulations of possible futures, not forecasts. Not financial advice.")
