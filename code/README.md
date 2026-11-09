# Mortgage mix: code (prototype)

- `model.py`: data loader (SNB CSV), AR(1) OLS estimation, Monte Carlo + bootstrap paths,
  tranche costs (SARON floor at 0, fixed renewals with parallel shift), risk measures,
  closed-form mix (10y fixed = riskless debt over 10y horizon) and constrained grid search,
  Swiss bank affordability check.
- `demo_synthetic.py`: runs everything on SYNTHETIC rates (replace with SNB data).

Run: `pip install numpy` then `python demo_synthetic.py`.

Findings on synthetic data (2026-11-09):
- Closed-form (Lagrange, shorting allowed) gives negative weights (e.g. -124% 2y fixed),
  meaningless for a mortgage -> motivates the w >= 0 constraint (Kuhn Tucker); grid search solves it.
- Higher risk aversion lambda moves the optimum from 100% SARON toward 10y fixed.
- A tight yearly budget constraint forces a mix (e.g. 55% SARON / 45% 10y).

## First run on real SNB data (2026-11-09, run_real.py)
Data: data/raw (untouched downloads, SNB publishing date 2026-11-02), cleaned by prepare_data.py into data/clean.
- AR(1) on SARON 1999-06..2026-10: a=0.00003, b=-0.0267, sigma=0.22%/month, long-run mean 0.11%, half-life 26 months.
  Post-2008 sample: half-life 53 months, long-run mean -0.16% -> results depend on the estimation window (limitation to show).
- SARON-mortgage (unlimited) minus max(SARON,0): mean 1.06% (2021-2026); latest margin 1.20%.
- Inputs 2026-09 (SNB mean published rates): SARON -0.057%, margin 1.20%, 2y 1.765%, 5y 2.026%, 10y 2.203%.
- CHF 800k, 10y: 100% SARON mean 126k (p95 178k); 100% 10y 176k (riskless); 50/50 mean 151k (p95 177k).
- Optimal (w>=0): lambda 1e-5 -> 100% SARON; 5e-5 -> 75/25 SARON/10y; 2e-4 -> 20/80. 2y and 5y never chosen.
Issues to fix:
- Plain bootstrap of monthly changes = random walk, no mean reversion -> paths drift to -5%. Replace with bootstrap of AR(1) residuals.
- Gaussian AR(1) is smooth while real SARON moves in policy steps (jumps): limitation, or extension with jumps.
- SNB rates are average published (list) rates, above best online offers (Swissquote 10y 1.66%): user must be able to enter own offers.

## Model v2 (2026-11-09, compare_models.py)
- Bootstrap fixed: resample AR(1) residuals (residual_bootstrap_paths). 10y SARON p5/median/p95 = -1.51/0.14/1.59 vs normal AR(1) -1.46/0.13/1.69 -> the two methods agree (validation).
- New "move or hold" SNB model (policy_move_paths), estimated on end-of-quarter SARON 1999-2026:
  P(move in a quarter) = 35% (20% post-2008); on move quarters OLS change = 0.0019 - 0.163*level -> long-run level 1.15%;
  moves rounded to 0.25% steps. Expected SARON path now rises from -0.06% to ~1.0% in 10y.
- Scenario theta = 0% ("rates stay low") for comparison. Long-run level = key assumption -> app slider; later market-implied (Confederation bond yields).
- To fix: no lower bound (paths reach -2.5%, SNB minimum was -0.75%); simulated moves somewhat larger than history.

## Decision example (2026-11-09, decision_example.py -> ../decision_example.png)
Buyer: price 1M, own 200k, income 180k (bank ratio 32.7% OK), interest budget 41,111/yr. Policy-move model, SNB Sep 2026 rates.
100% SARON 162,316 (p95 236,765, P trouble 0.6%); 50/50 169,294 (p95 206,519); 20/80 173,481 (p95 188,371); 100% 10y 176,272.
2y/5y never efficient (21 efficient mixes, all SARON+10y). Bank 50/50 is ON the frontier (efficient).
Hand check in ../method_sheet.md: w* = (c_f - mu_S)/(2 lambda sd_S^2) = 0.21 at lambda 2e-5, grid gives 20%.

## Sensitivity to long-run SARON level theta (sensitivity_theta.py), lambda 2e-5, same buyer
theta 0%: 100% SARON (mean 126.9k vs 10y 176.3k) | 0.5%: 80/20 | 1.15% (estimated): 20/80 | 2%: 100% 10y.
Break-even (SARON mean = 10y cost) at theta ~1.5% (linear interpolation 1.15%->2%).

## App (app.py)
Run locally: `pip install -r requirements.txt` then `streamlit run app.py` (from the code folder; data read from ../data/clean).
Inputs: price, own funds, income, offered rates (defaults = latest SNB averages), long-run level (estimated or slider), risk attitude.
Outputs: recommended mix, expected cost, worst-5% cost, P(unaffordable year), frontier chart, comparison table (100% SARON, bank 50/50, 100% 10y), SARON paths.
Input checks: own funds < 20%, no loan needed, income <= 0, bank affordability rule (5% theoretical rate) > 1/3.
Deploy: push project folder to GitHub, then share.streamlit.io -> New app -> code/app.py.

## Validation (2026-11-09, validation.py, full output in validation_output.txt)
- V1 hand formula w* = (c_f - mu_S)/(2 lambda sd_S^2) vs 0.1% grid: identical at lambda 1e-5 / 2e-5 / 5e-5 (0.447 / 0.223 / 0.089).
- V2 Lagrange closed form with shorting: [6.09, -5.12, -1.00, 1.03] -> meaningless, motivates w >= 0 (Kuhn Tucker).
- V3 Monte Carlo: SE of expected cost = 81 CHF; SE of P(trouble) ~0.08%. Across 5 seeds the optimum is 20% or 25% SARON (objective is flat near the optimum, ~900 CHF difference).
- V4 Model risk: monthly AR(1) (normal or bootstrap) has long-run level 0.11% -> 100% SARON; policy model (1.15%) -> 20/80. The long-run level drives the answer (same as sensitivity_theta.py).
- V5 UBS CIO benchmark (1M, 10y, 10y fixed 1.71%, SARON margin 0.75%): 10y 171k = UBS 171k; SARON p5/median/p95 = 87/150/249k vs UBS low/base/high 98/167/257k. Our range is consistent with UBS and adds probabilities.
- V6 Backtest 2007-12..2016-09 starts (106): SARON+1.06% cheaper than 10y fixed in 100% of cases, avg saving 125k on 800k. Falling-rate era: history alone would always say SARON -> why we need forward scenarios.
- V7 Invalid inputs all stop the app with a clear message. Extremes: theta 3% -> 100% 10y; theta -0.75% -> 100% SARON.
Issues found:
- The P(trouble) <= 5% constraint almost never binds for buyers who pass the bank rule (5% stress rate is stricter). It binds only below ~150k income, which the bank rule already rejects. Fix idea: let the user set their own comfort budget.
- "Start at 1.75%" test combined today's offers with a different SARON start -> inconsistent (gave 80% 2y). Scenarios must change start rate and offers together.
