"""Decision maps of the 'when to fix' model -> ../fix_timing_map.png"""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import fix_timing as F
r, MR, S = F.load(); s0 = r[-1]; Mo = MR[-1]; sl = F.fixed_rate_slopes(MR, S); grid, Q, info = F.lattice(r)
i0 = int(np.argmin(np.abs(grid - s0))); margin = float(Mo["saron_unlimited_pct"]) / 100 - max(s0, 0)
offers = {k: float(Mo[f"fixed_{k}y_pct"]) / 100 for k in (2, 5, 10)}
cmap = ListedColormap(["#eef1f5", "#eb6834", "#1baf7a", "#eda100"])
show = grid <= 0.03
fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
for ax, (name, a) in zip(axs, [("Relaxed (a = 0, expected cost)", 0.0), ("Moderate (a = 1.5e-5)", 1.5e-5), ("Careful (a = 2.5e-5)", 2.5e-5)]):
    V, pol, f = F.solve(grid, Q, grid[i0], 800_000, margin, offers, sl, a=a)
    code = np.vectorize({0: 0, 2: 1, 5: 2, 10: 3}.get)(pol[::4].T)[show]       # rows = levels, cols = years
    ax.imshow(code, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=3, extent=[-0.5, 9.5, 100*grid[show][0]-0.125, 100*grid[show][-1]+0.125])
    ax.scatter([0], [100 * grid[i0]], s=120, c="#0b0b0b", zorder=5); ax.annotate("today", (0, 100*grid[i0]), xytext=(8, 6), textcoords="offset points", fontsize=9)
    cost, ff, fk = F.simulate_policy(grid, Q, pol, f, i0, 800_000, margin, n=4000)
    ax.set_title(f"{name}\nmean {cost.mean()/1e3:.0f}k, worst 5% {np.percentile(cost,95)/1e3:.0f}k, P(fix) {np.mean(ff>=0):.0%}", fontsize=10)
    ax.set_xlabel("year"); ax.set_xticks(range(10)); ax.grid(alpha=.25)
axs[0].set_ylabel("SARON level (%)")
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(["#eef1f5", "#eb6834", "#1baf7a", "#eda100"], ["stay on SARON", "fix 2y", "fix 5y", "fix 10y"])], loc="lower center", ncol=4, frameon=False)
fig.suptitle("When should you fix? Optimal action for each year and SARON level (CHF 800k, SNB rates Sep 2026, backward induction)", fontsize=11)
plt.tight_layout(rect=[0, 0.06, 1, 0.95]); plt.savefig("../fix_timing_map.png", dpi=150)
