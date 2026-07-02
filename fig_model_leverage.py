"""
Figure: (A) schematic of the calibrated five-state achievement chain and
(B) the calibrated leverage values L_i.  Produces fig_model_leverage.png.
Run:  python fig_model_leverage.py   (imports model.py; no external data)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Arc

import model as m

L, best, adj = m.leverage(m.P_base)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2),
                               gridspec_kw={"width_ratios": [1.35, 1.0]})

# ---------------------------------------------------------------- Panel A
axA.set_xlim(-0.7, 4.7)
axA.set_ylim(-1.35, 1.5)
axA.axis("off")
xs = np.arange(5)
y = 0.0
r = 0.24
labels = ["$s_1$", "$s_2$", "$s_3$", "$s_4$", "$s_5$"]
sub = ["Below\nBasic", "Basic$-$", "Basic$+$", "Prof.", "Adv."]
node_fc = ["#e8703a", "#f0b060", "#f0d060", "#a9c98f", "#5b8c5a"]

for i, x in enumerate(xs):
    axA.add_patch(Circle((x, y), r, fc=node_fc[i], ec="#333", lw=1.4, zorder=3))
    axA.text(x, y, labels[i], ha="center", va="center", fontsize=12, zorder=4)
    axA.text(x, y - 0.55, sub[i], ha="center", va="top", fontsize=10.0, color="#222")

# self-loops (persistence) above each node
for i, x in enumerate(xs):
    axA.add_patch(Arc((x, y + r + 0.16), 0.34, 0.34, theta1=205, theta2=-25,
                      lw=1.6, color="#333", zorder=2))
    axA.annotate("", xy=(x + 0.10, y + r + 0.05), xytext=(x + 0.16, y + r + 0.18),
                 arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.4))
axA.text(2.0, y + r + 0.62, "self-loops: persistence $p_{ii}$",
         ha="center", fontsize=10.5, color="#222")

# adjacent transitions: solid (up above baseline, down below)
for i in range(4):
    axA.add_patch(FancyArrowPatch((i + r, y + 0.05), (i + 1 - r, y + 0.05),
                  connectionstyle="arc3,rad=-0.35", arrowstyle="-|>",
                  mutation_scale=13, lw=1.6, color="#1f4e79", zorder=2))
    axA.add_patch(FancyArrowPatch((i + 1 - r, y - 0.05), (i + r, y - 0.05),
                  connectionstyle="arc3,rad=-0.35", arrowstyle="-|>",
                  mutation_scale=12, lw=1.3, color="#8a8a8a", zorder=2))
axA.text(0.5, 0.52, "adjacent up", color="#1f4e79", fontsize=10.5, ha="center")
axA.text(3.5, -0.52, "adjacent down", color="#5a5a5a", fontsize=10.5, ha="center")

# a few representative non-adjacent arcs, faint dashed
for (a, b) in [(0, 2), (0, 4), (1, 3)]:
    axA.add_patch(FancyArrowPatch((a + r, y + 0.12), (b - r, y + 0.12),
                  connectionstyle=f"arc3,rad=-{0.28 + 0.10*(b-a)}",
                  arrowstyle="-|>", mutation_scale=9, lw=1.0, ls="--",
                  color="#b9b9b9", zorder=1))
axA.text(2.0, 1.28, "non-adjacent moves: dashed, weight $\\propto\\rho^{|j-i|-1}$ "
         "($\\rho=0.7$)", ha="center", fontsize=10.5, color="#555")
axA.set_title("(A) Calibrated five-state achievement chain", fontsize=11.5, pad=4)

# ---------------------------------------------------------------- Panel B
colors = ["#e8703a" if i == 0 else "#9bb8d3" for i in range(5)]
bars = axB.bar(range(5), L, color=colors, ec="#333", lw=0.8, width=0.66)
axB.axhline(0, color="#333", lw=0.8)
axB.set_xticks(range(5))
axB.set_xticklabels(labels, fontsize=11)
axB.set_ylabel("top-state leverage  $L_i$", fontsize=10.5)
axB.set_title("(B) Leverage on the top-state mass", fontsize=10, pad=4)
for i, v in enumerate(L):
    axB.text(i, v + (0.06 if v >= 0 else -0.06), f"{v:+.2f}",
             ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5)
axB.set_ylim(min(L) - 0.28, max(L) + 0.28)
axB.annotate("$s_1$ leads by $3.3\\times$", xy=(0, L[0]), xytext=(1.6, L[0] - 0.05),
             fontsize=9, color="#e8703a",
             arrowprops=dict(arrowstyle="-|>", color="#e8703a", lw=1.2))
for sp in ("top", "right"):
    axB.spines[sp].set_visible(False)

plt.tight_layout()
plt.savefig("fig_model_leverage.png", dpi=300, bbox_inches="tight")
print("saved fig_model_leverage.png  L =", np.round(L, 3))
