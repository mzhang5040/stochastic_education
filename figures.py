"""
figures.py
==========
Generates the paper's single figure, the ECLS-K:2011 sticky-floor / sticky-
ceiling / per-state-slope figure (Sec 3.4), from ecls.py.

Output: fig5_tails.png   (referenced as \ref{fig:tails} in the paper)

All three panels use the same design-based uncertainty procedure: the survey-
weighted logistic curve in panels A/B is a base-weight fit, and its pointwise
band is the ECLS JK2 jackknife SE obtained from the 80 replicate weights with
the wave-specific thresholds recomputed inside each replicate (the same
procedure that produces the per-state intervals in panel C and every
significance statement in the paper).

Run:  python figures.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import expit
import ecls as E

BLUE_DARK, BLUE_LIGHT, RED_MID, GREY = '#1a5276', '#aed6f1', '#c0392b', '#888780'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.edgecolor': '#888780',
                     'axes.linewidth': 0.8})
states = [r'$s_1$', r'$s_2$', r'$s_3$', r'$s_4$', r'$s_5$']

df = E.load()

# Base-weight and 80 replicate transition sets (thresholds recomputed per weight),
# built once and shared by both tail panels.
T0 = E.transitions_for_weight(df, E.BASE)
rep_T = [E.transitions_for_weight(df, r) for r in E.REPS]

# tercile SES group means (base-weighted) for the red dashed markers
lo, hi = E.wquantile(T0["ses"].values, T0["w"].values, [1/3, 2/3])
ses_lo = T0.loc[T0["ses"] <= lo, "ses"].mean()
ses_hi = T0.loc[T0["ses"] >= hi, "ses"].mean()

# panel C: per-state JK2 slopes (reuse ecls)
slopes = {k: E.state_slope(df, E.BASE, k) for k in range(5)}
ses_se = {k: E.jk2_se(slopes[k], [E.state_slope(df, r, k) for r in E.REPS]) for k in range(5)}

XR = np.linspace(-2.5, 2.0, 160)


def _fit(T, k):
    d = T[T["fr"] == k]
    return E.wlogit((d["to"] == k).values.astype(float),
                    np.column_stack([np.ones(len(d)), d["ses"].values]),
                    d["w"].values), int(len(d))


def jk2_curve(k):
    """Base-weight logistic curve for staying in s_k, with a pointwise JK2 band."""
    b0, n = _fit(T0, k)
    p0 = expit(b0[0] + b0[1] * XR)
    acc = np.zeros_like(XR)
    for Tr in rep_T:
        br, _ = _fit(Tr, k)
        acc += (expit(br[0] + br[1] * XR) - p0) ** 2
    se = np.sqrt(acc)                      # JK2 pointwise standard error
    return b0, p0, np.clip(p0 - 1.96 * se, 0, 1), np.clip(p0 + 1.96 * se, 0, 1), n


bA, mA, loA, hiA, nA = jk2_curve(0)
bB, mB, loB, hiB, nB = jk2_curve(4)

fig = plt.figure(figsize=(12, 3.7))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=.34)


def panel(ax, m_, lo_, hi_, b, title, ylab, ylim):
    ax.fill_between(XR, lo_, hi_, color=BLUE_LIGHT, alpha=.55)
    ax.plot(XR, m_, color=BLUE_DARK, lw=2)
    for xs, lbl, ha in [(ses_lo, 'Low-SES', 'right'), (ses_hi, 'High-SES', 'left')]:
        yp = expit(b[0] + b[1] * xs)
        ax.axvline(xs, color=RED_MID, lw=1, ls='--', alpha=.7)
        ax.annotate(f'{lbl}\n{yp:.2f}', xy=(xs, yp),
                    xytext=(xs + (-.12 if ha == 'right' else .1), yp + (.04 if b[1] < 0 else -.07)),
                    fontsize=8.5, ha=ha, color=RED_MID)
    ax.set_xlabel('SES composite'); ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=9, pad=6); ax.set_ylim(*ylim); ax.set_xlim(-2.7, 2.2)
    ax.spines[['top', 'right']].set_visible(False)


zA, zB = slopes[0] / ses_se[0], slopes[4] / ses_se[4]
panel(fig.add_subplot(gs[0]), mA, loA, hiA, bA,
      r'(A) Sticky floor at $s_1$' + f'\n$\\hat\\beta={slopes[0]:+.2f}$, $z={zA:+.1f}$, $n={nA:,}$',
      r'$P(s_1\to s_1\mid\mathrm{SES})$', (.45, .92))
panel(fig.add_subplot(gs[1]), mB, loB, hiB, bB,
      r'(B) Sticky ceiling at $s_5$' + f'\n$\\hat\\beta={slopes[4]:+.2f}$, $z={zB:+.1f}$, $n={nB:,}$',
      r'$P(s_5\to s_5\mid\mathrm{SES})$', (.35, .92))
ax3 = fig.add_subplot(gs[2])
xs = list(range(1, 6)); ys = [slopes[k] for k in range(5)]; es = [1.96 * ses_se[k] for k in range(5)]
ax3.axhline(0, color='#444', lw=.9)
ax3.errorbar(xs, ys, yerr=es, fmt='o', ms=6, capsize=4, lw=1.5, ecolor='#555',
             mfc='white', mec=BLUE_DARK, zorder=3)
for xi, k in zip(xs, range(5)):
    ax3.plot(xi, ys[k], 'o', ms=6, color=BLUE_DARK if abs(slopes[k]/ses_se[k]) > 2 else GREY, zorder=4)
ax3.set_xticks(xs); ax3.set_xticklabels(states)
ax3.set_xlabel('achievement state'); ax3.set_ylabel(r'SES persistence slope $\hat\beta_k$')
ax3.set_title('(C) significant at the tails, flat in the middle', fontsize=9, pad=6)
ax3.set_xlim(.5, 5.5); ax3.spines[['top', 'right']].set_visible(False)
fig.savefig("fig5_tails.png", dpi=200, bbox_inches='tight', facecolor='white')
print(f"saved fig5_tails.png   nA={nA:,}  nB={nB:,}  "
      f"betaA={slopes[0]:+.3f}(z{zA:+.1f})  betaB={slopes[4]:+.3f}(z{zB:+.1f})")
