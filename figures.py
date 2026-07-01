"""
figures.py
==========
Generates the paper's single figure, the ECLS-K:2011 sticky-floor / sticky-
ceiling / per-state-slope figure (Sec 3.4), from ecls.py.

Output: fig5_tails.png   (referenced as \ref{fig:tails} in the paper)

Panels A/B use survey-weighted logistic curves with clustered bootstrap bands
(children resampled) for VISUALIZATION ONLY; panel C uses the official JK2
jackknife standard errors from ecls.py, which underlie every significance
statement in the paper.

Run:  python figures.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import expit
import ecls as E

np.random.seed(42)
BLUE_DARK, BLUE_LIGHT, RED_MID, GREY = '#1a5276', '#aed6f1', '#c0392b', '#888780'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.edgecolor': '#888780',
                     'axes.linewidth': 0.8})
states = [r'$s_1$', r'$s_2$', r'$s_3$', r'$s_4$', r'$s_5$']

df = E.load()
thr = E.wave_thresholds(df, E.BASE)
sc = {w: df[w].values for w in E.WAVES}
ses = df['x12sesl'].values
pos = np.isfinite(ses) & (df[E.BASE].values > 0)

# transitions with childid (for clustered viz-only bands)
F, T, S, W, CID = [], [], [], [], []
for a, b in E.PAIRS:
    m = np.isfinite(sc[E.WAVES[a]]) & np.isfinite(sc[E.WAVES[b]]) & pos
    F.append(np.digitize(sc[E.WAVES[a]][m], thr[E.WAVES[a]][:4]))
    T.append(np.digitize(sc[E.WAVES[b]][m], thr[E.WAVES[b]][:4]))
    S.append(ses[m]); W.append(df[E.BASE].values[m]); CID.append(df['childid'].values[m])
F, T, S, W, CID = map(np.concatenate, (F, T, S, W, CID))
lo, hi = E.wquantile(ses[pos], df[E.BASE].values[pos], [1/3, 2/3])
ses_lo, ses_hi = S[S <= lo].mean(), S[S >= hi].mean()

# panel C: per-state JK2 slopes (reuse ecls)
slopes = {k: E.state_slope(df, E.BASE, k) for k in range(5)}
ses_se = {k: E.jk2_se(slopes[k], [E.state_slope(df, r, k) for r in E.REPS]) for k in range(5)}


def curve_and_band(state_idx):
    a = F == state_idx
    y = (T[a] == state_idx).astype(float); x = S[a]; w = W[a]; cid = CID[a]
    b = E.wlogit(y, np.column_stack([np.ones_like(x), x]), w)
    xr = np.linspace(-2.5, 2.0, 160)
    uniq = np.unique(cid); idx = {c: np.where(cid == c)[0] for c in uniq}
    boot = np.empty((200, len(xr)))
    for i in range(200):
        pick = np.concatenate([idx[c] for c in np.random.choice(uniq, uniq.size, replace=True)])
        bb = E.wlogit(y[pick], np.column_stack([np.ones(pick.size), x[pick]]), w[pick])
        boot[i] = expit(bb[0] + bb[1] * xr)
    return b, xr, expit(b[0] + b[1] * xr), np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0), int(a.sum())


bA, xr, mA, loA, hiA, nA = curve_and_band(0)
bB, _, mB, loB, hiB, nB = curve_and_band(4)

fig = plt.figure(figsize=(12, 3.7))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=.34)

def panel(ax, m_, lo_, hi_, b, title, ylab, ylim):
    ax.fill_between(xr, lo_, hi_, color=BLUE_LIGHT, alpha=.55)
    ax.plot(xr, m_, color=BLUE_DARK, lw=2)
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
ax3.set_title('(C) significant at the tails, null in the middle', fontsize=9, pad=6)
ax3.set_xlim(.5, 5.5); ax3.spines[['top', 'right']].set_visible(False)
fig.savefig("fig5_tails.png", dpi=200, bbox_inches='tight', facecolor='white')
print("saved fig5_tails.png")
