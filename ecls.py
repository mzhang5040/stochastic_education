"""
ecls.py
=======
ECLS-K:2011 empirical core (importable) and the paper's headline empirical
results, using the method the paper describes throughout: within-wave score
thresholds, complete-panel survey weighting, and JK2 jackknife standard errors with
the thresholds re-estimated inside each replicate.

As a library it exposes:
    load, wave_thresholds, build_transitions, transitions_for_weight,
    state_slope, wlogit_slope, jk2_se, and the constants
    WAVES, PAIRS, BASE, REPS, PANEL.
Run directly to print:
    Table 1  SES-stratified transition matrices (with positive-weight n's)
    Table 2  per-state SES persistence slopes with JK2 jackknife SEs

Robustness (cross-state difference tests, base-year weight, jump-size,
threshold decomposition) lives in ecls_robustness.py.

Input: ecls_k5_puf.csv  (ECLS-K:2011 K-5 public-use extract, Appendix A vars)
"""
import sys
import numpy as np
import pandas as pd
from scipy.special import expit

CSV = "ecls_k5_puf.csv"
WAVES = ["x2mscalk5", "x4mscalk5", "x6mscalk5", "x7mscalk5", "x8mscalk5", "x9mscalk5"]
PAIRS = [(i, i + 1) for i in range(len(WAVES) - 1)]   # K->1,1->2,2->3,3->4,4->5
BASE  = "w1c0"
REPS  = [f"w1c{r}" for r in range(1, 81)]             # 80 base-year JK2 replicates
PANEL = "w9c29p_9a0"                                   # complete-panel weight
PANEL_REPS = [f"w9c29p_9a{r}" for r in range(1, 81)]
# Primary inferential weight: complete-panel (K-5). Base-year is the broader-
# cohort sensitivity weight (see ecls_robustness.py).
PRIMARY, PRIMARY_REPS = PANEL, PANEL_REPS
SENSITIVITY, SENSITIVITY_REPS = BASE, REPS
PCTS = [0.25, 0.40, 0.60, 0.80]


def load(path=CSV):
    return pd.read_csv(path)


def wquantile(values, weights, qs):
    o = np.argsort(values)
    v, w = values[o], weights[o]
    cum = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(qs, cum, v)


def wave_thresholds(df, weightcol):
    thr = {}
    for wv in WAVES:
        msk = df[wv].notna() & (df[weightcol] > 0)
        thr[wv] = wquantile(df.loc[msk, wv].values, df.loc[msk, weightcol].values, PCTS)
    return thr


def _state(score, cuts):
    return np.digitize(score, [cuts[0], cuts[1], cuts[2], cuts[3]])   # 0..4


def build_transitions(df, thr, weightcol, require_pos=True):
    """
    One row per consecutive-wave transition (valid scores + SES). By default
    restricts to positive `weightcol` (the inferential sample); pass
    require_pos=False for the unweighted threshold-count samples of Sec 3.3.3.
    Column `iv` is the grade-interval index (for wave-pair fixed effects).
    """
    parts = []
    for k, (a, b) in enumerate(PAIRS):
        c1, c2 = WAVES[a], WAVES[b]
        msk = df[c1].notna() & df[c2].notna() & df["x12sesl"].notna()
        if require_pos:
            msk = msk & (df[weightcol] > 0)
        sub = df.loc[msk]
        parts.append(pd.DataFrame({
            "ses": sub["x12sesl"].values,
            "fr":  _state(sub[c1].values, thr[c1]),
            "to":  _state(sub[c2].values, thr[c2]),
            "w":   sub[weightcol].values,
            "iv":  k,
        }))
    return pd.concat(parts, ignore_index=True)


def transitions_for_weight(df, weightcol):
    """Convenience: thresholds AND transitions consistent with one weight."""
    return build_transitions(df, wave_thresholds(df, weightcol), weightcol)


def wlogit(y, X, w):
    """Survey-weighted (pseudo-MLE) logistic; X is the full design matrix."""
    b = np.zeros(X.shape[1])
    for _ in range(100):
        p = expit(X @ b)
        step = np.linalg.solve(X.T @ (X * (w * p * (1 - p))[:, None]) + 1e-9 * np.eye(X.shape[1]),
                               X.T @ (w * (y - p)))
        b += step
        if np.max(np.abs(step)) < 1e-11:
            break
    return b


def wlogit_slope(y, x, w):
    """Survey-weighted logistic slope of y on a single covariate x."""
    return wlogit(y, np.column_stack([np.ones_like(x), x]), w)[1]


def jk2_se(point, replicate_estimates):
    """ECLS JK2 SE: sqrt(sum_r (theta_r - theta_full)^2)."""
    return np.sqrt(np.nansum((np.asarray(replicate_estimates) - point) ** 2))


def state_slope(df, weightcol, k):
    """Slope of P(stay in s_k | SES), using `weightcol`'s own thresholds."""
    T = transitions_for_weight(df, weightcol)
    d = T[T["fr"] == k]
    return wlogit_slope((d["to"] == k).values.astype(float), d["ses"].values, d["w"].values)


# ---------------------------------------------------------------------------
def table1(df):
    ms = df["x12sesl"].notna() & (df[PRIMARY] > 0)
    t_lo, t_hi = wquantile(df.loc[ms, "x12sesl"].values, df.loc[ms, PRIMARY].values, [1/3, 2/3])
    T = transitions_for_weight(df, PRIMARY)
    T["grp"] = np.where(T["ses"] <= t_lo, "low", np.where(T["ses"] >= t_hi, "high", "mid"))
    print(f"Transitions: {len(T):,}  (low {int((T.grp=='low').sum()):,} / "
          f"high {int((T.grp=='high').sum()):,})\n")
    print("TABLE 1 - SES-stratified transition matrices (survey-weighted)")
    for g in ["low", "high"]:
        d = T[T["grp"] == g]
        print(f"\n  {g.upper()}-SES (N = {len(d):,})")
        print("       ->s1    ->s2    ->s3    ->s4    ->s5       n")
        for i in range(5):
            di = d[d["fr"] == i]
            row = np.array([di.loc[di["to"] == j, "w"].sum() for j in range(5)])
            row = row / row.sum() if row.sum() else row
            print("    s%d  %s   %5d" % (i+1, "  ".join(f"{v:.3f}" for v in row), len(di)))


def table2(df):
    print("\nTABLE 2 - per-state SES persistence slopes (weighted logit, JK2 SE)")
    print("  state    beta1     JK SE      z")
    rep_T = [transitions_for_weight(df, r) for r in PRIMARY_REPS]   # per-replicate thresholds
    T0 = transitions_for_weight(df, PRIMARY)
    for k in range(5):
        d = T0[T0["fr"] == k]
        b0 = wlogit_slope((d["to"] == k).values.astype(float), d["ses"].values, d["w"].values)
        breps = []
        for Tr in rep_T:
            dr = Tr[Tr["fr"] == k]
            breps.append(wlogit_slope((dr["to"] == k).values.astype(float),
                                      dr["ses"].values, dr["w"].values))
        se = jk2_se(b0, breps)
        print(f"   s{k+1}   {b0:+.3f}    {se:.3f}   {b0/se:+6.1f}")


if __name__ == "__main__":
    df = load(sys.argv[1] if len(sys.argv) > 1 else CSV)
    table1(df)
    table2(df)
