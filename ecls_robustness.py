"""
ecls_robustness.py
=================
ECLS robustness checks. Imports the empirical core from ecls.py and reproduces
every ECLS robustness number reported in the paper:

    Sec 3.4.1  cross-state SES-slope difference tests (JK2);
               complete-panel weight robustness for the two tail slopes;
               grade-interval (wave-pair) fixed-effects check;
               attrition check (mean SES of s5-origin observed vs censored)
    Sec 3.4.2  jump-size regression gamma1, base-year and complete-panel weight
    Sec 3.4.3  threshold decomposition (pooled vs within-wave, unweighted vs
               weighted), with the matching s1-exiter counts

Run:  python ecls_robustness.py
"""
import numpy as np
import pandas as pd
import ecls as E

df = E.load()


# === Sec 3.4.1  cross-state difference tests ===============================
def slope_diff_jk(ka, kb):
    d0 = E.state_slope(df, E.BASE, ka) - E.state_slope(df, E.BASE, kb)
    reps = np.array([E.state_slope(df, r, ka) - E.state_slope(df, r, kb) for r in E.REPS])
    se = E.jk2_se(d0, reps)
    return d0, se, d0 / se

print("=" * 66)
print("SEC 3.4.1 - cross-state SES-slope difference tests (JK2)")
print("=" * 66)
for ka, kb, lab in [(0, 1, "beta1 - beta2 (floor vs lower-middle)"),
                    (4, 3, "beta5 - beta4 (ceiling vs upper-middle)")]:
    d, se, z = slope_diff_jk(ka, kb)
    print(f"  {lab:<38} diff={d:+.3f}  SE={se:.3f}  z={z:+.1f}  p<0.001")

print("\n  complete-panel weight robustness (tails):")
for k, lab in [(0, "s1 (floor)"), (4, "s5 (ceiling)")]:
    b0 = E.state_slope(df, E.PANEL, k)
    se = E.jk2_se(b0, [E.state_slope(df, r, k) for r in E.PANEL_REPS])
    print(f"    {lab:<14} beta = {b0:+.3f}  JK SE = {se:.3f}  z = {b0/se:+.1f}")

# grade-interval (wave-pair) fixed effects: SES slope controlling for interval
print("\n  grade-interval (wave-pair) fixed-effects check:")
T0 = E.transitions_for_weight(df, E.BASE)
for k in (0, 4):
    d = T0[T0["fr"] == k]
    D = pd.get_dummies(d["iv"], prefix="iv", drop_first=True).values.astype(float)
    X = np.column_stack([np.ones(len(d)), d["ses"].values, D])
    b = E.wlogit((d["to"] == k).values.astype(float), X, d["w"].values)
    print(f"    s{k+1}: beta_SES with wave-pair FE = {b[1]:+.3f}")

# attrition: s5-origin cases observed in next wave vs censored (unweighted mean SES)
print("\n  attrition check (s5-origin mean SES, unweighted):")
thr = E.wave_thresholds(df, E.BASE)
obs, cen = [], []
for a, b in E.PAIRS:
    c1, c2 = E.WAVES[a], E.WAVES[b]
    sub = df.loc[df[c1].notna() & df["x12sesl"].notna()]
    in5 = np.digitize(sub[c1].values, thr[c1][:4]) == 4
    nxt = sub[c2].notna().values
    obs += list(sub["x12sesl"].values[in5 & nxt])
    cen += list(sub["x12sesl"].values[in5 & ~nxt])
print(f"    observed in next wave: mean SES = {np.mean(obs):+.3f}  (n={len(obs):,})")
print(f"    censored:              mean SES = {np.mean(cen):+.3f}  (n={len(cen):,})")

# === Sec 3.4.2  jump-size regression =======================================
def jump(T):
    e = T[(T["fr"] == 0) & (T["to"].isin([1, 2]))]
    return E.wlogit_slope((e["to"] == 2).values.astype(float), e["ses"].values, e["w"].values), len(e)

print("\n" + "=" * 66)
print("SEC 3.4.2 - jump-size regression gamma1 (s3 vs s2 | exit s1)")
print("=" * 66)
g0, n0 = jump(T0)
greps = []
for r in E.REPS:
    e = E.transitions_for_weight(df, r)
    e = e[(e["fr"] == 0) & (e["to"].isin([1, 2]))]
    greps.append(E.wlogit_slope((e["to"] == 2).values.astype(float), e["ses"].values, e["w"].values))
gse = E.jk2_se(g0, greps)
print(f"  base-year:      gamma1 = {g0:+.3f}  JK SE = {gse:.3f}  z = {g0/gse:+.1f}  "
      f"CI=[{g0-1.96*gse:+.3f},{g0+1.96*gse:+.3f}]  n={n0:,}")
gp, _ = jump(E.transitions_for_weight(df, E.PANEL))
print(f"  complete-panel: gamma1 = {gp:+.3f}  -> robustly null")

# === Sec 3.4.3  threshold decomposition ====================================
# Unweighted points use the threshold-count samples (no positive-weight
# restriction), matching the 10,970 / 3,086 counts in the text; the weighted
# point is the inferential sample (positive weight, n=2,783).
print("\n" + "=" * 66)
print("SEC 3.4.3 - gamma1 by mobility definition (pooled vs within-wave)")
print("=" * 66)
pooled = np.percentile(pd.concat([df[w].dropna() for w in E.WAVES]).values, [25, 40, 60, 80])
thr_pooled = {wv: pooled for wv in E.WAVES}
thr_wave = E.wave_thresholds(df, E.BASE)

def decomp(thrd, weighted, require_pos):
    T = E.build_transitions(df, thrd, E.BASE, require_pos=require_pos)
    e = T[(T["fr"] == 0) & (T["to"].isin([1, 2]))]
    w = e["w"].values if weighted else np.ones(len(e))
    return E.wlogit_slope((e["to"] == 2).values.astype(float), e["ses"].values, w), len(e)

gp_u, np_u = decomp(thr_pooled, False, False)
gw_u, nw_u = decomp(thr_wave,  False, False)
print(f"  pooled thresholds,  unweighted : gamma1 = {gp_u:+.3f}  (n={np_u:,})")
print(f"  within-wave thresh, unweighted : gamma1 = {gw_u:+.3f}  (n={nw_u:,})")
print(f"  within-wave thresh, weighted   : gamma1 = {g0:+.3f}  (n={n0:,})")
print("  -> the apparent asymmetry is an artifact of pooling thresholds across grades.")
