"""
model_robustness.py
===================
Robustness checks for the model side. Imports the calibrated chain from model.py
and reproduces every non-headline model claim, each computed once:

    Sec 4.2  resource decomposition (income vs funding channel)
    Sec 4.3  probability-floor sensitivity of the resource gap
    Sec 4.4  first-hitting vs year-12 occupancy
    Sec 5.3  Monte-Carlo leverage robustness across NAEP-consistent matrices
    Sec 5.4  Proposition 1 (one-parameter family, Z(c)=(1/c)A#+1.pi, c-sweep),
             the (rho, delta) kernel grid, and the shared-persistence restriction
    Sec 6    tutoring-boost sweep

Run:  python model_robustness.py
"""
import numpy as np
import model as m

P = m.P_base
pi = m.stationary(P)


def leverage_vec(M):
    Z, p = m.fundamental_Z(M)
    return np.array([max(p[i] * (Z[j, 4] - Z[i, 4]) for j in range(5) if j != i)
                     for i in range(5)])


# === Sec 4.2  resource decomposition =======================================
print("=" * 66)
print("SEC 4.2 - resource decomposition of the pi_1 gap")
print("=" * 66)
print(f"  income shift  alpha_inc*60 = {m.ALPHA['income']*60:+.4f}")
print(f"  funding shift alpha_pp*9   = {m.ALPHA['perpupil']*9:+.4f}"
      f"   (ratio ~ {m.ALPHA['income']*60/(m.ALPHA['perpupil']*9):.0f}:1)")
for label, lo, hi in [
    ("income only (20 vs 80)",  dict(income=20, perpupil=15.591), dict(income=80, perpupil=15.591)),
    ("funding only ($9k vs $18k)", dict(income=50, perpupil=9),   dict(income=50, perpupil=18)),
    ("both (low- vs high-resource)", m.LOW_RES, m.HIGH_RES)]:
    a = m.stationary(m.apply_covariates(P, **lo))[0] * 100
    b = m.stationary(m.apply_covariates(P, **hi))[0] * 100
    print(f"  {label:<30} pi1 {a:5.1f}% -> {b:5.1f}%  (gap {a-b:+.1f} pp)")

# === Sec 4.3  probability-floor sensitivity ================================
print("\n" + "=" * 66)
print("SEC 4.3 - probability-floor sensitivity of the resource gap")
print("=" * 66)
def cov_floor(income, perpupil, floor):
    shift = (m.ALPHA['income']*(income-m.C0['income']) +
             m.ALPHA['perpupil']*(perpupil-m.C0['perpupil']))
    Q = P.copy().astype(float)
    for i in range(4):
        Q[i, i+1] = Q[i, i+1] + shift
        Q[i, i]   = Q[i, i]   - shift
        Q[i, :] = np.maximum(Q[i, :], floor); Q[i, :] /= Q[i, :].sum()
    return Q
for fl in (0.0, 0.0001, 0.001, 0.005):
    lo, hi = cov_floor(20, 9, fl), cov_floor(80, 18, fl)
    print(f"  floor={fl:.4f}:  low P12|s1={m.p12yr(lo)[0]*100:.1f}%  "
          f"high={m.p12yr(hi)[0]*100:.1f}%  ratio={m.p12yr(hi)[0]/m.p12yr(lo)[0]:.2f}")

# === Sec 4.4  first-hitting vs occupancy ===================================
print("\n" + "=" * 66)
print("SEC 4.4 - first-hitting Pr(T<=12) vs year-12 occupancy Pr(X12=s5)")
print("=" * 66)
Plow, Phigh = m.apply_covariates(P, **m.LOW_RES), m.apply_covariates(P, **m.HIGH_RES)
hl, hh = m.p12yr(Plow), m.p12yr(Phigh)
ol = np.linalg.matrix_power(Plow, 12)[:, 4]
oh = np.linalg.matrix_power(Phigh, 12)[:, 4]
print(f"  {'start':<6}{'hit low':>9}{'hit high':>10}{'occ low':>10}{'occ high':>10}")
for i in range(4):
    print(f"  s{i+1}   {hl[i]*100:>8.1f}%{hh[i]*100:>9.1f}%{ol[i]*100:>9.1f}%{oh[i]*100:>9.1f}%")

# === Sec 5.4  Proposition 1 ================================================
print("\n" + "=" * 66)
print("SEC 5.4 - Proposition 1: one-parameter family and invariant ranking")
print("=" * 66)
D = 1 - np.diag(P)
M = np.array([[0 if j == i else P[i, j] / D[i] for j in range(5)] for i in range(5)])
mu = m.stationary(M)
c_vec = pi * D / mu
c0 = c_vec.mean()
print(f"  pi_i(1-p_ii)/mu_i constant: spread std/mean = {c_vec.std()/c_vec.mean():.2e}"
      f"   c = {c0:.5f}")
A = np.diag(mu / pi) @ (np.eye(5) - M)
Ash = np.linalg.inv(A + np.outer(np.ones(5), pi)) - np.outer(np.ones(5), pi)
Z, _ = m.fundamental_Z(P)
Zpred = (1 / c0) * Ash + np.outer(np.ones(5), pi)
print(f"  A1=0 ({np.max(np.abs(A@np.ones(5))):.1e}), piA=0 ({np.max(np.abs(pi@A)):.1e}),"
      f"  max|Z-((1/c)A#+1.pi)| = {np.max(np.abs(Z-Zpred)):.1e}")
print("  c-sweep (scale-dependent finite-step quantities):")
for c in (0.02, 0.06, 0.113):
    Dc = c * mu / pi
    Pc = np.array([[Dc[i]*M[i, j] if j != i else 1-Dc[i] for j in range(5)] for i in range(5)])
    Pa = Pc.copy(); Pa[4] = [0, 0, 0, 0, 1]
    ET = np.linalg.solve(np.eye(4) - Pa[:4, :4], np.ones(4))[0]
    print(f"    c={c:.3f}: P(T<=12|s1)={m.p12yr(Pc)[0]*100:4.1f}%  E[T|s1]={ET:4.0f}"
          f"  argmaxL=s{int(np.argmax(leverage_vec(Pc)))+1}")

def kernel(rho, delta):
    K = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            if j != i:
                K[i, j] = (1.0 if j > i else delta) * rho ** (abs(j-i)-1)
        K[i] /= K[i].sum()
    return K
cells = s1first = 0
for rho in (0.5, 0.6, 0.7, 0.8, 0.9):
    for delta in (0.2, 0.3, 0.4, 0.5, 0.6):
        Kd = 0.08 * m.stationary(kernel(rho, delta)) / pi
        if (Kd <= 0).any() or (Kd >= 1).any():
            continue
        Kk = kernel(rho, delta)
        Pk = np.array([[Kd[i]*Kk[i, j] if j != i else 1-Kd[i] for j in range(5)] for i in range(5)])
        cells += 1; s1first += int(np.argmax(leverage_vec(Pk)) == 0)
print(f"  (rho,delta) grid: s1 highest-leverage in {s1first}/{cells} kernels")
psh = minimize_shared = None
for p in (0.80, 0.902, 0.95):
    Lsh = leverage_vec(m.build_P([p]*5))
    print(f"  shared-persistence p={p}: argmax = s{int(np.argmax(Lsh))+1}"
          f"  (L=[{', '.join(f'{v:+.2f}' for v in Lsh)}])")

# === Sec 5.3  Monte-Carlo leverage robustness ==============================
print("\n" + "=" * 66)
print("SEC 5.3 - Monte-Carlo leverage robustness (seed 11, 400,000 matrices)")
print("=" * 66)
def sample_matrix(rng):
    d = rng.uniform(0.55, 0.97, 5); Pm = np.zeros((5, 5))
    for i in range(5):
        w = rng.uniform(0.15, 0.6) ** np.abs(np.arange(5)-i) * np.exp(rng.normal(0, 0.5, 5))
        w[i] = 0
        if w.sum() == 0: w[(i+1) % 5] = 1
        Pm[i] = (1-d[i]) * w / w.sum(); Pm[i, i] = d[i]
    return Pm
rng = np.random.default_rng(11)
keeps = {0.06: [], 0.04: [], 0.02: []}
for _ in range(400_000):
    Pm = sample_matrix(rng)
    if not all(Pm[i, i] >= Pm[i].max() - 1e-12 for i in range(5)):
        continue
    try:
        s = m.stationary(Pm)
    except np.linalg.LinAlgError:
        continue
    dev = np.max(np.abs(s - m.NAEP_TARGET))
    for tol in keeps:
        if dev < tol:
            Lk = leverage_vec(Pm)
            keeps[tol].append((int(np.argmax(Lk)) == 0, Pm[0, 0], Lk[0] - max(Lk[1:])))
for tol in sorted(keeps, reverse=True):
    arr = keeps[tol]; n = len(arr)
    s1 = sum(a for a, _, _ in arr); p11 = [p for _, p, _ in arr]
    dd = np.array([d for _, _, d in arr])
    # exact one-sided 95% lower bound on the s1-first proportion (k successes of n)
    lb = 0.05 ** (1.0 / n) if s1 == n and n > 0 else np.nan
    print(f"  tol<{tol}: kept {n:5d}  s1-first {100*s1/max(n,1):5.1f}%  "
          f"(95% one-sided LB {lb:.3f})  "
          f"p11 in [{min(p11):.3f},{max(p11):.3f}]  "
          f"margin min/med/max = {dd.min():+.3f}/{np.median(dd):+.3f}/{dd.max():+.3f}")

# === Sec 6  tutoring sweep =================================================
print("\n" + "=" * 66)
print("SEC 6 - tutoring-boost sweep at low-resource s1")
print("=" * 66)
for b in (0.0, 0.05, 0.10, 0.20, 0.30):
    print(f"  boost {b*100:4.0f}pp -> P(T<=12|s1) = {m.p12yr(m.apply_tutoring(Plow, b))[0]*100:.1f}%")

# === Sec 7.1  expected absorption time to s5 from s1 =======================
print("\n" + "=" * 66)
print("SEC 7.1 - expected absorption time to s5 from s1")
print("=" * 66)
def absorption_time_s1(Pm):
    A = Pm.copy().astype(float); A[4] = [0, 0, 0, 0, 1]
    return np.linalg.solve(np.eye(4) - A[:4, :4], np.ones(4))[0]
print(f"  baseline P    : E[T|s1] = {absorption_time_s1(P):.0f} model steps")
print(f"  low-resource  : E[T|s1] = {absorption_time_s1(Plow):.0f} model steps")
