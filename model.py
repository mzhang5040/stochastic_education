"""
model.py
========
Markov-chain core (importable) and the paper's headline model results.

As a library it exposes the calibrated chain and the operators used throughout:
    build_P, stationary, calibrate, apply_covariates, apply_tutoring,
    p12yr, fundamental_Z, leverage.
Importing the module calibrates once and exposes `P_base`, `pi_base`.

Run directly to print:
    Table 3  baseline transition matrix P
    Table 4  stationary distributions by resource stratum
    Table 5  12-step top-state first-hitting probabilities
    Sec 5.2  fundamental matrix Z, leverage L_i, adjacent transfers
    Sec 6    policy simulation table

Robustness checks (floor sweep, Monte Carlo, Proposition 1, tutoring sweep,
occupancy-vs-first-hitting, resource decomposition) live in model_robustness.py.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution

np.random.seed(0)

# NAEP 2022 Grade-8 math, all students: 38% Below Basic, 36% Basic (split
# equally into s2/s3), 19% Proficient, 7% Advanced.
NAEP_TARGET = np.array([0.38, 0.18, 0.18, 0.19, 0.07])

ALPHA = {'income': 0.0008, 'perpupil': 0.0005}
C0    = {'income': 50.0,   'perpupil': 15.591}   # $15,591 = FY2022 mean
TUTORING_BOOST = 0.05
RHO, DELTA = 0.7, 0.4

# Resource-stratum and policy covariate values (Sections 4, 6)
LOW_RES  = dict(income=20, perpupil=9)
HIGH_RES = dict(income=80, perpupil=18)
PP_EQUITY = 15.591   # funding-equity scenario raises per-pupil to the national mean


def build_P(params):
    """Transition matrix from the five diagonal persistences (Eq. 1)."""
    n = 5
    p_diag = np.clip(params, 0.50, 0.995)
    P = np.zeros((n, n))
    for i in range(n):
        P[i, i] = p_diag[i]
        remain = 1.0 - p_diag[i]
        w = np.zeros(n)
        for j in range(n):
            if j == i:
                continue
            w[j] = (1.0 if j > i else DELTA) * (RHO ** (abs(j - i) - 1))
        for j in range(n):
            if j != i:
                P[i, j] = max(remain * w[j] / w.sum(), 0.001)
        P[i, :] /= P[i, :].sum()
    return P


def stationary(P):
    ev, evec = np.linalg.eig(P.T)
    pi = np.abs(np.real(evec[:, np.argmin(np.abs(ev - 1.0))]))
    return pi / pi.sum()


def calibrate(target=NAEP_TARGET):
    loss = lambda p: np.sum((stationary(build_P(p)) - target) ** 2)
    r = differential_evolution(loss, [(0.60, 0.995)] * 5, seed=42,
                               maxiter=3000, tol=1e-12, workers=1)
    r = minimize(loss, r.x, method='Nelder-Mead',
                 options={'maxiter': 100000, 'xatol': 1e-12, 'fatol': 1e-12})
    P = build_P(r.x)
    return P, stationary(P), np.max(np.abs(stationary(P) - target))


def apply_covariates(P_base, income, perpupil):
    """Income + funding shift on adjacent upward transitions (Eq. 2): the shift
    is moved between p_{i,i+1} and p_ii, then every entry is floored at 0.001 and
    the row renormalized (steps iii-iv). The 0.001 floor is the only imposed bound."""
    shift = (ALPHA['income']   * (income   - C0['income']) +
             ALPHA['perpupil'] * (perpupil - C0['perpupil']))
    P = P_base.copy().astype(float)
    for i in range(4):
        P[i, i+1] = P[i, i+1] + shift
        P[i, i]   = P[i, i]   - shift
        P[i, :] = np.maximum(P[i, :], 0.001)
        P[i, :] /= P[i, :].sum()
    return P


def apply_tutoring(P, boost=TUTORING_BOOST):
    """+boost on s1->s2 and s2->s3, absorbed from the diagonal; floor at 0.001
    and renormalize."""
    P = P.copy().astype(float)
    for i in [0, 1]:
        P[i, i+1] = P[i, i+1] + boost
        P[i, i]   = P[i, i]   - boost
        P[i, :] = np.maximum(P[i, :], 0.001)
        P[i, :] /= P[i, :].sum()
    return P


def p12yr(P, n=12):
    """12-step top-state first-hitting probabilities (s5 absorbing, Eq. 4)."""
    A = P.copy().astype(float)
    A[4, :] = [0, 0, 0, 0, 1]
    Q = A[:4, :4]
    Qn = np.linalg.matrix_power(Q, n)
    return np.array([1 - Qn[i, :].sum() for i in range(4)])


def fundamental_Z(P):
    """Kemeny-Snell fundamental matrix Z = (I - P + 1 pi)^{-1}."""
    pi = stationary(P)
    n = len(pi)
    return np.linalg.inv(np.eye(n) - P + np.ones((n, 1)) @ pi.reshape(1, n)), pi


def leverage(P):
    """
    Row-sum-preserving leverage of pi_5 (Eq. 5-6): mass moved from p_ii to p_ij,
    s5 coordinate pi_i (Z_j5 - Z_i5). Returns (L, argmax destinations, adjacent).
    """
    Z, pi = fundamental_Z(P)
    L, best = np.zeros(5), np.zeros(5, int)
    for i in range(5):
        vals = {j: pi[i] * (Z[j, 4] - Z[i, 4]) for j in range(5) if j != i}
        best[i] = max(vals, key=vals.get)
        L[i] = vals[best[i]]
    adjacent = np.array([pi[i] * (Z[i+1, 4] - Z[i, 4]) for i in range(4)])
    return L, best, adjacent


def mfpt_to_top(P):
    """Mean first-passage times m_{i5} from transient states s1..s4 to s5
    (expected steps to first reach the top state). Prop.: L_i = pi_i pi_5 m_i5."""
    Q = np.asarray(P, float)[:4, :4]
    return np.linalg.solve(np.eye(4) - Q, np.ones(4))


def worked_example():
    """Three-state illustration (Section 5.2): fundamental matrix Z, mean first-
    passage to the target, unrestricted leverage L_i^(t), the adjacent-leverage
    corollary A_i=pi_i*pi_t*(m_it-m_{i+1,t}), and a finite-difference check of the
    constrained-sensitivity formula. Target is the top state s3."""
    P = np.array([[0.85, 0.12, 0.03],
                  [0.10, 0.75, 0.15],
                  [0.08, 0.22, 0.70]])
    t = 2
    Z, pi = fundamental_Z(P)
    m = np.array([(Z[t, t] - Z[i, t]) / pi[t] for i in range(3)])      # m_it (m_tt=0)
    L = [max(pi[i] * (Z[j, t] - Z[i, t]) for j in range(3) if j != i) for i in range(3)]
    A1 = pi[0] * pi[t] * (m[0] - m[1])          # Cor: i=1, i+1=2 != target
    L2 = pi[1] * pi[t] * m[1]                    # i=2 borders target -> full leverage
    eps = 1e-6
    Pe = P.copy(); Pe[0, 0] -= eps; Pe[0, 2] += eps
    fd = (stationary(Pe)[t] - pi[t]) / eps
    an = pi[0] * (Z[2, t] - Z[0, t])
    print("\nSECTION 5.2 - three-state worked example (target s3)")
    print("  pi      = " + ", ".join(f"{v:.4f}" for v in pi))
    print("  Z =\n" + "\n".join("    " + "  ".join(f"{Z[i, j]:+.4f}" for j in range(3))
                                 for i in range(3)))
    print(f"  m_i3    = m13={m[0]:.2f}, m23={m[1]:.2f}")
    print("  L_i^(3) = " + ", ".join(f"L{i+1}={L[i]:+.3f}" for i in range(3)))
    print(f"  adjacent: s1->s2  A1=pi1*pi3*(m13-m23)={A1:+.4f};  "
          f"s2->s3 is direct-to-target, full leverage L2={L2:+.3f}")
    print(f"  Eq.(5) finite-diff s1->s3: numeric={fd:+.5f}  analytic={an:+.5f}")


# Calibrate once on import (silent); importers use model.P_base etc.
P_base, pi_base, _CAL_DEV = calibrate()


def _main():
    P_low  = apply_covariates(P_base, **LOW_RES)
    P_high = apply_covariates(P_base, **HIGH_RES)
    pi_low, pi_high = stationary(P_low), stationary(P_high)
    pl, ph = p12yr(P_low), p12yr(P_high)
    base = pl[0]
    Z, pi = fundamental_Z(P_base)
    L, best, adj = leverage(P_base)

    print("=" * 66)
    print(f"CALIBRATION  max deviation from NAEP target = {_CAL_DEV*100:.4f}%")
    print("=" * 66)

    print("\nTABLE 3 - baseline transition matrix P")
    print("         s1      s2      s3      s4      s5")
    for i in range(5):
        print(f"  s{i+1}   " + "  ".join(f"{v:.3f}" for v in P_base[i]))

    print("\nTABLE 4 - stationary distributions by resource stratum")
    print(f"  {'state':<6}{'low':>10}{'high':>10}{'gap':>10}")
    for i in range(5):
        g = (pi_low[i] - pi_high[i]) * 100
        print(f"  pi{i+1}  {pi_low[i]*100:>9.1f}%{pi_high[i]*100:>9.1f}%{g:>+8.1f}pp")

    print("\nTABLE 5 - 12-step top-state first-hitting probabilities")
    print(f"  {'start':<6}{'low':>10}{'high':>10}{'ratio':>8}")
    for i in range(4):
        print(f"  s{i+1}   {pl[i]*100:>9.1f}%{ph[i]*100:>9.1f}%{ph[i]/pl[i]:>7.2f}x")

    print("\nSECTION 5.2 - fundamental matrix Z and leverage")
    print("  Z diag: " + ", ".join(f"Z{i+1}{i+1}={Z[i,i]:.2f}" for i in range(5)))
    print("  leverage L_i (most favorable transfer):")
    for i in range(5):
        print(f"    L{i+1} = {L[i]:+.3f}  (via s{i+1}->s{best[i]+1})")
    print(f"  s1 leads next state by {L[0]/np.sort(L)[-2]:.1f}x")
    print("  adjacent transfers: " +
          ", ".join(f"s{i+1}->s{i+2}={adj[i]:+.3f}" for i in range(4)))
    mfpt = mfpt_to_top(P_base)
    print("  mean first-passage m_i5 (s_i -> s5): " +
          ", ".join(f"m{i+1}5={mfpt[i]:.1f}" for i in range(4)))
    print("  Prop 1 check  L_i = pi_i*pi_5*m_i5 (i=1..4): " +
          ", ".join(f"{pi[i]*pi[4]*mfpt[i]:+.3f}" for i in range(4)))
    adj_cor = [pi[i]*pi[4]*(mfpt[i]-mfpt[i+1]) for i in range(3)] + [pi[3]*pi[4]*mfpt[3]]
    print("  Cor 2 check A_i (adjacent) = pi_i*pi_5*(m_i5-m_{i+1,5}), A_4=pi_4*pi_5*m_45: " +
          ", ".join(f"{a:+.3f}" for a in adj_cor))

    worked_example()

    print("\nSECTION 6 - policy simulation")
    P_S1 = apply_tutoring(P_low)
    P_S2 = apply_covariates(P_base, income=LOW_RES['income'], perpupil=PP_EQUITY)
    P_S3 = apply_tutoring(P_S2)
    print(f"  {'scenario':<24}{'P12|s1':>9}{'P12|s2':>9}{'pi5':>8}{'gain@s1':>9}")
    for name, Psc in [('Baseline (low-resource)', P_low),
                      ('Reference (high-resource)', P_high),
                      ('S1: Tutoring', P_S1),
                      ('S2: Funding equity', P_S2),
                      ('S3: Combined', P_S3)]:
        pr, pisc = p12yr(Psc), stationary(Psc)
        gain = '---' if 'resource' in name else f"{(pr[0]-base)/base*100:+.0f}%"
        print(f"  {name:<24}{pr[0]*100:>8.1f}%{pr[1]*100:>8.1f}%{pisc[4]*100:>7.1f}%{gain:>9}")


if __name__ == "__main__":
    _main()
