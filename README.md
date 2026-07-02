# A Markov-Chain Framework for Educational Achievement Mobility and Transition Sensitivity

Replication code for the manuscript *"A Markov-Chain Framework for Educational
Achievement Mobility and Transition Sensitivity."* The model-side analyses
(Sections 4–6) require no external data; the empirical analyses (Section 3.3)
use the ECLS-K:2011 public-use file.

Every script below has been run against the current manuscript and reproduces the
reported numbers.

---

## Repository layout

| File | Purpose | Reproduces | External data |
|------|---------|-----------|----------------|
| `model.py` | Markov-chain core (importable) **and** headline model results | Table 3, §4.2, Table 4, §5.2, §6 | none |
| `model_robustness.py` | Model-side robustness checks | §4.2–4.4, §5.3, §5.4, §6, §7.1 | none |
| `ecls.py` | ECLS-K:2011 empirical core (importable) **and** headline empirical results | Table 1, Table 2 | `ecls_k5_puf.csv` |
| `ecls_robustness.py` | ECLS robustness checks | §3.3.1, §3.3.2, §3.3.3 | `ecls_k5_puf.csv` |
| `figures.py` | Figure 1 (ECLS sticky-floor / sticky-ceiling / per-state slopes) → `fig5_tails.png` | Figure 1 | `ecls_k5_puf.csv` |
| `fig_model_leverage.py` | Figure 2 (calibrated chain schematic + leverage bars) → `fig_model_leverage.png` | Figure 2 | none |
| `extract_ecls.py` | Rebuild `ecls_k5_puf.csv` from the raw NCES fixed-width file (auto-reads the SAS layout) | — | `ChildK5p.zip` |
| `verify_ecls.py` | Verify a local `ecls_k5_puf.csv` (SHA-256 + Appendix-A column/range checks) | — | `ecls_k5_puf.csv` |
| `CHECKSUMS.txt` | Pinned SHA-256 of the extract | — | — |
| `calibrated_diagonal.json` | Cached fitted diagonal for the calibrated chain | — | none |
| `requirements.txt` | Pinned Python dependencies | — | — |
| `paper.tex`, `paper.pdf` | Manuscript source and compiled PDF | — | — |

The two `*_robustness.py` scripts import their respective core module
(`model.py`, `ecls.py`) so that every number is computed from the same
calibrated chain / empirical pipeline as the headline results.

---

## Requirements and installation

Tested with **Python 3.11** (works on 3.9+).

```bash
pip install -r requirements.txt
```

`numpy` and `scipy` suffice for the model side; `pandas` is additionally needed
for the ECLS analyses and `matplotlib` for the figures.

---

## How to run

```bash
# 1. Headline model results (Table 3, §4.2, Table 4, §5.2, §6)
python model.py                 # loads the cached calibration; ~instant
python model.py --recalibrate   # re-runs the global optimizer, ~3 s

# 2. Model-side robustness (§4.2–4.4, §5.3, §5.4, §6, §7.1)
python model_robustness.py      # note: the §5.3 Monte Carlo (400k draws) is the slow tail

# 3. ECLS headline results (Table 1, Table 2) — requires the data file
python ecls.py                  # or:  python ecls.py path/to/ecls_k5_puf.csv

# 4. ECLS robustness (§3.3.1–3.3.3)
python ecls_robustness.py

# 5. Figures
python figures.py               # Figure 1 -> fig5_tails.png  (requires data)
python fig_model_leverage.py    # Figure 2 -> fig_model_leverage.png
```

### Entry points and flags

- **`model.py --recalibrate`** — forces a fresh differential-evolution +
  Nelder–Mead fit of the five diagonal persistences and rewrites
  `calibrated_diagonal.json`. Without the flag, `model.py` (and any script that
  imports it) loads the cached diagonal, so the tables reproduce in seconds
  rather than re-running the global optimizer on every import.
- **`model_robustness.py`** — the model-side robustness entry point. Reproduces
  the resource decomposition (§4.2), probability-floor sensitivity (§4.3),
  first-hitting vs. occupancy (§4.4), the Monte-Carlo leverage robustness over
  NAEP-consistent matrices (§5.3), the one-parameter calibration family and the
  (ρ, δ) / shared-persistence checks for Proposition 5.4 (§5.4), the tutoring
  sweep (§6), and expected absorption time (§7.1).
- **`ecls_robustness.py`** — the empirical robustness entry point. Reproduces
  the cross-state slope-difference tests, complete-panel-weight robustness,
  grade-interval fixed effects, and the attrition check (§3.3.1); the jump-size
  regression under both weights (§3.3.2); and the pooled-vs-within-wave threshold
  decomposition (§3.3.3).

### Approximate runtimes

| Command | Time |
|---------|------|
| `python model.py` (cached) | < 1 s |
| `python model.py --recalibrate` | ~3 s |
| `python fig_model_leverage.py` | a few seconds |
| `python ecls.py` | ~1–2 min (80 JK2 replicates, thresholds recomputed per replicate) |
| `python figures.py` | ~1–2 min (same JK2 pipeline) |
| `python ecls_robustness.py` | a few minutes |
| `python model_robustness.py` | a few minutes (dominated by the 400k-draw Monte Carlo) |

---

## Manuscript cross-reference

Sections, tables, figures, and results map to the code as follows.

**Model (no external data):**

| Manuscript | Code |
|------------|------|
| Table 3 — calibrated transition matrix | `model.py` |
| §4.2 — stationary distributions by resource stratum (reported inline, not a numbered table) | `model.py`, `model_robustness.py` |
| §4.3 — probability-floor sensitivity | `model_robustness.py` |
| §4.4 — first-hitting vs. year-12 occupancy | `model_robustness.py` |
| **Table 4** — 12-step top-state first-hitting probabilities | `model.py` |
| §5.2 — fundamental matrix Z, leverage `L_i`, adjacent transfers, three-state worked example | `model.py` |
| Proposition 5.1 — first-passage form of the leverage | proof in paper; `L_i = π_i π_5 m_{i5}` checked in `model.py` |
| Corollary 5.2 — top-state specialization | `model.py` |
| Corollary 5.3 — upward-adjacent leverage | `model.py` |
| §5.3 — Monte-Carlo leverage robustness | `model_robustness.py` |
| Proposition 5.4 — one-parameter calibration family, scale-invariant ranking | `model_robustness.py` |
| Corollary 5.5 — scale-invariance of the adjacent ranking | `model_robustness.py` |
| §6 — policy simulation | `model.py`, `model_robustness.py` |
| §7.1 — expected absorption time | `model_robustness.py` |

**Empirical (ECLS-K:2011):**

| Manuscript | Code |
|------------|------|
| Table 1 — SES-stratified transition matrices | `ecls.py` |
| Table 2 — per-state SES persistence slopes | `ecls.py` |
| §3.3.1 — tails result, difference tests, complete-panel & FE checks, attrition | `ecls_robustness.py` |
| §3.3.2 — jump-size regression (`γ₁`) | `ecls_robustness.py` |
| §3.3.3 — pooled-vs-within-wave threshold decomposition | `ecls_robustness.py` |
| Figure 1 — sticky floor / sticky ceiling / per-state slopes (`fig5_tails.png`) | `figures.py` |
| Figure 2 — calibrated chain schematic + leverage (`fig_model_leverage.png`) | `fig_model_leverage.py` |

> Note on numbering: the ECLS analyses are **Section 3.3** (subsections 3.3.1–3.3.4),
> the first-hitting probabilities are **Table 4** (not Table 5), and the
> theoretical results are **Propositions 5.1 and 5.4** with **Corollaries 5.2,
> 5.3, and 5.5**. Earlier drafts of this README used stale references; the labels
> above are current.

---

## Key numerical outputs (verified against the manuscript)

**Calibration / model (`model.py`):**

```
Stationary distribution : (0.38, 0.18, 0.18, 0.19, 0.07)   max deviation ~ 0
Z diagonal              : 12.0, 7.29, 6.29, 6.32, 2.81
Leverage L_i            : +1.55, +0.47, +0.39, +0.33, -0.12   (argmax s1; leads by 3.3x)
Table 4 first-hitting|s1: low 8.3%,  high 13.7%   (ratio 1.66x)
Policy (low-resource s1): baseline 8.3% -> tutoring 12.6% (+52%); expenditure shift 8.4%
```

**Empirical (`ecls.py`, `ecls_robustness.py`):**

```
Transitions             : 42,253   (low-SES 12,682 / high-SES 17,937)
Table 2 slopes (JK2)    : s1 beta=-0.451 (z=-9.1);  s5 beta=+0.456 (z=+10.3);  middle ~0
Difference tests        : beta1-beta2=-0.460 (z=-6.5);  beta5-beta4=+0.449 (z=+7.7)
Jump-size (within-wave) : gamma1=+0.028 (z=+0.3, n=2,298) -> null
```

---

## Data

`ecls_k5_puf.csv` is an extract of the **ECLS-K:2011 Kindergarten–Fifth Grade
public-use file** containing the variables listed in Appendix A of the paper
(spring-wave mathematics IRT scores `x2mscalk5`…`x9mscalk5`, the SES composite
`x12sesl`, the base-year child weight `w1c0` with its 80 JK2 replicates, the
complete-panel weight `w9c29p_9a0` with its replicates, and `childid`). The
source file (`ChildK5p.zip`) is freely available from NCES at
<https://nces.ed.gov/ecls/kindergarten2011.asp>; consult the current NCES terms
of use before redistributing any extract.

**Rebuild and verify.** Rather than relying on a redistributed CSV, rebuild the
extract from the raw NCES file and check it against the pinned SHA-256:

```bash
# rebuild ecls_k5_puf.csv from the raw file (reads the SAS read-in layout inside the zip)
python extract_ecls.py ChildK5p.zip

# verify any local copy (checksum + Appendix-A column/range validation)
python verify_ecls.py           # or:  sha256sum -c CHECKSUMS.txt
```

`extract_ecls.py` parses the column offsets from the SAS read-in program bundled
in `ChildK5p.zip`, cuts the Appendix-A variables from the fixed-width data file,
maps NCES reserved codes to `NaN`, writes the CSV, and then runs `verify_ecls.py`
automatically. The pinned checksum in `CHECKSUMS.txt`
(`b033e139…08086819`) is the extract used for the submitted paper, so a SHA-256
match confirms a byte-for-byte reproduction; a mismatch flags an implied-decimal
or missing-code adjustment to make for your copy (see the notes at the top of
`extract_ecls.py`).

**Survey weighting.** The primary analyses use the complete-panel (K–5) weight
(`w9c29p_9a0`), which carries the ECLS-K:2011 longitudinal nonresponse
adjustments across all waves used here. The base-year weight (`w1c0`), which
covers the full kindergarten cohort but does not adjust for later-wave attrition,
is reported as a broader-cohort sensitivity check in `ecls_robustness.py`. The
tails result is robust to the choice: under the base-year weight the sticky floor
is β₁ = −0.498 (z = −8.9) and the sticky ceiling is β₅ = +0.487 (z = +13.6), with
the middle states null. The primary weight is set by `PRIMARY`/`PRIMARY_REPS` in
`ecls.py`; swap it with `SENSITIVITY`/`SENSITIVITY_REPS` to reproduce the
base-year analysis.

---

## Reproducibility notes

- The calibrated diagonal is cached in `calibrated_diagonal.json`; delete it or
  pass `--recalibrate` to refit. The refined diagonal is clipped and asserted to
  lie in the calibration bounds `[0.60, 0.995]`.
- The stationary distribution is obtained from the constrained linear system
  (`πP = π`, `Σπ = 1`) with an explicit residual and positivity check.
- All ECLS inference uses the 80 JK2 replicate weights with the wave-specific
  score thresholds, state assignments, and regressions recomputed inside each
  replicate. Monte-Carlo robustness uses a fixed seed (11).
