========================================================================
A Markov-Chain Framework for Educational Achievement Mobility and
Transition Sensitivity 
========================================================================

Copyright: Michael Zhang

Every numerical result and the figure in the paper are reproduced by exactly one of five scripts.

------------------------------------------------------------------------
CONTENTS
------------------------------------------------------------------------

model.py             Markov core + headline model results: Table 3 (baseline P),
                     Table 4 (stationary by stratum), Table 5 (12-step first-
                     hitting), Sec 5.2 (Z + leverage L_i), Sec 6 (policy).

model_robustness.py  Sec 4.2 decomposition, Sec 4.3 floor sweep, Sec 4.4
                     first-hitting vs occupancy, Sec 5.3 Monte Carlo, Sec 5.4
                     Proposition 1 + c-sweep + (rho,delta) grid + shared
                     persistence, Sec 6 tutoring sweep.  (imports model.py)

ecls.py              ECLS core + Table 1 (matrices with positive-weight n's)
                     and Table 2 (per-state slopes, JK2 SE).

ecls_robustness.py   Sec 3.4.1 difference tests, complete-panel weight,
                     wave-pair fixed effects, attrition check;
                     Sec 3.4.2 jump-size; Sec 3.4.3 threshold decomposition.
                     (imports ecls.py)

figures.py           The paper's single figure -> fig5_tails.png  (\ref{fig:tails}).
                     (imports ecls.py)

ecls_k5_puf.csv      REQUIRED INPUT. Extract of the ECLS-K:2011 Kindergarten-
                     Fifth Grade public-use file (NCES 2019-050), containing the
                     Appendix-A variables only (childid, x12sesl, the six spring
                     math IRT scores, and the base-year + complete-panel replicate
                     weights). Extracted from the raw NCES fixed-width data file:
                       ChildK5p.zip
                       https://nces.ed.gov/ecls/data/2019/ChildK5p.zip
                     (public-use, no license required). The scripts read this
                     extract directly; regenerating it from ChildK5p.zip requires
                     the record layout shipped with that file.

------------------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------------------

  pip install numpy scipy pandas matplotlib

  python model.py             # Tables 3-5, Sec 5.2, Sec 6   (no data needed)
  python model_robustness.py  # Sec 4.2-4.4, 5.3, 5.4, 6 sweeps  (~few min: 400k MC)
  python ecls.py              # Tables 1-2                   (needs the CSV)
  python ecls_robustness.py   # Sec 3.4.1-3.4.3
  python figures.py           # fig5_tails.png

------------------------------------------------------------------------
KEY OUTPUTS (all verified to reproduce the paper exactly)
------------------------------------------------------------------------

model.py           Z diag 12.01/7.29/6.29/6.33/2.81; L_i +1.554/+0.470/+0.390/
                   +0.331/-0.122 (s1 leads 3.3x); pi1 low/high 48.7/26.3;
                   P(T<=12|s1) 8.3/13.7 (1.66x); policy 8.3->12.6 (+52%)/8.4/12.8.
model_robustness   decomposition 22.1 vs 3.3 pp; floor 8.2-8.3% (1.66-1.68);
                   occupancy 2.4/4.7; Prop 1 machine-exact, c=0.112; Monte Carlo
                   kept 3,499/780/54 s1-first 100%; tutoring 8.3->26.9%.
ecls.py            51,982 transitions (16,450/18,631); Table 1 s1 rows 0.806
                   (n=6,546)/0.634 (n=1,757); Table 2 -0.498/-0.041/-0.010/
                   +0.000/+0.487, JK2 SE 0.056/0.096/0.037/0.035/0.036.
ecls_robustness    beta1-beta2=-0.456 (z-6.2); beta5-beta4=+0.486 (z+9.2);
                   complete-panel s1 z=-9.1, s5 +0.456 (z+10.3);
                   wave-pair FE s1 -0.500, s5 +0.485;
                   attrition mean SES observed +0.443 / censored +0.430;
                   gamma1=+0.092 (SE 0.090, z+1.0, n 2,783), panel +0.028;
                   decomposition +0.520 (n 10,970) / +0.117 (n 3,086) /
                   +0.092 (n 2,783).

Every number cited in the paper is backed by one of these scripts and
reproduces exactly.
------------------------------------------------------------------------