"""
verify_ecls.py
==============
Confirms that a local ecls_k5_puf.csv is the exact extract behind the paper.

It (1) checks the file's SHA-256 against the pinned value below, and (2)
validates that every Appendix-A variable is present and within its expected
range. Anyone who has the CSV can run this to confirm they hold the same data;
anyone rebuilding the CSV from the raw NCES file (see extract_ecls.py) can run it
to confirm their rebuild matches.

Run:  python verify_ecls.py            # checks ./ecls_k5_puf.csv
      python verify_ecls.py FILE.csv
Exit code 0 = all checks pass, 1 = a check failed.
"""
import sys
import hashlib
import numpy as np
import pandas as pd

# ---- pinned reference (the extract used for the submitted paper) -------------
EXPECTED_SHA256 = "b033e1397ef3db3bf2f885417d31540585520de0fa178cc7b2dbfb7408086819"
EXPECTED_ROWS = 18174

# ---- Appendix-A variables ----------------------------------------------------
MATH = ["x2mscalk5", "x4mscalk5", "x6mscalk5", "x7mscalk5", "x8mscalk5", "x9mscalk5"]
SES  = ["x12sesl"]
W1   = ["w1c0"] + [f"w1c{i}" for i in range(1, 81)]                 # base-year + 80 JK2
W9   = ["w9c29p_9a0"] + [f"w9c29p_9a{i}" for i in range(1, 81)]     # complete-panel + 80
ID   = ["childid"]
APPENDIX_A = MATH + SES + W1 + W9 + ID                              # 170 columns
WEIGHTS = W1 + W9


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(path="ecls_k5_puf.csv"):
    checks = []   # (ok, message)

    digest = sha256(path)
    checks.append((digest == EXPECTED_SHA256,
                   f"SHA-256 {digest[:16]}... "
                   + ("matches pinned value" if digest == EXPECTED_SHA256
                      else f"DOES NOT MATCH pinned {EXPECTED_SHA256[:16]}...")))

    df = pd.read_csv(path)

    checks.append((df.shape[0] == EXPECTED_ROWS,
                   f"row count {df.shape[0]:,} (expected {EXPECTED_ROWS:,})"))

    missing = [c for c in APPENDIX_A if c not in df.columns]
    checks.append((not missing,
                   f"all {len(APPENDIX_A)} Appendix-A variables present"
                   + ("" if not missing else f"  MISSING: {missing}")))

    if not missing:
        checks.append((df["childid"].is_unique and
                       np.issubdtype(df["childid"].dtype, np.integer),
                       "childid is a unique integer key"))

        wmin = min(df[c].min() for c in WEIGHTS)
        wnan = any(df[c].isna().any() for c in WEIGHTS)
        checks.append((wmin >= 0 and not wnan,
                       f"all {len(WEIGHTS)} weight columns non-negative and complete "
                       f"(min={wmin:.3f})"))

        smin = min(np.nanmin(df[c]) for c in MATH)
        smax = max(np.nanmax(df[c]) for c in MATH)
        checks.append((0 < smin and smax < 200,
                       f"math IRT scores in range ({smin:.1f}, {smax:.1f})"))

        e = df["x12sesl"]
        checks.append((-5 < np.nanmin(e) and np.nanmax(e) < 5,
                       f"SES composite in range ({np.nanmin(e):.2f}, {np.nanmax(e):.2f})"))

    print("=" * 64)
    print(f"ECLS extract verification: {path}")
    print("=" * 64)
    for ok, msg in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {msg}")
    allok = all(ok for ok, _ in checks)
    print("-" * 64)
    print("RESULT:", "ALL CHECKS PASSED" if allok else "ONE OR MORE CHECKS FAILED")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "ecls_k5_puf.csv"))
