"""
extract_ecls.py
===============
Rebuild ecls_k5_puf.csv from the raw NCES ECLS-K:2011 public-use file.

NCES distributes the K-5 public-use file as a fixed-width ASCII data file plus a
SAS/SPSS/Stata read-in program that defines the column layout. This script reads
the SAS read-in program to get the byte offsets of the Appendix-A variables, cuts
those columns out of the fixed-width data file, converts NCES missing/reserved
codes to NaN, and writes ecls_k5_puf.csv. It then verifies the result against the
pinned checksum, so a correct rebuild is confirmed automatically.

Source file (download separately; too large to redistribute):
    ChildK5p.zip  --  https://nces.ed.gov/ecls/kindergarten2011.asp
    (consult the current NCES terms of use before redistributing any extract)

Usage:
    python extract_ecls.py ChildK5p.zip                 # auto-detects .dat and .sas inside
    python extract_ecls.py DATA.dat  READIN.sas         # explicit paths
    python extract_ecls.py ChildK5p.zip --out ecls_k5_puf.csv

After it writes the CSV it runs verify_ecls.py automatically. A SHA-256 match
means the rebuild is byte-for-byte the extract used in the paper; a mismatch
almost always means the implied-decimal or missing-code handling below needs a
one-line adjustment for your copy (see NOTES).

NOTES / assumptions (adjust here if the checksum does not match):
  * The SAS layout is parsed from the common NCES pointer form
        @<start> <NAME> [$]<width>[.<decimals>]
    (1-based start column; `decimals` = implied decimals when the data field
    carries no explicit decimal point). If your read-in program uses column-range
    form (`NAME start-end`), fill LAYOUT_OVERRIDE below instead.
  * NCES reserved/missing codes are mapped to NaN (MISSING_CODES). ECLS uses
    negative sentinels such as -1, -7, -8, -9; extend the set if your codebook
    lists others.
"""
import sys
import io
import re
import os
import zipfile
import numpy as np
import pandas as pd

# Appendix-A variables to extract (kept in this order in the output CSV).
MATH = ["x2mscalk5", "x4mscalk5", "x6mscalk5", "x7mscalk5", "x8mscalk5", "x9mscalk5"]
SES  = ["x12sesl"]
W1   = ["w1c0"] + [f"w1c{i}" for i in range(1, 81)]
W9   = ["w9c29p_9a0"] + [f"w9c29p_9a{i}" for i in range(1, 81)]
ID   = ["childid"]
VARS = ID + MATH + SES + W1 + W9        # 170 columns

MISSING_CODES = {-1, -7, -8, -9}        # NCES reserved codes -> NaN (numeric fields)

# If the SAS layout cannot be parsed automatically, hand-fill this mapping:
#   name -> (start_col_1based, width, decimals)
LAYOUT_OVERRIDE = {}

_SAS_RE = re.compile(
    r"@\s*(\d+)\s+([A-Za-z_]\w*)\s+(\$)?\s*(\d+)\s*\.?\s*(\d*)", re.IGNORECASE)


def parse_sas_layout(sas_text):
    """Return {name_lower: (start, width, decimals)} from an NCES SAS read-in."""
    layout = {}
    for m in _SAS_RE.finditer(sas_text):
        start, name, dollar, width, dec = m.groups()
        layout[name.lower()] = (int(start), int(width),
                                 0 if dollar else (int(dec) if dec else 0))
    return layout


def _find(zf, suffixes):
    for n in zf.namelist():
        if n.lower().endswith(suffixes):
            return n
    return None


def _read_sources(args):
    """Return (data_bytes, sas_text) from a .zip or explicit .dat/.sas paths."""
    paths = [a for a in args if not a.startswith("--")]
    if len(paths) == 1 and paths[0].lower().endswith(".zip"):
        with zipfile.ZipFile(paths[0]) as zf:
            dat = _find(zf, (".dat", ".txt", ".asc"))
            sas = _find(zf, (".sas",))
            if not dat or not sas:
                sys.exit(f"Could not find both a data file and a .sas read-in "
                         f"inside {paths[0]} (found data={dat}, sas={sas}). "
                         f"Pass explicit paths: python extract_ecls.py DATA.dat READIN.sas")
            return zf.read(dat), zf.read(sas).decode("latin-1")
    dat = next((p for p in paths if p.lower().endswith((".dat", ".txt", ".asc"))), None)
    sas = next((p for p in paths if p.lower().endswith(".sas")), None)
    if not dat or not sas:
        sys.exit("Provide either ChildK5p.zip or an explicit DATA file and a .sas file.")
    return open(dat, "rb").read(), open(sas, "r", encoding="latin-1").read()


def extract(data_bytes, layout):
    fields = {}
    for name in VARS:
        spec = LAYOUT_OVERRIDE.get(name) or layout.get(name)
        if spec is None:
            sys.exit(f"Variable {name!r} not found in the SAS layout. "
                     f"Add it to LAYOUT_OVERRIDE or check the read-in program.")
        fields[name] = spec

    text = io.TextIOWrapper(io.BytesIO(data_bytes), encoding="latin-1", newline="")
    rows = []
    for line in text:
        line = line.rstrip("\r\n")
        rec = {}
        for name, (start, width, dec) in fields.items():
            raw = line[start - 1: start - 1 + width].strip()
            if raw == "" or raw == ".":
                rec[name] = np.nan
                continue
            if name == "childid":
                rec[name] = int(raw)
                continue
            val = float(raw)
            if "." not in raw and dec:            # apply implied decimals
                val /= 10 ** dec
            rec[name] = np.nan if int(val) in MISSING_CODES and val == int(val) else val
        rows.append(rec)
    return pd.DataFrame(rows, columns=VARS)


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    out = "ecls_k5_puf.csv"
    if "--out" in args:
        out = args[args.index("--out") + 1]

    data_bytes, sas_text = _read_sources(args)
    layout = parse_sas_layout(sas_text)
    print(f"parsed {len(layout)} variable positions from the SAS read-in")

    df = extract(data_bytes, layout)
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({df.shape[0]:,} rows x {df.shape[1]} cols)")

    try:
        import verify_ecls
        sys.exit(verify_ecls.main(out))
    except ImportError:
        print("(verify_ecls.py not found; run it separately to confirm the checksum)")


if __name__ == "__main__":
    main()
