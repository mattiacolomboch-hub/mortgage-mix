"""Clean SNB raw downloads -> data/clean/*.csv. Raw files are never edited.
Sources (downloaded by Mattia on 2026-11-09, SNB publishing date 2026-11-02 14:30):
 - zimoma (Money market rates): SARON (D0=SARON, from 1999-06), CHF LIBOR 3M (D0=3M0, 1989-01..2021-12)
 - zikrepro (Published interest rates for new business, mean value D0=M), from 2007-12:
   MV variable mortgage; 11,20,30,40,50,60,70,80,9,10 = fixed 1..10y; HYP_11..15 = fixed 11..15y;
   SARJ03 / SARJ05 / SARJOBD = SARON-linked 3y / 5y / unlimited (from 2021-01).
   Code mapping verified against the portal table (Sep 2026: 1y 1.61, 2y 1.76, 3y 1.90, 5y 2.03, 7y 2.11, 10y 2.20).
 - zikrepro part 1 (until 1961-04) contains no mortgage values: not used.
Units: % per year in raw files; kept in % here."""
import csv, os
RAW = "../data/raw/"; OUT = "../data/clean/"; os.makedirs(OUT, exist_ok=True)

def read(path):
    L = open(path, encoding="utf-8-sig").read().splitlines()
    s = next(i for i, l in enumerate(L) if l.replace('"', '').startswith("Date"))
    return list(csv.reader(L[s + 1:], delimiter=";"))

# SARON + LIBOR
sar, lib = {}, {}
for r in read(RAW + "zimoma_SARON_LIBOR3M.csv"):
    if len(r) < 3 or r[-1] == "": continue
    (sar if r[1] == "SARON" else lib)[r[0]] = float(r[-1])
with open(OUT + "saron_monthly.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["month", "saron_pct", "libor3m_pct"])
    for d in sorted(set(sar) | set(lib)): w.writerow([d, sar.get(d, ""), lib.get(d, "")])

# Mortgage rates
CODES = {"MV": "variable", "SARJ03": "saron_3y", "SARJ05": "saron_5y", "SARJOBD": "saron_unlimited",
         "11": "fixed_1y", "20": "fixed_2y", "30": "fixed_3y", "40": "fixed_4y", "50": "fixed_5y",
         "60": "fixed_6y", "70": "fixed_7y", "80": "fixed_8y", "9": "fixed_9y", "10": "fixed_10y",
         "HYP_15": "fixed_15y"}
mort = {}
for r in read(RAW + "zikrepro_part2_from_1961-05.csv"):
    if len(r) < 4 or r[-1] == "" or r[1] != "M" or r[2] not in CODES: continue
    mort.setdefault(r[0], {})[CODES[r[2]]] = float(r[-1])
cols = list(CODES.values())
with open(OUT + "mortgage_rates_monthly.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["month"] + [c + "_pct" for c in cols])
    for d in sorted(mort): w.writerow([d] + [mort[d].get(c, "") for c in cols])
print("SARON", min(sar), max(sar), len(sar), "| mortgages", min(mort), max(mort), len(mort))
