from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("TRAINING DATA SAFETY CHECK")
print("=" * 80)

print("\nTotal rows:", len(df))

# ------------------------------------------------------------
# Check columns
# ------------------------------------------------------------

print("\nColumns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# Filter A -> D
# ------------------------------------------------------------

if "grade_from" in df.columns and "grade_to" in df.columns:

    transition = df[
        (df["grade_from"].astype(str).str.upper() == "A") &
        (df["grade_to"].astype(str).str.upper() == "D")
    ].copy()

else:

    print("\nERROR: grade_from / grade_to columns not found.")
    raise SystemExit


print("\nA -> D records:", len(transition))

# ------------------------------------------------------------
# Basis Weight statistics
# ------------------------------------------------------------

if "basis_weight" not in transition.columns:

    print("\nERROR: basis_weight column not found.")
    raise SystemExit


print("\nA -> D Basis Weight statistics:")
print(
    transition["basis_weight"].describe()
)

# ------------------------------------------------------------
# Safe range for 100 GSM
# ------------------------------------------------------------

TARGET = 100.0

LOWER = TARGET * 0.975
UPPER = TARGET * 1.025

safe = transition[
    transition["basis_weight"].between(
        LOWER,
        UPPER
    )
].copy()

print("\n" + "=" * 80)
print("SAFE RANGE")
print("=" * 80)

print(f"Target       : {TARGET:.2f} GSM")
print(f"Minimum safe : {LOWER:.2f} GSM")
print(f"Maximum safe : {UPPER:.2f} GSM")

print("\nSafe A -> D historical records:", len(safe))

if len(safe) > 0:

    print("\nYES - historical safe examples exist.")

    columns = [
        "machine_speed",
        "stock_flow",
        "steam_pressure",
        "basis_weight"
    ]

    available = [
        c for c in columns
        if c in safe.columns
    ]

    print("\nExamples:")
    print(
        safe[available]
        .sort_values("basis_weight")
        .head(20)
        .to_string(index=False)
    )

else:

    print("\nNO - there are no historical A -> D records")
    print("inside the 97.5 - 102.5 GSM safe range.")

# ------------------------------------------------------------
# Overall dataset check
# ------------------------------------------------------------

overall_safe = df[
    df["basis_weight"].between(
        LOWER,
        UPPER
    )
]

print("\n" + "=" * 80)
print("OVERALL DATASET")
print("=" * 80)

print(
    "Rows within 97.5-102.5 GSM:",
    len(overall_safe)
)

if len(overall_safe) > 0:

    print("\nBasis Weight examples from entire dataset:")

    columns = [
        "grade_from",
        "grade_to",
        "machine_speed",
        "stock_flow",
        "steam_pressure",
        "basis_weight"
    ]

    available = [
        c for c in columns
        if c in overall_safe.columns
    ]

    print(
        overall_safe[available]
        .head(20)
        .to_string(index=False)
    )

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)