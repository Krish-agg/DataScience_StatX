import numpy as np
import pandas as pd

RAW = "dataset-train-LT.csv"
OUT = "dataset-train-LT-clean.csv"

df = pd.read_csv(RAW)
before = df.copy()
log = []


def step(name, rows_before):
    log.append((name, rows_before, len(df), rows_before - len(df)))


# 1. placeholder sanction amount
n = len(df)
df = df[df["PRI.SANCTIONED.AMOUNT"] < 1e9]
step("drop PRI.SANCTIONED.AMOUNT = 1e9 row", n)

# 2. century bug: pandas maps yy 00-68 to 2000-2068; borrowers born after 2000 would be under 18
n = len(df)
dob = pd.to_datetime(df["Date.of.Birth"], format="%d-%m-%y")
century_bug = dob.dt.year > 2000
dob = dob.where(~century_bug, dob - pd.DateOffset(years=100))
step(f"fix DOB century bug ({int(century_bug.sum())} values changed)", n)

# 3. keep birth year only, so the 01-Jan placeholder day/month carries no false precision
n = len(df)
placeholder = df["Date.of.Birth"].str.startswith("01-01-")
df["Date.of.Birth"] = dob.dt.year
step(f"DOB -> birth year ({int(placeholder.sum())} were 01-Jan placeholders)", n)

# 4. missing employment type as its own category
n = len(df)
n_missing = int(df["Employment.Type"].isna().sum())
df["Employment.Type"] = df["Employment.Type"].fillna("Missing")
step(f"Employment.Type NaN -> 'Missing' ({n_missing} values)", n)

# 5. constant column
n = len(df)
df = df.drop(columns="MobileNo_Avl_Flag")
step("drop MobileNo_Avl_Flag column", n)

# 6. percentile trim on disbursed_amount
n = len(df)
p1, p99 = df["disbursed_amount"].quantile([0.01, 0.99])
trimmed = df[(df["disbursed_amount"] < p1) | (df["disbursed_amount"] > p99)]
df = df[df["disbursed_amount"].between(p1, p99)]
step(f"trim disbursed_amount outside [P1={p1:,.0f}, P99={p99:,.0f}]", n)

# 7. score 0 means no bureau record and 11-18 are "not scored" reason codes, not scores
n = len(df)
desc_before = df["PERFORM_CNS.SCORE.DESCRIPTION"].copy()
score = df["PERFORM_CNS.SCORE"]
df["PERFORM_CNS.SCORE.DESCRIPTION"] = pd.Series(
    np.select([score == 0, score < 300], ["no_history", "not_scored"], "scored"), index=df.index)
df["PERFORM_CNS.SCORE"] = score.where(score >= 300)
step(f"bureau score 0/11-18 -> NaN ({int(df['PERFORM_CNS.SCORE'].isna().sum())} values)", n)

df.to_csv(OUT, index=False)

# ---------------------------------------------------------------- report
print(f"{'step':62s} {'before':>8s} {'after':>8s} {'removed':>8s}")
for name, b, a, r in log:
    print(f"{name:62s} {b:>8,d} {a:>8,d} {r:>8,d}")

print(f"\nshape: {before.shape} -> {df.shape}")
print("missing values:", before.isna().sum()[lambda s: s > 0].to_dict(), "->", df.isna().sum()[lambda s: s > 0].to_dict())
print("  (PERFORM_CNS.SCORE NaN = no score exists for that borrower, by design)")
print("default rate:", round(before.loan_default.mean(), 4), "->", round(df.loan_default.mean(), 4))

lo, hi = trimmed[trimmed.disbursed_amount < p1], trimmed[trimmed.disbursed_amount > p99]
print(f"\ntrimmed below P1: {len(lo):,} rows, default rate {lo.loan_default.mean():.4f}")
print(f"trimmed above P99: {len(hi):,} rows, default rate {hi.loan_default.mean():.4f}")
spread = p99 - p1
print(f"(fences at P1 - 1.5*(P99-P1) and P99 + 1.5*(P99-P1) would remove "
      f"{int(((before.disbursed_amount < p1 - 1.5 * spread) | (before.disbursed_amount > p99 + 1.5 * spread)).sum())} rows)")

age = 2018 - df["Date.of.Birth"]
print("\napprox. age at disbursal (2018 - birth year):", age.describe().round(1).to_dict())
print("disbursed_amount:", before.disbursed_amount.describe().round(0).to_dict())
print("              -> ", df.disbursed_amount.describe().round(0).to_dict())
print("Employment.Type:", df["Employment.Type"].value_counts().to_dict())

status = df["PERFORM_CNS.SCORE.DESCRIPTION"]
print(f"\nCNS description levels: {desc_before.nunique()} -> {status.nunique()}")
print(df.groupby(status).agg(n=("loan_default", "size"), default_rate=("loan_default", "mean"),
                             score_min=("PERFORM_CNS.SCORE", "min"), score_max=("PERFORM_CNS.SCORE", "max")).round(4))
scored = df[status == "scored"]
print("corr(score, default), scored only:", round(scored["PERFORM_CNS.SCORE"].corr(scored.loan_default), 4))

dd = pd.to_datetime(df["DisbursalDate"], format="%d-%m-%y")
last5 = dd.dt.day > dd.dt.days_in_month - 5
dd0 = pd.to_datetime(before["DisbursalDate"], format="%d-%m-%y")
last5_0 = dd0.dt.day > dd0.dt.days_in_month - 5
print("\nmonth-end / rest counts:", last5_0.value_counts().to_dict(), "->", last5.value_counts().to_dict())
print("month-end / rest default rate:", before.groupby(last5_0).loan_default.mean().round(4).to_dict(),
      "->", df.groupby(last5).loan_default.mean().round(4).to_dict())
