import pandas as pd

CLEAN = "dataset-train-LT-clean.csv"
OUT = "dataset-train-LT-encoded.csv"
ONE_HOT = ["PERFORM_CNS.SCORE.DESCRIPTION", "Employment.Type"]

df = pd.read_csv(CLEAN)
before_shape = df.shape

enc = df.copy()
for col in ONE_HOT:
    dummies = pd.get_dummies(enc[col], prefix=col, dtype=int)
    dummies.columns = dummies.columns.str.replace(" ", "_")
    pos = enc.columns.get_loc(col)
    enc = enc.drop(columns=col)
    for i, name in enumerate(dummies.columns):
        enc.insert(pos + i, name, dummies[name])

enc.to_csv(OUT, index=False)

# ---------------------------------------------------------------- report
print(f"shape: {before_shape} -> {enc.shape}")
for col in ONE_HOT:
    new = [c for c in enc.columns if c.startswith(col + "_")]
    counts = enc[new].sum().to_dict()
    assert (enc[new].sum(axis=1) == 1).all(), f"{col}: rows not exactly one level"
    assert all(counts[f"{col}_{lvl.replace(' ', '_')}"] == n for lvl, n in df[col].value_counts().items())
    print(f"\n{col} -> {len(new)} columns")
    for c in new:
        print(f"  {c:45s} ones={counts[c]:>7,d}  default rate={enc.loc[enc[c] == 1, 'loan_default'].mean():.4f}")
