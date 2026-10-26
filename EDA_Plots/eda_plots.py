"""
06_eda_plots.py - EDA figures F1-F5 on the exploration set.

Input : dataset-train-LT-encoded.csv
Output: eda_plots/F1 ... F5 (.png)

Rebuilds the same exploration/confirmation split as 04_eda.py
(30/70, stratified on loan_default x is_last5, seed 42) in memory.
It does not overwrite splits/.

Run: python3 06_eda_plots.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split

SEED = 42
EXPLORE_FRAC = 0.30
IN_FILE = "dataset-train-LT-encoded.csv"
OUT_DIR = "eda_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- colours and style ----
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False,
})

def save(fig, name):
    fig.savefig(f"{OUT_DIR}/{name}.png")
    plt.close(fig)
    print(f"saved {OUT_DIR}/{name}.png")

def label_bars(ax, bars, values, fmt="{:.1f}%", pad=0.4):
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + pad, fmt.format(v),
                ha="center", fontsize=8, color=INK)

# ---------------------------------------------------------------- load + split
df = pd.read_csv(IN_FILE, low_memory=False)
d = pd.to_datetime(df["DisbursalDate"], format="%d-%m-%y")
df["dte"] = d.dt.days_in_month - d.dt.day          # days before month end (0 = last day)
df["is_last5"] = (df["dte"] < 5).astype(int)

strata = df["loan_default"].astype(str) + "_" + df["is_last5"].astype(str)
e, _ = train_test_split(df, train_size=EXPLORE_FRAC, stratify=strata, random_state=SEED)
e = e.copy()
overall = 100 * e["loan_default"].mean()
print(f"Exploration set: {len(e):,} loans (seed {SEED}), default rate {overall:.2f}%")

# ---------------------------------------------------------------- F1 credit score
def score_band(r):
    if r["PERFORM_CNS.SCORE.DESCRIPTION_no_history"] == 1:
        return "No history"
    if r["PERFORM_CNS.SCORE.DESCRIPTION_not_scored"] == 1:
        return "Not scored"
    s = r["PERFORM_CNS.SCORE"]
    for lo, hi in [(300, 500), (500, 600), (600, 700), (700, 800)]:
        if lo <= s < hi:
            return f"{lo}-{hi - 1}"
    return "800-899"

e["score_band"] = e.apply(score_band, axis=1)
order = ["No history", "Not scored", "300-499", "500-599", "600-699", "700-799", "800-899"]
g = e.groupby("score_band")["loan_default"].mean().reindex(order) * 100

fig, ax = plt.subplots(figsize=(6.4, 3.2))
bars = ax.bar([o.replace(" ", "\n") for o in order], g, width=0.65,
              color=[INK2, INK2] + [BLUE] * 5)
label_bars(ax, bars, g)
ax.axhline(overall, color=INK2, lw=1, ls="--")
ax.text(6.45, overall + 0.4, "overall", fontsize=8, color=INK2, ha="right")
ax.set_ylabel("Default rate (%)")
ax.set_xlabel("Bureau score band (grey = no score exists)")
ax.set_ylim(0, g.max() * 1.18)
save(fig, "F1_default_by_score")

# ---------------------------------------------------------------- F2 LTV
bins = [0, 60, 65, 70, 75, 80, 85, 90, 95.01]
labels = ["<60", "60-65", "65-70", "70-75", "75-80", "80-85", "85-90", "90-95"]
e["ltv_band"] = pd.cut(e["ltv"], bins=bins, labels=labels, right=False)
g = e.groupby("ltv_band", observed=False)["loan_default"].mean() * 100

fig, ax = plt.subplots(figsize=(6.4, 3.2))
bars = ax.bar(labels, g, color=BLUE, width=0.65)
label_bars(ax, bars, g)
ax.set_ylabel("Default rate (%)")
ax.set_xlabel("LTV (%)")
ax.set_ylim(0, g.max() * 1.18)
save(fig, "F2_default_by_ltv")

# ---------------------------------------------------------------- F3 borrower drivers
e["employment"] = np.select(
    [e["Employment.Type_Salaried"] == 1, e["Employment.Type_Self_employed"] == 1],
    ["Salaried", "Self-emp."], "Missing")

def cap(s, k):
    return s.clip(upper=k).astype(int).astype(str).replace({str(k): f"{k}+"})

e["overdue_c"] = cap(e["PRI.OVERDUE.ACCTS"], 3)
e["delinq_c"] = cap(e["DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS"], 3)
e["inq_c"] = cap(e["NO.OF_INQUIRIES"], 3)
panels = [("employment", "Employment type", ["Salaried", "Self-emp.", "Missing"]),
          ("overdue_c", "Primary accounts overdue", ["0", "1", "2", "3+"]),
          ("delinq_c", "Delinquent accts, last 6 months", ["0", "1", "2", "3+"]),
          ("inq_c", "Credit inquiries", ["0", "1", "2", "3+"])]

fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.8), sharey=True)
for ax, (col, title, lv) in zip(axes.ravel(), panels):
    g = e.groupby(col)["loan_default"].mean().reindex(lv) * 100
    bars = ax.bar(lv, g, color=BLUE, width=0.6)
    label_bars(ax, bars, g, fmt="{:.0f}%", pad=0.6)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
for ax in axes[:, 0]:
    ax.set_ylabel("Default rate (%)")
axes[0, 0].set_ylim(0, 48)
fig.tight_layout()
save(fig, "F3_default_by_borrower")

# ---------------------------------------------------------------- F4 age
e["age"] = 2018 - e["Date.of.Birth"]
e["age_band"] = pd.cut(e["age"], [17, 25, 30, 35, 40, 45, 50, 70],
                       labels=["18-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51+"])
g = e.groupby("age_band", observed=False)["loan_default"].mean() * 100

fig, ax = plt.subplots(figsize=(6.4, 3.0))
bars = ax.bar(g.index.astype(str), g, color=BLUE, width=0.65)
label_bars(ax, bars, g)
ax.set_ylabel("Default rate (%)")
ax.set_xlabel("Borrower age at disbursal (years)")
ax.set_ylim(0, g.max() * 1.18)
save(fig, "F4_default_by_age")

# ---------------------------------------------------------------- F5 loan volume
# Aligned on days before month end so the last-5 window is at the same x in every month.
g = e.groupby("dte").size()

fig, ax = plt.subplots(figsize=(6.6, 3.0))
ax.bar(g.index, g, color=[ORANGE if k < 5 else BLUE for k in g.index], width=0.8)
ax.set_ylabel("Loans disbursed")
ax.set_xlabel("Days before month end (0 = last day)")
ax.invert_xaxis()
ax.legend(handles=[Patch(color=BLUE, label="Rest of month"),
                   Patch(color=ORANGE, label="Last 5 days")], loc="upper left", fontsize=8)
save(fig, "F5_loan_volume_by_days_to_month_end")

last5, rest = g[g.index < 5], g[g.index >= 5]
print(f"Last 5 days: {last5.sum() / g.sum():.1%} of loans; "
      f"{last5.mean():.0f} loans/day vs {rest.mean():.0f} on other days")
