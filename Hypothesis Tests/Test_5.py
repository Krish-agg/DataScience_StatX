import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
MIN_R = -0.3

df = pd.read_csv("confirmation.csv", usecols=["DisbursalDate", "loan_default"])
date = pd.to_datetime(df["DisbursalDate"], format="%d-%m-%y")

# x = days before month-end (0 = last day of that month)
df["days_to_end"] = date.dt.days_in_month - date.dt.day

# Keep x = 0..29 (x = 30 exists only in 31-day months)
dropped = (df["days_to_end"] > 29).sum()
df = df[df["days_to_end"] <= 29]

# One point per day: y = default rate of loans disbursed that day
daily = (df.groupby("days_to_end")["loan_default"]
           .agg(loans="size", defaults="sum", rate="mean")
           .sort_index())

x = daily.index.to_numpy(dtype=float)
y = daily["rate"].to_numpy() * 100         
n = len(x)
dof = n - 2


dx, dy = x - x.mean(), y - y.mean()
r = (dx * dy).sum() / np.sqrt((dx**2).sum() * (dy**2).sum())

# t = r * sqrt(n-2) / sqrt(1-r^2)
t = r * np.sqrt(dof) / np.sqrt(1 - r**2)
t_crit = stats.t.ppf(ALPHA, dof)             
p = stats.t.cdf(t, dof)                       

z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
ci = np.tanh([z - 1.96 * se, z + 1.96 * se])


slope = -(dx * dy).sum() / (dx**2).sum()

print(f"Rows dropped (x = 30)      : {dropped:,}")
print(f"Loans used                 : {daily['loans'].sum():,}")
print(f"Days (n)                   : {n}")
print(f"r                          : {r:.4f}")
print(f"95% CI for rho             : [{ci[0]:.3f}, {ci[1]:.3f}]")
print(f"t (d.f. = {dof})            : {t:.3f}")
print(f"Critical t (one-sided)     : {t_crit:.3f}")
print(f"p-value (one-sided)        : {p:.3e}")
print(f"Slope                      : {slope:+.4f} pp per day closer to month-end")
print("Decision                   :", "reject H0" if t < t_crit else "fail to reject H0")
print("Meets r <= -0.3?           :", r <= MIN_R)

print("\nDaily table:")
print(daily.assign(rate=lambda d: (d.rate * 100).round(2)).to_string())