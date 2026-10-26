import sys
import numpy as np
import pandas as pd
from scipy import stats

# Data: month-end = last 5 calendar days of the month
df = pd.read_csv(sys.argv[1], usecols=["DisbursalDate", "loan_default"])
date = pd.to_datetime(df["DisbursalDate"], format="%d-%m-%y")
is_last5 = date.dt.day > date.dt.days_in_month - 5
y = df["loan_default"]

x1, n1 = y[is_last5].sum(), is_last5.sum()      # month-end
x2, n2 = y[~is_last5].sum(), (~is_last5).sum()  # rest
p1, p2 = x1 / n1, x2 / n2
diff = p1 - p2

# z-test (pooled SE under H0), one-sided right
p_pool = (x1 + x2) / (n1 + n2)
z = diff / np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
p_value = stats.norm.sf(z)

# 95% CI (unpooled SE)
se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
lo, hi = diff - 1.96 * se, diff + 1.96 * se


print(f"Estimate        : {diff*100:+.2f} pp ({p1:.2%} month-end, {p2:.2%} rest)")
print(f"95% CI          : {lo*100:+.2f} pp to {hi*100:+.2f} pp")
print(f"Test statistic  : Z = {z:.2f} (no d.f., standard normal)")
print(f"p-value         : {p_value:.1e}")
print(f"Decision        : {'reject H0' if p_value < 0.05 else 'fail to reject H0'}")