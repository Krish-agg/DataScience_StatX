import pandas as pd
from scipy import stats
import numpy as np


df = pd.read_csv("confirmation.csv")

df['DisbursalDate'] = pd.to_datetime(df['DisbursalDate'], format='%d-%m-%y')
is_last_5_days = (df['DisbursalDate'].dt.daysinmonth - df['DisbursalDate'].dt.day) < 5

month_end_ltv = df[is_last_5_days]['ltv'].dropna()
regular_ltv = df[~is_last_5_days]['ltv'].dropna()

print(f"Month-end sample size: {len(month_end_ltv)}")
print(f"Normal sample size: {len(regular_ltv)}")
print(f"Mean LTV for Month-end: {month_end_ltv.mean():.2f}")
print(f"Mean LTV for normal: {regular_ltv.mean():.2f}\n")


print("H0: Month-end loans have an equal or lower average LTV compared to regular loans, i.e.\nmu_month_end_ltv <= mu_regular\n")
print("H1: Month-end loans have a higher average LTV, i.e.\nmu_month_end_ltv > mu_regular\n")

stat, p_value = stats.ttest_ind(month_end_ltv, regular_ltv, equal_var=False, alternative='greater')
print("Using Welch's T-Test")
print(f"T-statistic value= {stat:.4f}\np value= {p_value:.4f}")


alpha = 0.05
print("Result of Hypothesis Test")
if p_value < alpha:
    print(f"Reject H0, as p_value ({p_value:.4f}) is less than alpha ({alpha})")
else:
    print(f"Fail to reject H0, as p_value ({p_value:.4f}) is more than alpha ({alpha})")

mean1 = month_end_ltv.mean()
mean2 = regular_ltv.mean()
var1 = month_end_ltv.var(ddof=1)
var2 = regular_ltv.var(ddof=1)
n1 = len(month_end_ltv)
n2 = len(regular_ltv)

se1 = var1/n1
se2 = var2/n2
df = ((se1+se2)**2)/((se1**2/(n1-1))+(se2**2/(n2-1)))

t_conf = stats.t.ppf(1-alpha/2,df)
margin_of_error = t_conf*np.sqrt(se1 + se2)
ci_start = mean1 - mean2 - margin_of_error
ci_end = mean1 - mean2 + margin_of_error

print(f"Estimate (Difference of the Means): {mean1 - mean2:.4f}")
print(f"Degrees of Freedom: {df:.4f}")
print(f"95% Confidence Interval: [{ci_start:.4f}, {ci_end:.4f}]")
