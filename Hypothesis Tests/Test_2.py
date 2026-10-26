"""Test whether month-end loans have a higher share of low bureau scores."""

from math import erf, sqrt
from pathlib import Path

import pandas as pd


CSV_PATH = Path(__file__).with_name("confirmation.csv")
DATE_COL = "DisbursalDate"
SCORE_COL = "PERFORM_CNS.SCORE"
LOW_SCORE_CUTOFF = 600  # Low score is defined as a genuine score below 600.
ALPHA = 0.05
ONE_SIDED_CRITICAL_Z = 1.6448536269514722


def normal_cdf(value: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def main() -> None:
    df = pd.read_csv(CSV_PATH, usecols=[DATE_COL, SCORE_COL])
    dates = pd.to_datetime(df[DATE_COL], format="%d-%m-%y", errors="coerce")
    df["is_month_end"] = (dates.dt.days_in_month - dates.dt.day) < 5
    df = df.dropna(subset=["is_month_end", SCORE_COL])

    # Score 0 encodes no bureau record in the raw dataset, not a genuinely
    # low score. Exclude it and compare only borrowers with a genuine score.
    df = df[df[SCORE_COL] >= 300].copy()
    df["low_score"] = (df[SCORE_COL] < LOW_SCORE_CUTOFF).astype(int)

    month_end = df[df["is_month_end"]]["low_score"]
    other = df[~df["is_month_end"]]["low_score"]
    n1, n2 = len(month_end), len(other)
    if not n1 or not n2:
        raise ValueError("Both groups must contain borrowers with genuine scores.")

    x1, x2 = int(month_end.sum()), int(other.sum())
    p1, p2 = x1 / n1, x2 / n2
    difference = p1 - p2

    # H0: p1 <= p2; H1: p1 > p2. Pool under the null boundary.
    pooled = (x1 + x2) / (n1 + n2)
    pooled_se = sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z_stat = difference / pooled_se
    p_value = 1 - normal_cdf(z_stat)

    # 95% confidence interval for the difference, using the unpooled SE.
    ci_se = sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ci_low = difference - 1.96 * ci_se
    ci_high = difference + 1.96 * ci_se

    print("Month-end low-credit-score proportion test")
    print(f"Data: {CSV_PATH.name}")
    print(f"Definition: genuine bureau score < {LOW_SCORE_CUTOFF}; unscored borrowers excluded")
    print(f"Month-end: {x1:,}/{n1:,} = {p1:.4%}")
    print(f"Other days: {x2:,}/{n2:,} = {p2:.4%}")
    print(f"Difference (month-end - other): {difference:.4%} ({difference * 100:.2f} percentage points)")
    print(f"95% CI for difference: [{ci_low:.4%}, {ci_high:.4%}]")
    print(f"One-sided pooled z statistic: {z_stat:.4f}")
    print(f"Critical z at alpha = {ALPHA:.2f}: {ONE_SIDED_CRITICAL_Z:.3f}")
    print(f"One-sided p-value: {p_value:.6g}")
    print("Decision by critical value:",
          "Reject H0" if z_stat > ONE_SIDED_CRITICAL_Z else "Fail to reject H0")


if __name__ == "__main__":
    main()
