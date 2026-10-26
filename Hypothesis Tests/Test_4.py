"""Run the planned one-sided two-proportion z-test for hypothesis T8."""

from math import erf, sqrt
from pathlib import Path

import pandas as pd


CSV_PATH = Path(__file__).with_name("confirmation.csv")
MONTH_END_COL = "is_last5"
NO_HISTORY_COL = "PERFORM_CNS.SCORE.DESCRIPTION_no_history"
ALPHA = 0.05
PRACTICAL_GAP = 0.02  # 2 percentage points


def normal_cdf(value: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def main() -> None:
    df = pd.read_csv(CSV_PATH, usecols=[MONTH_END_COL, NO_HISTORY_COL])
    df = df.dropna(subset=[MONTH_END_COL, NO_HISTORY_COL])

    # Ensure the flags represent the expected binary indicators.
    for column in (MONTH_END_COL, NO_HISTORY_COL):
        values = set(df[column].unique())
        if not values.issubset({0, 1, 0.0, 1.0, False, True}):
            raise ValueError(f"{column} must contain only 0/1 values; found {values}")

    month_end = df[df[MONTH_END_COL] == 1][NO_HISTORY_COL]
    other = df[df[MONTH_END_COL] == 0][NO_HISTORY_COL]
    n1, n2 = len(month_end), len(other)
    if not n1 or not n2:
        raise ValueError("Both month-end and other-loan groups must be non-empty.")

    x1, x2 = int(month_end.sum()), int(other.sum())
    p1, p2 = x1 / n1, x2 / n2
    difference = p1 - p2

    # H0: p1 <= p2; H1: p1 > p2. Under the boundary of H0, use the
    # pooled estimate in the standard error.
    pooled = (x1 + x2) / (n1 + n2)
    pooled_se = sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z_stat = difference / pooled_se
    p_value = 1 - normal_cdf(z_stat)

    # Two-sided 95% Wald confidence interval for the difference, using
    # the unpooled standard error as specified in the project plan.
    ci_se = sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ci_low = difference - 1.96 * ci_se
    ci_high = difference + 1.96 * ci_se

    print("T8: No-credit-history share among month-end vs other loans")
    print(f"Data: {CSV_PATH.name}")
    print(f"Month-end loans: {x1:,}/{n1:,} = {p1:.4%}")
    print(f"Other loans:     {x2:,}/{n2:,} = {p2:.4%}")
    print(f"Difference (month-end - other): {difference:.4%} ({difference * 100:.2f} percentage points)")
    print(f"95% CI for difference: [{ci_low:.4%}, {ci_high:.4%}]")
    print(f"One-sided pooled z statistic: {z_stat:.4f}")
    print(f"One-sided p-value: {p_value:.6g}")
    print(f"Significance level: alpha = {ALPHA:.2f}")
    print("Decision:", "Reject H0" if p_value < ALPHA else "Fail to reject H0")
    print(f"Practical threshold (>= {PRACTICAL_GAP * 100:.1f} pp):",
          "Met" if difference >= PRACTICAL_GAP else "Not met")


if __name__ == "__main__":
    main()
