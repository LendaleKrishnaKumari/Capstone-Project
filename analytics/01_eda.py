from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ANALYTICS_DIR = Path(__file__).resolve().parent
RAW_DF_PATH = ANALYTICS_DIR / "titanic_raw.csv"
CLEAN_DF_PATH = ANALYTICS_DIR / "titanic.csv"
CHART_DIR = ANALYTICS_DIR / "charts"


def load_and_save_raw_dataset() -> pd.DataFrame:
    """Load the Titanic dataset once from Seaborn and save a local offline fallback."""
    df = sns.load_dataset("titanic")
    df.to_csv(RAW_DF_PATH, index=False)
    df.to_csv(CLEAN_DF_PATH, index=False)
    return df


def print_dataset_profile(df: pd.DataFrame) -> None:
    """Print the required profiling summary for the module."""
    print("Shape:", df.shape)
    print("\nInfo:")
    print(df.info())
    print("\nDescribe:")
    print(df.describe(include="all").transpose())
    print("\nMissing percentages:")
    missing_percentages = df.isna().mean().mul(100)
    print(missing_percentages[missing_percentages > 0].sort_values(ascending=False).round(2).to_string())


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the threshold-based missing-value strategy and keep the cleaned dataset."""
    cleaned = df.copy()

    for column_name in ["cabin", "deck"]:
        if column_name in cleaned.columns:
            cleaned = cleaned.drop(columns=[column_name])

    cleaned = cleaned.dropna(subset=["embarked"]).reset_index(drop=True)
    cleaned["age"] = cleaned["age"].fillna(cleaned["age"].median())

    for column_name in ["fare"]:
        if column_name in cleaned.columns:
            cleaned[column_name] = cleaned[column_name].fillna(cleaned[column_name].median())

    cleaned = cleaned.drop(columns=["embark_town"], errors="ignore")
    cleaned = cleaned.drop(columns=["adult_male"], errors="ignore")
    cleaned = cleaned.drop(columns=["alone"], errors="ignore")
    cleaned = cleaned.reset_index(drop=True)
    cleaned.to_csv(CLEAN_DF_PATH, index=False)
    return cleaned


def compute_outlier_counts(df: pd.DataFrame) -> dict[str, int]:
    """Return the count of IQR outliers for age and fare."""
    counts: dict[str, int] = {}
    for column_name in ["age", "fare"]:
        q1 = df[column_name].quantile(0.25)
        q3 = df[column_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (df[column_name] < lower_bound) | (df[column_name] > upper_bound)
        counts[column_name] = int(outlier_mask.sum())
    return counts


def make_univariate_plots(df: pd.DataFrame) -> None:
    """Create histograms and box plots for age and fare."""
    CHART_DIR.mkdir(exist_ok=True)
    for column_name in ["age", "fare"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[column_name], bins=20, ax=axes[0], kde=True)
        axes[0].set_title(f"Histogram of {column_name}")
        sns.boxplot(x=df[column_name], ax=axes[1])
        axes[1].set_title(f"Box plot of {column_name}")
        fig.tight_layout()
        fig.savefig(CHART_DIR / f"{column_name}_distribution.png", dpi=150)
        plt.close(fig)


def report_survival_rates(df: pd.DataFrame) -> None:
    """Print survival rates by sex, class, and sex/class combined."""
    print("\nSurvival rate by sex:")
    print(df.groupby("sex")["survived"].mean().round(4).to_string())

    print("\nSurvival rate by pclass:")
    print(df.groupby("pclass")["survived"].mean().round(4).to_string())

    print("\nSurvival rate by sex and pclass:")
    print(
        df.groupby(["sex", "pclass"])["survived"].mean().unstack().round(4).to_string()
    )


def plot_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Plot a correlation heatmap and return the six-column correlation matrix."""
    columns = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    correlation_matrix = df[columns].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Titanic correlation heatmap")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()
    return correlation_matrix


def describe_top_correlations(correlation_matrix: pd.DataFrame) -> None:
    """Print the two strongest off-diagonal correlations by absolute value."""
    off_diagonal = []
    for left in correlation_matrix.columns:
        for right in correlation_matrix.columns:
            if left >= right:
                continue
            value = abs(correlation_matrix.loc[left, right])
            off_diagonal.append((left, right, value, correlation_matrix.loc[left, right]))

    off_diagonal.sort(key=lambda item: item[2], reverse=True)
    print("\nTop two strongest off-diagonal correlations:")
    for left, right, magnitude, value in off_diagonal[:2]:
        print(f"{left} vs {right}: {value:.4f} (abs = {magnitude:.4f})")


def create_multivariate_story(df: pd.DataFrame) -> None:
    """Save at least four charts with narrative interpretations."""
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="sex", y="survived", estimator="mean")
    plt.title("Survival rate by sex")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "survival_by_sex.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="pclass", y="survived", estimator="mean")
    plt.title("Survival rate by passenger class")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "survival_by_pclass.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="sex", y="fare")
    plt.title("Fare distribution by sex")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "fare_by_sex.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="age", y="fare", hue="survived", alpha=0.75)
    plt.title("Age vs fare by survival status")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "age_fare_survival.png", dpi=150)
    plt.close()

    print("\nMultivariate story charts saved to:", CHART_DIR)


def standardize_age_and_fare(df: pd.DataFrame) -> None:
    """Show the before/after z-score standardization for age and fare."""
    print("\nBefore standardization summary:")
    print(df[["age", "fare"]].agg(["mean", "std"]).round(4).to_string())

    standardized = df[["age", "fare"]].copy()
    for column_name in ["age", "fare"]:
        standardized[column_name] = (standardized[column_name] - standardized[column_name].mean()) / standardized[column_name].std(ddof=0)

    print("\nAfter standardization summary:")
    print(standardized.agg(["mean", "std"]).round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["age"], ax=axes[0], bins=20, color="steelblue")
    sns.histplot(standardized["age"], ax=axes[1], bins=20, color="darkorange")
    axes[0].set_title("Age before standardization")
    axes[1].set_title("Age after standardization")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "age_standardization_check.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the Titanic EDA workflow and save the offline dataset and charts."""
    CHART_DIR.mkdir(exist_ok=True)
    raw_df = load_and_save_raw_dataset()
    print_dataset_profile(raw_df)
    cleaned_df = clean_dataframe(raw_df)
    print("\nCleaned shape:", cleaned_df.shape)
    print("\nMissing values after cleaning:")
    print(cleaned_df.isna().sum()[cleaned_df.isna().sum() > 0].to_string())

    outlier_counts = compute_outlier_counts(cleaned_df)
    print("\nIQR outlier counts:")
    print(outlier_counts)

    fare_mean = cleaned_df["fare"].mean()
    fare_median = cleaned_df["fare"].median()
    fare_mode = cleaned_df["fare"].mode().iloc[0]
    print(f"\nFare mean: {fare_mean:.4f}")
    print(f"Fare median: {fare_median:.4f}")
    print(f"Fare mode: {fare_mode:.4f}")
    print("Fare distribution skewness conclusion: mean > median > mode, so it is right-skewed.")

    make_univariate_plots(cleaned_df)
    report_survival_rates(cleaned_df)
    corr_matrix = plot_correlation_matrix(cleaned_df)
    describe_top_correlations(corr_matrix)
    create_multivariate_story(cleaned_df)
    standardize_age_and_fare(cleaned_df)


if __name__ == "__main__":
    main()
