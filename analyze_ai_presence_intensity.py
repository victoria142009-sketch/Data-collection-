"""
Actual analysis: AI Presence and AI Intensity.

In plain words:
1. Open the cleaned combined job dataset
2. Make sure every job is scored for AI words
3. Calculate AI Presence (yes/no) and AI Intensity (your formula)
4. Compare Entry vs Senior
5. Also show results by marketing subdiscipline
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from score_genai_requirements import score_requirements

DATA_PATH = Path(__file__).resolve().parent / "analysis_sample_combined.csv"
OUT_SCORED = Path(__file__).resolve().parent / "analysis_results_scored.csv"
OUT_SUMMARY = Path(__file__).resolve().parent / "analysis_results_summary.csv"


def refresh_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Re-calculate AI Presence and AI Intensity with the current dictionary/formula."""
    scored = df["requirements_text"].fillna("").astype(str).map(score_requirements)
    score_df = pd.DataFrame(scored.tolist(), index=df.index)
    for col in score_df.columns:
        df[col] = score_df[col]
    return df


def seniority_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("Seniority")
        .agg(
            n_jobs=("ai_presence", "size"),
            n_with_ai=("ai_presence", "sum"),
            ai_presence_pct=("ai_presence", lambda s: 100 * s.mean()),
            mean_ai_intensity=("ai_intensity", "mean"),
            median_ai_intensity=("ai_intensity", "median"),
            mean_intensity_if_ai_present=(
                "ai_intensity",
                lambda s: s[df.loc[s.index, "ai_presence"] == 1].mean()
                if (df.loc[s.index, "ai_presence"] == 1).any()
                else np.nan,
            ),
        )
        .round(3)
        .reset_index()
    )
    return summary


def subdiscipline_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["Subdiscipline", "Seniority"])
        .agg(
            n_jobs=("ai_presence", "size"),
            ai_presence_pct=("ai_presence", lambda s: 100 * s.mean()),
            mean_ai_intensity=("ai_intensity", "mean"),
        )
        .round(3)
        .reset_index()
    )
    return summary


def run_tests(df: pd.DataFrame) -> None:
    entry = df[df["Seniority"] == "Entry"]
    senior = df[df["Seniority"] == "Senior"]

    # AI Presence: 2x2 table Entry/Senior x AI yes/no
    table = pd.crosstab(df["Seniority"], df["ai_presence"])
    chi2, p_chi, dof, expected = stats.chi2_contingency(table)
    # Cramer's V
    n = table.to_numpy().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1))) if n else np.nan

    # If any expected count < 5, also report Fisher
    fisher_p = None
    if (expected < 5).any():
        # Fisher needs 2x2 with numeric order
        oddsratio, fisher_p = stats.fisher_exact(table.loc[["Entry", "Senior"], [0, 1]])

    # AI Intensity: nonparametric test (data is skewed, many zeros)
    u_stat, p_u = stats.mannwhitneyu(
        entry["ai_intensity"],
        senior["ai_intensity"],
        alternative="two-sided",
    )

    print("=" * 60)
    print("WHAT THE NUMBERS MEAN")
    print("=" * 60)
    print("AI Presence = 1 if at least one AI/GenAI word appears in requirements, else 0")
    print("AI Intensity = (number of GenAI keywords / total words in requirements) * 1000")
    print("               <- this is exactly your paper formula")
    print()

    print("=" * 60)
    print("ENTRY vs SENIOR — AI PRESENCE")
    print("=" * 60)
    print(table.to_string())
    print()
    print(f"Chi-square = {chi2:.3f}, p = {p_chi:.4f}, Cramer's V = {cramers_v:.3f}")
    if fisher_p is not None:
        print(f"Fisher exact p = {fisher_p:.4f} (used because some expected counts < 5)")
    print()
    if p_chi < 0.05 or (fisher_p is not None and fisher_p < 0.05):
        print("Plain language: the difference in AI Presence between Entry and Senior")
        print("is statistically significant (p < 0.05).")
    else:
        print("Plain language: the difference in AI Presence between Entry and Senior")
        print("is NOT statistically significant (p >= 0.05).")
    print()

    print("=" * 60)
    print("ENTRY vs SENIOR — AI INTENSITY")
    print("=" * 60)
    print(f"Entry mean intensity  = {entry['ai_intensity'].mean():.3f}")
    print(f"Senior mean intensity = {senior['ai_intensity'].mean():.3f}")
    print(f"Mann-Whitney U = {u_stat:.1f}, p = {p_u:.4f}")
    print()
    if p_u < 0.05:
        print("Plain language: Entry and Senior differ significantly in AI Intensity.")
    else:
        print("Plain language: Entry and Senior do NOT differ significantly in AI Intensity.")


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df = refresh_scores(df)

    print("=" * 60)
    print("SAMPLE USED FOR ANALYSIS")
    print("=" * 60)
    print(f"Total jobs: {len(df)}")
    print(df["Seniority"].value_counts().to_string())
    print()
    print("Jobs with at least one AI/GenAI keyword:", int(df["ai_presence"].sum()),
          f"({100 * df['ai_presence'].mean():.1f}%)")
    print()

    summary = seniority_summary(df)
    print("=" * 60)
    print("SUMMARY BY SENIORITY")
    print("=" * 60)
    print(summary.to_string(index=False))
    print()

    sub = subdiscipline_summary(df)
    print("=" * 60)
    print("SUMMARY BY SUBDISCIPLINE x SENIORITY")
    print("=" * 60)
    print(sub.to_string(index=False))
    print()

    run_tests(df)

    print()
    print("Top AI-present jobs (highest intensity):")
    cols = ["Seniority", "Subdiscipline", "title", "companyName", "genai_matches", "ai_intensity"]
    top = df.loc[df["ai_presence"] == 1, cols].sort_values("ai_intensity", ascending=False).head(10)
    print(top.to_string(index=False))

    df.to_csv(OUT_SCORED, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    sub.to_csv(Path(__file__).resolve().parent / "analysis_results_by_subdiscipline.csv", index=False)
    print(f"\nSaved: {OUT_SCORED}")
    print(f"Saved: {OUT_SUMMARY}")
    print("Saved: analysis_results_by_subdiscipline.csv")
