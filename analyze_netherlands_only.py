"""Netherlands-only AI Presence and AI Intensity analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from score_genai_requirements import score_requirements

DATA_PATH = Path(__file__).resolve().parent / "analysis_sample_combined.csv"
OUT_DATA = Path(__file__).resolve().parent / "analysis_results_netherlands_only.csv"
OUT_SUMMARY = Path(__file__).resolve().parent / "analysis_summary_netherlands_only.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # Keep original NL LinkedIn + Indeed; drop UK LinkedIn_extra
    nl = df[df["source_platform"].isin(["LinkedIn_NL", "Indeed"])].copy()
    nl = nl[
        ~nl["location"]
        .fillna("")
        .astype(str)
        .str.contains(r"United Kingdom|England|London,", case=False, regex=True)
    ].copy()

    scored = nl["requirements_text"].fillna("").astype(str).map(score_requirements)
    score_df = pd.DataFrame(scored.tolist(), index=nl.index)
    for col in score_df.columns:
        nl[col] = score_df[col]

    print("=" * 60)
    print("NETHERLANDS-ONLY SAMPLE")
    print("=" * 60)
    print(f"Total jobs: {len(nl)}")
    print(nl["source_platform"].value_counts().to_string())
    print()
    print(nl["Seniority"].value_counts().to_string())
    print()
    pct = 100 * nl["ai_presence"].mean()
    print(f"Jobs with AI keyword: {int(nl['ai_presence'].sum())} ({pct:.1f}%)")
    print()

    summary = (
        nl.groupby("Seniority")
        .agg(
            n_jobs=("ai_presence", "size"),
            n_with_ai=("ai_presence", "sum"),
            ai_presence_pct=("ai_presence", lambda s: 100 * s.mean()),
            mean_ai_intensity=("ai_intensity", "mean"),
            median_ai_intensity=("ai_intensity", "median"),
        )
        .round(3)
    )
    print("=" * 60)
    print("ENTRY vs SENIOR")
    print("=" * 60)
    print(summary.to_string())
    print()

    sub = (
        nl.groupby(["Subdiscipline", "Seniority"])
        .agg(
            n_jobs=("ai_presence", "size"),
            ai_presence_pct=("ai_presence", lambda s: 100 * s.mean()),
            mean_ai_intensity=("ai_intensity", "mean"),
        )
        .round(3)
    )
    print("=" * 60)
    print("BY SUBDISCIPLINE")
    print("=" * 60)
    print(sub.to_string())
    print()

    table = pd.crosstab(nl["Seniority"], nl["ai_presence"])
    print("Presence table:")
    print(table.to_string())
    chi2, p_chi, _, expected = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    print(f"Chi-square = {chi2:.3f}, p = {p_chi:.4f}, Cramer's V = {cramers_v:.3f}")
    if (expected < 5).any():
        t = table.loc[["Entry", "Senior"], [0, 1]]
        oddsratio, fisher_p = stats.fisher_exact(t)
        print(f"Fisher p = {fisher_p:.4f}, odds ratio = {oddsratio:.3f}")

    entry = nl.loc[nl["Seniority"] == "Entry", "ai_intensity"]
    senior = nl.loc[nl["Seniority"] == "Senior", "ai_intensity"]
    u_stat, p_u = stats.mannwhitneyu(entry, senior, alternative="two-sided")
    print()
    print(f"Entry mean intensity  = {entry.mean():.3f}")
    print(f"Senior mean intensity = {senior.mean():.3f}")
    print(f"Mann-Whitney U = {u_stat:.1f}, p = {p_u:.4f}")
    print()

    if p_chi < 0.05:
        print("Presence: significant difference Entry vs Senior.")
    else:
        print("Presence: NO significant difference Entry vs Senior.")

    if p_u < 0.05:
        print("Intensity: significant difference Entry vs Senior.")
    else:
        print("Intensity: NO significant difference Entry vs Senior.")

    nl.to_csv(OUT_DATA, index=False)
    summary.to_csv(OUT_SUMMARY)
    print(f"\nSaved: {OUT_DATA}")
    print(f"Saved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
