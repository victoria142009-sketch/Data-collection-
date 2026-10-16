"""
H2 test: does the Entry vs Senior AI difference depend on Subdiscipline?

Model:
  ai_presence ~ Seniority * Subdiscipline

In plain words:
- If the interaction is significant, Entry vs Senior is NOT the same
  across Brand / Content / Digital.
- If not significant, we cannot confirm that the pattern truly differs by area.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA_PATH = Path(__file__).resolve().parent / "analysis_sample_combined.csv"
OUT_PATH = Path(__file__).resolve().parent / "analysis_h2_interaction_results.txt"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # Keep the three main subdisciplines for a clean H2 test
    data = df[df["Subdiscipline"].isin(["Brand", "Content & Creative", "Digital & Performance"])].copy()
    data["ai_presence"] = data["ai_presence"].astype(int)
    data["Seniority"] = pd.Categorical(data["Seniority"], categories=["Senior", "Entry"])
    data["Subdiscipline"] = pd.Categorical(
        data["Subdiscipline"],
        categories=["Brand", "Content & Creative", "Digital & Performance"],
    )

    print("=" * 60)
    print("SAMPLE FOR H2")
    print("=" * 60)
    print(f"N = {len(data)}")
    print(pd.crosstab(data["Subdiscipline"], data["Seniority"]).to_string())
    print()
    print("AI Presence % by cell:")
    print(
        data.groupby(["Subdiscipline", "Seniority"], observed=True)["ai_presence"]
        .mean()
        .mul(100)
        .round(1)
        .unstack()
        .to_string()
    )
    print()

    # Model 1: seniority only
    m1 = smf.logit("ai_presence ~ C(Seniority)", data=data).fit(disp=False)
    # Model 2: seniority + subdiscipline
    m2 = smf.logit("ai_presence ~ C(Seniority) + C(Subdiscipline)", data=data).fit(disp=False)
    # Model 3: interaction
    m3 = smf.logit("ai_presence ~ C(Seniority) * C(Subdiscipline)", data=data).fit(disp=False)

    # Likelihood-ratio test for interaction: model 3 vs model 2
    lr_stat = 2 * (m3.llf - m2.llf)
    df_diff = m3.df_model - m2.df_model
    from scipy import stats

    p_interaction = stats.chi2.sf(lr_stat, df_diff)

    print("=" * 60)
    print("LOGISTIC REGRESSION (main effects, no interaction)")
    print("=" * 60)
    print(m2.summary().tables[1].as_text())
    print()
    ors = np.exp(m2.params)
    ci = np.exp(m2.conf_int())
    ci.columns = ["OR_low", "OR_high"]
    or_table = pd.DataFrame({"OR": ors, "OR_low": ci["OR_low"], "OR_high": ci["OR_high"], "p": m2.pvalues})
    print("Odds Ratios:")
    print(or_table.round(3).to_string())
    print()

    print("=" * 60)
    print("INTERACTION MODEL (Seniority x Subdiscipline)")
    print("=" * 60)
    print(m3.summary().tables[1].as_text())
    print()
    print(f"Likelihood-ratio test for interaction: chi2({df_diff}) = {lr_stat:.3f}, p = {p_interaction:.4f}")
    print()

    if p_interaction < 0.05:
        print("PLAIN CONCLUSION FOR H2:")
        print("SUPPORTED: the Entry vs Senior difference in AI Presence")
        print("significantly depends on marketing subdiscipline.")
    else:
        print("PLAIN CONCLUSION FOR H2:")
        print("NOT SUPPORTED: we cannot confirm that the Entry vs Senior")
        print("difference significantly depends on marketing subdiscipline.")
        print("(Cell patterns can still be described as exploratory.)")

    # Predicted probabilities by cell from interaction model
    print()
    print("Predicted AI Presence probability by cell (interaction model):")
    grid = pd.DataFrame(
        [
            {"Seniority": s, "Subdiscipline": d}
            for d in ["Brand", "Content & Creative", "Digital & Performance"]
            for s in ["Senior", "Entry"]
        ]
    )
    grid["pred_prob"] = m3.predict(grid)
    print(grid.assign(pred_pct=lambda x: (100 * x["pred_prob"]).round(1)).to_string(index=False))

    # Also report Entry OR within each subdiscipline using stratified logits if possible
    print()
    print("Entry vs Senior Odds Ratio within each subdiscipline:")
    rows = []
    for disc in ["Brand", "Content & Creative", "Digital & Performance"]:
        part = data[data["Subdiscipline"] == disc]
        # need both outcomes present
        if part["ai_presence"].nunique() < 2 or part["Seniority"].nunique() < 2:
            rows.append((disc, np.nan, np.nan, np.nan, np.nan, "skipped"))
            continue
        try:
            m = smf.logit("ai_presence ~ C(Seniority)", data=part).fit(disp=False)
            # Entry coefficient (Senior is reference)
            term = [t for t in m.params.index if "Entry" in t][0]
            or_ = float(np.exp(m.params[term]))
            lo, hi = np.exp(m.conf_int().loc[term])
            p = float(m.pvalues[term])
            rows.append((disc, or_, lo, hi, p, "ok"))
        except Exception as exc:  # noqa: BLE001
            rows.append((disc, np.nan, np.nan, np.nan, np.nan, str(exc)))

    strat = pd.DataFrame(rows, columns=["Subdiscipline", "OR_Entry_vs_Senior", "OR_low", "OR_high", "p", "note"])
    print(strat.round(3).to_string(index=False))

    text = []
    text.append(f"N={len(data)}")
    text.append(f"Interaction LR p={p_interaction:.4f}")
    text.append(strat.to_string(index=False))
    OUT_PATH.write_text("\n".join(text), encoding="utf-8")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
