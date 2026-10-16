"""
Step-by-step GenAI scoring for the thesis dataset.

In plain words, this script:
1. Opens the job table we already made
2. Keeps only jobs where we found a requirements section
3. Labels each job's marketing area (Brand / Content / Digital)
4. Looks for Generative AI words in the requirements text
5. Calculates AI Presence (yes/no) and AI Intensity (your formula)
6. Compares Entry vs Senior in a simple summary
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "vacancies_with_requirements.pkl"
OUT_PATH = Path(__file__).resolve().parent / "analysis_sample_scored.csv"

# ---------------------------------------------------------------------------
# Broader AI / GenAI dictionary (English + Dutch)
# Plain words: we count both specific GenAI tools (ChatGPT, Claude, ...)
# and wider AI skill language used in marketing ads (AI tools, AI-native,
# artificial intelligence, prompt skills, etc.).
# Longer phrases are listed first so they are matched cleanly.
# ---------------------------------------------------------------------------
GENAI_TERMS: list[str] = [
    # Specific GenAI tools
    "chatgpt",
    "chat gpt",
    "gpt-4",
    "gpt-5",
    "gpt4",
    "claude",
    "midjourney",
    "dall-e",
    "dalle",
    "microsoft copilot",
    "github copilot",
    "copilot",
    "gemini",
    "openai",
    "stable diffusion",
    "adobe firefly",
    "firefly",
    "jasper",
    "writesonic",
    "perplexity",
    "notion ai",
    "canva ai",
    "synthesia",
    "runway",
    # GenAI concepts / practices
    "generative ai",
    "generative artificial intelligence",
    "generatieve ai",
    "generatieve kunstmatige intelligentie",
    "generatief",
    "genai",
    "gen-ai",
    "gen ai",
    "large language model",
    "large language models",
    "prompt engineering",
    "prompt skills",
    "prompting",
    "prompten",
    "llm",
    "llms",
    "ai-generated",
    "ai generated",
    "ai-generated content",
    "ai content",
    "ai copywriting",
    "ai copy",
    "content genereren met ai",
    # Broader AI skill language common in marketing vacancies
    "artificial intelligence",
    "kunstmatige intelligentie",
    "ai-tools",
    "ai tools",
    "ai-tool",
    "ai tool",
    "ai-driven",
    "ai driven",
    "ai-powered",
    "ai powered",
    "ai-native",
    "ai native",
    "ai-gestuurd",
    "ai gestuurd",
    "ai-ondersteund",
    "ai marketing",
    "marketing ai",
    "machine learning",
    "deep learning",
    # Standalone AI (word on its own, so 'email' is NOT counted)
    "ai",
    "a.i.",
]


def _term_to_pattern(term: str) -> re.Pattern[str]:
    """
    Turn a dictionary word into a careful text search.

    Why: if we search for 'gemini' too loosely, we could wrongly count
    company names like 'Capgemini'. Word boundaries help avoid that.
    """
    escaped = re.escape(term)
    # Allow flexible spaces/hyphens inside multi-word terms
    escaped = escaped.replace(r"\ ", r"[\s\-]+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.IGNORECASE)


TERM_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (term, _term_to_pattern(term)) for term in GENAI_TERMS
]


def count_words(text: str) -> int:
    """Count normal words in the requirements text (for the Intensity formula)."""
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))


def find_genai_keywords(text: str) -> tuple[list[str], int]:
    """
    Find AI/GenAI dictionary terms in one requirements text.

    Returns:
    - unique matched terms (for easy reading)
    - total keyword hits used in the Intensity formula

    Longer phrases win: 'AI tools' is counted once, and the short word
    'AI' inside that same phrase is not counted again.
    """
    if not text:
        return [], 0

    ranked = sorted(TERM_PATTERNS, key=lambda item: len(item[0]), reverse=True)
    used_spans: list[tuple[int, int]] = []
    matched_terms: list[str] = []
    hit_count = 0

    def overlaps(start: int, end: int) -> bool:
        return any(not (end <= s or start >= e) for s, e in used_spans)

    for term, pattern in ranked:
        term_found = False
        for match in pattern.finditer(text):
            start, end = match.span()
            if overlaps(start, end):
                continue
            used_spans.append((start, end))
            hit_count += 1
            term_found = True
        if term_found:
            matched_terms.append(term)

    return matched_terms, hit_count


def score_requirements(text: str) -> dict:
    """
    Score one job's requirements text.

    AI Presence = 1 if at least one AI/GenAI word is found, else 0
    AI Intensity = (number of GenAI keywords / total words) * 1000
      -> exactly the formula from your research paper
    """
    text = text or ""
    matched, n_keywords = find_genai_keywords(text)
    n_words = count_words(text)

    if n_words > 0:
        ai_intensity = (n_keywords / n_words) * 1000
    else:
        ai_intensity = 0.0

    return {
        "genai_matches": ", ".join(matched),
        "genai_keyword_count": n_keywords,
        "requirements_word_count": n_words,
        "ai_presence": 1 if n_keywords > 0 else 0,
        "ai_intensity": ai_intensity,
    }


def map_subdiscipline(source_file: str) -> str:
    """Attach marketing area based on which original file the job came from."""
    name = (source_file or "").lower()
    if name.startswith("brand"):
        return "Brand"
    if name.startswith("content"):
        return "Content & Creative"
    if name.startswith("digital"):
        return "Digital & Performance"
    return "Other"


def build_analysis_sample(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Prepare the final analysis table:
    - keep only jobs with requirements text
    - add Subdiscipline
    - add AI Presence and AI Intensity
    """
    df = pd.read_pickle(data_path)

    # Step 2 decision: only jobs where requirements were found
    has_requirements = df["requirements_text"].fillna("").str.len() > 0
    analysis = df.loc[has_requirements].copy()

    analysis["Subdiscipline"] = analysis["source_file"].map(map_subdiscipline)

    scores = analysis["requirements_text"].astype(str).map(score_requirements)
    score_df = pd.DataFrame(scores.tolist(), index=analysis.index)
    analysis = pd.concat([analysis, score_df], axis=1)

    return analysis


def print_simple_summary(analysis: pd.DataFrame) -> None:
    """Print an easy Entry vs Senior overview (no heavy statistics yet)."""
    print("=" * 60)
    print("ANALYSIS SAMPLE")
    print("=" * 60)
    print(f"Jobs kept (with requirements text): {len(analysis)}")
    print(analysis["Seniority"].value_counts().to_string())
    print()
    print("By Subdiscipline:")
    print(analysis.groupby(["Subdiscipline", "Seniority"]).size().unstack(fill_value=0))
    print()

    print("=" * 60)
    print("ENTRY vs SENIOR — AI Presence and AI Intensity")
    print("=" * 60)
    summary = (
        analysis.groupby("Seniority")
        .agg(
            n_jobs=("ai_presence", "size"),
            ai_presence_pct=("ai_presence", lambda s: 100 * s.mean()),
            mean_ai_intensity=("ai_intensity", "mean"),
            median_ai_intensity=("ai_intensity", "median"),
        )
        .round(3)
    )
    print(summary.to_string())
    print()
    print("How to read this:")
    print("- ai_presence_pct = % of jobs that mention at least one GenAI term")
    print("- mean_ai_intensity = average of your formula (keywords / words * 1000)")
    print()

    print("Jobs that mention GenAI (examples):")
    hits = analysis.loc[analysis["ai_presence"] == 1, ["Seniority", "title", "companyName", "genai_matches", "ai_intensity"]]
    print(f"Total AI-present jobs: {len(hits)}")
    if len(hits):
        print(hits.sort_values("ai_intensity", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    analysis_df = build_analysis_sample()
    print_simple_summary(analysis_df)

    # Save a smaller/cleaner file for the next statistical steps
    keep_cols = [
        "id",
        "title",
        "companyName",
        "Seniority",
        "Subdiscipline",
        "location",
        "postedAt",
        "companyEmployeesCount",
        "source_file",
        "requirements_text",
        "requirements_word_count",
        "genai_matches",
        "genai_keyword_count",
        "ai_presence",
        "ai_intensity",
    ]
    existing = [c for c in keep_cols if c in analysis_df.columns]
    analysis_df[existing].to_csv(OUT_PATH, index=False)
    print(f"\nSaved analysis file: {OUT_PATH}")
