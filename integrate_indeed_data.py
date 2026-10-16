"""
Add Indeed vacancies to the main analysis dataset.

In plain words, this script:
1. Opens the Indeed JSON file
2. Puts it in the same table format as the LinkedIn jobs
3. Extracts only the requirements / candidate-profile text
4. Labels Entry (0-4 years) vs Senior (5+ years)
5. Labels marketing subdiscipline (Brand / Content / Digital / Other)
6. Shows clear counts, then merges with the LinkedIn analysis sample
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from load_and_extract_requirements import extract_requirements
from score_genai_requirements import score_requirements

INDEED_PATH = Path(r"D:\Downloads\dataset_indeed-scraper_2026-10-11_13-28-53-464.json")
EXTRA_LINKEDIN_PATH = Path(
    r"D:\Downloads\dataset_linkedin-jobs-scraper_2026-10-11_13-50-29-399.json"
)
LINKEDIN_SCORED = Path(__file__).resolve().parent / "analysis_sample_scored.csv"
OUT_INDEED = Path(__file__).resolve().parent / "indeed_structured.csv"
OUT_EXTRA_LINKEDIN = Path(__file__).resolve().parent / "linkedin_extra_structured.csv"
OUT_COMBINED = Path(__file__).resolve().parent / "analysis_sample_combined.csv"

# Years of experience mentioned near "experience/ervaring"
EXPERIENCE_YEARS = re.compile(
    r"(?ix)"
    r"(?:"
    r"(?P<min1>\d+)\s*[-–to]+\s*(?P<max1>\d+)\s*(?:\+|plus)?\s*(?:years?|jaar|jaren)"
    r"(?:[^.\n]{0,40}(?:experience|ervaring))?"
    r"|"
    r"(?P<min2>\d+)\s*(?:\+|plus)?\s*(?:years?|jaar|jaren)"
    r"(?:[^.\n]{0,40}(?:experience|ervaring|relevant|werkervaring|progressive))?"
    r"|"
    r"(?:experience|ervaring|werkervaring)[^.\n]{0,40}?"
    r"(?P<min3>\d+)\s*(?:\+|plus)?\s*(?:years?|jaar|jaren)"
    r")"
)

ENTRY_TITLE = re.compile(
    r"(?i)\b(junior|entry[- ]level|trainee|traineeship|intern|internship|stagiair|stage|starter|graduate|assistent)\b"
)
SENIOR_TITLE = re.compile(
    r"(?i)\b(senior|lead|principal|director|head|hoofd|manager|vp|chief)\b"
)

# Marketing relevance (exclude pure IT/web-dev noise from this scrape)
MARKETING_HINT = re.compile(
    r"(?i)\b("
    r"marketing|marketeer|brand|campaign|campagne|content|creative|seo|sea|"
    r"social media|performance|growth|communicatie|copywriter|advertising|"
    r"digitale marketing|online marketing|crm|pr\b|product marketing"
    r")\b"
)
NON_MARKETING_TITLE = re.compile(
    r"(?i)\b(web developer|software|frontend|front-end|backend|back-end|java|python developer|devops|data engineer)\b"
)


def infer_years_required(text: str) -> float | None:
    """
    Try to find how many years of experience the job asks for.
    Returns the lowest number found in an experience sentence, or None.
    """
    if not text:
        return None

    years: list[float] = []
    for match in EXPERIENCE_YEARS.finditer(text):
        vals = []
        for key in ("min1", "max1", "min2", "min3"):
            raw = match.group(key)
            if raw is not None:
                vals.append(int(raw))
        if not vals:
            continue
        # Ignore unrealistic company-age style numbers in experience phrasing
        vals = [v for v in vals if v <= 20]
        if not vals:
            continue
        years.append(min(vals))

    if not years:
        return None
    return float(min(years))


def classify_seniority(title: str, description: str, requirements_text: str) -> tuple[str, str, float | None]:
    """
    Entry = 0-4 years (includes earlier 0-2 rule, plus 3-4 years as Entry)
    Senior = 5+ years (or senior/lead/manager title if years missing)
    Unclassified = no clear years/title signal
    """
    search_text = " ".join([requirements_text or "", description or "", title or ""])
    years = infer_years_required(search_text)

    if years is not None:
        if years <= 4:
            reason = "years_0_to_2" if years <= 2 else f"years_{int(years)}_as_entry"
            return "Entry", reason, years
        if years >= 5:
            return "Senior", "years_5_plus", years
        return "Unclassified", f"years_{int(years)}", years

    title = title or ""
    if ENTRY_TITLE.search(title):
        return "Entry", "title_entry_keyword", None
    if SENIOR_TITLE.search(title):
        return "Senior", "title_senior_keyword", None

    return "Unclassified", "no_clear_signal", None


def classify_subdiscipline(title: str, description: str) -> str:
    """Guess marketing area from job title + description words."""
    text = f"{title or ''}\n{description or ''}".lower()

    scores = {
        "Brand": 0,
        "Content & Creative": 0,
        "Digital & Performance": 0,
    }

    brand_terms = [
        "brand",
        "merk",
        "campaign strategy",
        "campagne strategie",
        "branding",
        "positionering",
        "corporate communication",
        "merkstrategie",
    ]
    content_terms = [
        "content",
        "creative",
        "creatief",
        "copywriter",
        "copywriting",
        "redacteur",
        "social media",
        "ux writing",
        "design",
        "vormgeving",
        "storytelling",
    ]
    digital_terms = [
        "digital",
        "performance",
        "seo",
        "sea",
        "ppc",
        "google ads",
        "paid media",
        "paid social",
        "growth",
        "crm",
        "marketing automation",
        "analytics",
        "online marketing",
        "digitale marketing",
        "conversion",
        "ecommerce",
        "e-commerce",
    ]

    for term in brand_terms:
        if term in text:
            scores["Brand"] += text.count(term)
    for term in content_terms:
        if term in text:
            scores["Content & Creative"] += text.count(term)
    for term in digital_terms:
        if term in text:
            scores["Digital & Performance"] += text.count(term)

    # Prefer title signals a bit more
    title_l = (title or "").lower()
    if any(t in title_l for t in ["brand", "merk", "campaign"]):
        scores["Brand"] += 3
    if any(t in title_l for t in ["content", "creative", "copy", "social", "redacteur"]):
        scores["Content & Creative"] += 3
    if any(t in title_l for t in ["digital", "performance", "seo", "sea", "growth", "ads", "online"]):
        scores["Digital & Performance"] += 3

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "Other / Unclear"
    return best


def is_marketing_job(title: str, description: str, search_input) -> bool:
    """Keep marketing jobs; drop clear non-marketing roles from this scrape."""
    title = title or ""
    description = description or ""

    if isinstance(search_input, dict):
        position = str(search_input.get("position") or "").lower()
        if "web developer" in position or "software" in position:
            # still keep if the job itself is clearly marketing
            if not MARKETING_HINT.search(title):
                return False

    if NON_MARKETING_TITLE.search(title) and not MARKETING_HINT.search(title):
        return False

    return bool(MARKETING_HINT.search(title) or MARKETING_HINT.search(description[:1500]))


def _score_row(
    *,
    rec_id,
    title: str,
    company: str,
    description: str,
    location,
    posted_at,
    company_employees,
    source_file: str,
    source_platform: str,
    link,
    salary,
    search_input=None,
) -> dict:
    requirements_text = extract_requirements(description)
    seniority, seniority_reason, years = classify_seniority(title, description, requirements_text)
    subdiscipline = classify_subdiscipline(title, description)
    marketing = is_marketing_job(title, description, search_input)

    if posted_at:
        posted_at = str(posted_at)[:10]

    row = {
        "id": rec_id,
        "title": title,
        "companyName": company,
        "Seniority": seniority,
        "seniority_reason": seniority_reason,
        "years_required": years,
        "Subdiscipline": subdiscipline,
        "location": location,
        "postedAt": posted_at,
        "companyEmployeesCount": company_employees,
        "source_file": source_file,
        "source_platform": source_platform,
        "is_marketing": marketing,
        "descriptionText": description,
        "requirements_text": requirements_text,
        "link": link,
        "salary": salary,
    }
    row.update(score_requirements(requirements_text))
    return row


def load_indeed(path: Path = INDEED_PATH) -> pd.DataFrame:
    with path.open(encoding="utf-8") as fh:
        records = json.load(fh)

    rows = []
    for rec in records:
        rows.append(
            _score_row(
                rec_id=rec.get("id"),
                title=rec.get("positionName") or "",
                company=rec.get("company") or "",
                description=rec.get("description") or "",
                location=rec.get("location"),
                posted_at=rec.get("postingDateParsed") or rec.get("postedAt"),
                company_employees=pd.NA,
                source_file=path.name,
                source_platform="Indeed",
                link=rec.get("url"),
                salary=rec.get("salary"),
                search_input=rec.get("searchInput"),
            )
        )
    return pd.DataFrame(rows)


def load_extra_linkedin(path: Path = EXTRA_LINKEDIN_PATH) -> pd.DataFrame:
    """Load an extra LinkedIn scrape and classify Entry 0-4 / Senior 5+."""
    with path.open(encoding="utf-8") as fh:
        records = json.load(fh)

    rows = []
    for rec in records:
        rows.append(
            _score_row(
                rec_id=rec.get("id"),
                title=rec.get("title") or "",
                company=rec.get("companyName") or "",
                description=rec.get("descriptionText") or "",
                location=rec.get("location"),
                posted_at=rec.get("postedAt"),
                company_employees=rec.get("companyEmployeesCount"),
                source_file=path.name,
                source_platform="LinkedIn_extra",
                link=rec.get("link"),
                salary=rec.get("salary"),
                search_input=None,
            )
        )
    return pd.DataFrame(rows)


def print_overview(df: pd.DataFrame, label: str) -> None:
    print("=" * 60)
    print(f"{label} — SIMPLE OVERVIEW")
    print("=" * 60)
    print(f"Total vacancies loaded: {len(df)}")
    print(f"Marketing-related vacancies: {df['is_marketing'].sum()}")
    print(f"Non-marketing filtered out later: {(~df['is_marketing']).sum()}")
    print()

    marketing = df[df["is_marketing"]].copy()
    print("SENIORITY (marketing jobs only)")
    print("Entry = 0-4 years (or junior/trainee title)")
    print("Senior = 5+ years (or senior/manager/lead title)")
    print(marketing["Seniority"].value_counts().to_string())
    print()

    print("SENIORITY x SUBDISCIPLINE (marketing jobs only)")
    pivot = (
        marketing.groupby(["Subdiscipline", "Seniority"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Entry", "Senior", "Unclassified"], fill_value=0)
    )
    pivot["Total"] = pivot.sum(axis=1)
    print(pivot.to_string())
    print()

    print("Requirements text found?")
    print(
        marketing.assign(has_req=marketing["requirements_text"].fillna("").str.len() > 0)["has_req"]
        .value_counts()
        .to_string()
    )
    print()

    usable = marketing[
        marketing["Seniority"].isin(["Entry", "Senior"])
        & (marketing["requirements_text"].fillna("").str.len() > 0)
    ]
    print("Usable for main analysis (marketing + Entry/Senior + requirements text):", len(usable))
    if len(usable):
        print(usable.groupby(["Subdiscipline", "Seniority"]).size().unstack(fill_value=0).to_string())


def usable_slice(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["is_marketing"]
        & df["Seniority"].isin(["Entry", "Senior"])
        & (df["requirements_text"].fillna("").str.len() > 0)
    ].copy()


def build_combined(*extra_frames: pd.DataFrame) -> pd.DataFrame:
    linkedin = pd.read_csv(LINKEDIN_SCORED)
    linkedin["source_platform"] = "LinkedIn_NL"
    linkedin["is_marketing"] = True
    linkedin["seniority_reason"] = "linkedin_source_file"
    linkedin["years_required"] = pd.NA

    keep = [
        "id",
        "title",
        "companyName",
        "Seniority",
        "Subdiscipline",
        "location",
        "postedAt",
        "companyEmployeesCount",
        "source_file",
        "source_platform",
        "seniority_reason",
        "years_required",
        "requirements_text",
        "requirements_word_count",
        "genai_matches",
        "genai_keyword_count",
        "ai_presence",
        "ai_intensity",
    ]

    parts = []
    for frame in (linkedin, *extra_frames):
        cols = [c for c in keep if c in frame.columns]
        parts.append(frame[cols])

    combined = pd.concat(parts, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["companyName", "title"], keep="first")
    print(f"\nCombined rows before dedupe: {before}; after: {len(combined)}")
    return combined


if __name__ == "__main__":
    indeed = load_indeed()
    print_overview(indeed, "INDEED FILE")
    indeed.to_csv(OUT_INDEED, index=False)
    print(f"\nSaved full Indeed structured file: {OUT_INDEED}")

    extra_li = load_extra_linkedin()
    print()
    print_overview(extra_li, "EXTRA LINKEDIN FILE")
    print("\nNOTE: this extra LinkedIn scrape is United Kingdom locations, not Netherlands.")
    extra_li.to_csv(OUT_EXTRA_LINKEDIN, index=False)
    print(f"Saved extra LinkedIn structured file: {OUT_EXTRA_LINKEDIN}")

    combined = build_combined(usable_slice(indeed), usable_slice(extra_li))
    print("\nCOMBINED DATASET (LinkedIn NL + Indeed + extra LinkedIn)")
    print(combined["source_platform"].value_counts().to_string())
    print("\nSeniority:")
    print(combined["Seniority"].value_counts().to_string())
    print("\nSubdiscipline x Seniority:")
    print(combined.groupby(["Subdiscipline", "Seniority"]).size().unstack(fill_value=0).to_string())
    print("\nAI presence % by Seniority:")
    print(
        combined.groupby("Seniority")["ai_presence"]
        .mean()
        .mul(100)
        .round(1)
        .to_string()
    )

    combined.to_csv(OUT_COMBINED, index=False)
    print(f"\nSaved combined analysis file: {OUT_COMBINED}")
