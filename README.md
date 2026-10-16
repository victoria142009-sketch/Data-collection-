# Generative AI Skill Requirements in Marketing Job Postings

This repository contains the Python scripts used for a university research project on Generative AI (GenAI) skill requirements in marketing vacancies (entry-level vs senior roles), using job postings from the Netherlands and the United Kingdom.

## What this project does
1. Loads LinkedIn and Indeed job vacancy data
2. Extracts the candidate requirements section from each job description
3. Codes seniority (Entry vs Senior) and marketing subdiscipline (Brand / Content & Creative / Digital & Performance)
4. Measures AI Presence and AI Intensity in the requirements text
5. Runs statistical tests for the research hypotheses

## Main metrics
- **AI Presence**: 1 if at least one AI/GenAI keyword appears in the requirements text, otherwise 0
- **AI Intensity**: (number of GenAI keywords / total words in requirements) × 1000

## Python scripts

### 1. `load_and_extract_requirements.py`
Loads the original LinkedIn JSON files, adds a Seniority label, removes duplicates (same company + title), and extracts only the requirements section from each job description using English/Dutch section headers (e.g. Requirements, You Bring, Jouw profiel, Functie-eisen).

### 2. `score_genai_requirements.py`
Searches the requirements text with a bilingual AI/GenAI keyword dictionary and calculates:
- AI Presence
- AI Intensity
- matched keywords

### 3. `integrate_indeed_data.py`
Adds Indeed vacancies and an extra LinkedIn scrape to the main dataset. It:
- extracts requirements text
- codes Entry (0–4 years) vs Senior (5+ years)
- codes marketing subdiscipline
- merges all sources into one analysis file

### 4. `analyze_ai_presence_intensity.py`
Runs the main Entry vs Senior analysis:
- descriptive statistics for AI Presence and AI Intensity
- Chi-square test for AI Presence
- Mann-Whitney U test for AI Intensity

### 5. `analyze_netherlands_only.py`
Repeats the main analysis using only Netherlands vacancies (excluding UK postings), as a robustness check.

### 6. `analyze_h2_interaction.py`
Tests Hypothesis 2 with binary logistic regression:
- AI Presence ~ Seniority × Subdiscipline
- checks whether the Entry vs Senior difference depends on marketing subdiscipline

## How to run
1. Install required packages:
   `pandas`, `numpy`, `scipy`, `statsmodels`
2. Place the input JSON vacancy files in the folder paths used in the scripts (or update the file paths).
3. Run the scripts in this order:
   1. `load_and_extract_requirements.py`
   2. `score_genai_requirements.py`
   3. `integrate_indeed_data.py`
   4. `analyze_ai_presence_intensity.py`
   5. `analyze_netherlands_only.py` (optional)
   6. `analyze_h2_interaction.py`

## Notes for the reader
- Only vacancies with an identifiable requirements section were included in the final analysis sample.
- Duplicate vacancies with the same company name and job title were removed.
- The final analysis sample contained 371 vacancies (256 from the Netherlands and 115 from the United Kingdom).
