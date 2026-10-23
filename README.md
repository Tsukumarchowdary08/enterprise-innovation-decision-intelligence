# Enterprise Innovation Decision Intelligence

Reproducible research benchmark on uncertainty-aware AI and cost-sensitive review for enterprise innovation using the Harmonized Latin American Innovation Surveys (LAIS).

**Outcome:** observed product innovation (`dnewpd`). This study does not claim to measure digital transformation readiness or entrepreneurship directly.

## Data
Download the LAIS firm-level CSV from the Inter-American Development Bank: https://doi.org/10.60966/hbz4-cm10. Data and dictionary are not redistributed here. The variables are `country`, `year`, `dictexp`, `drdintexp`, and `dIPexp` (knowledge acquisition/consultancy/technical-assistance expenditure).

## Reproduce
```bash
python -m pip install -r requirements.txt
export LAIS_CSV=/absolute/path/to/lais-public.csv
python run.py
python robustness.py
```
Windows PowerShell: `$env:LAIS_CSV="C:\\path\\to\\lais-public.csv"`.

`run.py` generates **two figures**, **two CSV tables**, and `results.json`. `robustness.py` generates a paired bootstrap, country subgroup report, and reviewer-accuracy sensitivity.

## Scientific limitations
- Pooled random split, **not** country-held-out or prospective validation.
- Contemporaneous predictors may introduce temporal ambiguity; associations are **not causal**.
- Decision costs (missed positive 5, false positive 1, review 0.2) are **hypothetical**, not observed business savings.
- Main review policy assumes perfect reviewers; sensitivity analysis explores imperfect accuracy.
- Expected-loss review prioritization is an **established technique**, not a new algorithm.
- Confidence review is a heuristic baseline, not a fully optimized cost-sensitive comparator.

**Dataset citation:** Inter-American Development Bank, Harmonized Latin American Innovation Surveys Database (LAIS), DOI: [10.60966/hbz4-cm10](https://doi.org/10.60966/hbz4-cm10).

This repository is an academic research artifact, not production decision software.
