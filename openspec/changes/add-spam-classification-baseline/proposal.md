```markdown
# Change: Add spam classification baseline (phase1)

## Why

This project needs a reproducible baseline for spam classification to compare future models and experiments. A simple, well-documented baseline reduces ambiguity and gives a point of comparison for later phases.

## What Changes

- Add a baseline training and evaluation pipeline that downloads a public dataset, preprocesses text, trains a classifier, and writes evaluation artifacts (JSON metrics + human-readable report).
- Phase 1 will train a baseline classifier (Logistic Regression). The user also mentioned SVM as an alternative baseline; we will include an experimental task to train an SVM so you can compare results.
- Create integration tests and CI checks that run the pipeline on a small fixture dataset.

## Plan / Phases

- phase1-baseline: Build a baseline spam classifier using Logistic Regression (and optionally SVM) using the dataset at:
  - https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv
- phase2: (placeholder) additional experiments / features (left empty as requested)
- phase3: (placeholder) additional experiments / features (left empty as requested)

## Impact

- Affected specs: `email-classification` (add baseline capability)
- Affected code: new `src/train_baseline.py`, `src/evaluate.py` (or integrate into `train_baseline`), `src/reporting/`, fixtures under `tests/fixtures/`, and CI workflow updates.
- Non-breaking: project structure changes only; no API used by external services.

## Evaluation & Reporting (merged)

This change also includes an evaluation and reporting pipeline that generates consistent metrics and human-readable reports for each experiment. The evaluation pipeline will be runnable as a CLI (e.g. `python -m src.evaluate --config config/eval.yaml`) and will produce a JSON metrics file and a Markdown/HTML report.

## What I merged in from the evaluation proposal

- Add `src/evaluate.py` and `src/reporting/` to render reports and write JSON metrics.
- Add `config/eval.yaml` for evaluation parameters (metrics, thresholds, dataset paths).
- Add integration test `tests/integration/test_evaluate.py` and CI job to validate the evaluation run on a fixture dataset.

**BREAKING**: None.
```
