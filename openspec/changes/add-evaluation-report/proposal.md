```markdown
# Change: Add evaluation & reporting pipeline

## Why

We need a reproducible, automated evaluation pipeline that generates consistent metrics and human-readable reports for each experiment. Right now experiments may be ad-hoc in notebooks and hard to reproduce or compare.

## What Changes

- Add an evaluation script and report generator that runs model evaluation on a test set and writes a machine-readable JSON metrics file and a human-friendly HTML/Markdown report.
- Add a reusable CLI: `python -m src.evaluate --config config/eval.yaml`.
- Add a small integration test that runs evaluation on a canned fixture dataset.

**BREAKING**: None.

## Impact

- Affected specs: `email-classification` (new ADDED requirement: evaluation/reporting)
- Affected code: `src/evaluate.py` (new), `src/reporting/` (new), `tests/integration/test_evaluate.py` (new)
- CI: add an evaluation run to validate reports in CI pipeline (non-blocking)
```
