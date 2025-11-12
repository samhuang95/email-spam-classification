```markdown
## 1. Data

- [ ] 1.1 Add downloader script `scripts/download_dataset.py` to fetch the dataset from the provided URL and place a small fixture under `tests/fixtures/`.

## 2. Implementation

- [ ] 2.1 Create `src/train_baseline.py` to preprocess text, train a Logistic Regression classifier, and save model artifacts.
- [ ] 2.2 Add optional SVM training task to compare performance.
- [ ] 2.3 Create `src/evaluate.py` or integrate evaluation into `src/train_baseline.py` and produce JSON metrics + Markdown report.

## 3. Tests

- [ ] 3.1 Add unit tests for preprocessing and feature extraction.
- [ ] 3.2 Add integration test `tests/integration/test_baseline_train.py` using fixture dataset to ensure end-to-end run.

## Evaluation / Reporting tasks

- [ ] 3.3 Create `src/evaluate.py` CLI to load a model and test data and compute metrics.
- [ ] 3.4 Create `src/reporting/` to render Markdown/HTML reports and write JSON metrics.
- [ ] 3.5 Add `config/eval.yaml` for evaluation parameters (metrics, thresholds, dataset paths).
- [ ] 3.6 Add integration test `tests/integration/test_evaluate.py` using a small fixture dataset.

## 4. Docs

- [ ] 4.1 Update `README.md` with phase1 instructions and how to run baseline training/evaluation.

## 5. CI

- [ ] 5.1 Add CI job to run the baseline training on a small fixture dataset and validate outputs.
```
