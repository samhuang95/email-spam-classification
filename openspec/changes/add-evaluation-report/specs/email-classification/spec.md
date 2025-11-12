```markdown
## ADDED Requirements

### Requirement: Evaluation and Report Generation

The system SHALL provide a reproducible evaluation pipeline that computes standard classification metrics (precision, recall, F1, ROC-AUC) and outputs both a machine-readable metrics file (JSON) and a human-readable report (Markdown or HTML).

#### Scenario: Generate evaluation report for a trained model

- **GIVEN** a trained classifier artifact and a held-out test dataset
- **WHEN** the evaluation pipeline is executed with `python -m src.evaluate --config config/eval.yaml`
- **THEN** a JSON metrics file is written to the configured output directory
- **AND** a Markdown (or HTML) report summarizing the metrics and confusion matrix is produced

#### Scenario: CI validation of evaluation run

- **GIVEN** a small fixture dataset included in the repo
- **WHEN** the CI pipeline runs the evaluation script
- **THEN** the script exits with code 0 and produces a metrics JSON file
```
