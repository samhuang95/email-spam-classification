# Project Context

## Purpose

[Describe your project's purpose and goals]
This repository implements an email spam classification project (NCHU HW3). The primary purpose is to train, evaluate, and deploy models that classify emails as spam or ham and to provide reproducible evaluation and reporting for experiments.

## Tech Stack

- [List your primary technologies]
- [e.g., TypeScript, React, Node.js]
  Python 3.10+ (data processing, model training)
  scikit-learn, pandas, numpy (modeling and data manipulation)
  Jupyter / ipynb for experiments
  pytest for unit tests
  Git for version control (main branch protected)

## Project Conventions

### Code Style

[Describe your code style preferences, formatting rules, and naming conventions]
Code style:
Follow PEP8 for Python code; use black for formatting and flake8 for linting.
Module and package names: lowercase_with_underscores.
Function and variable names: lowercase_with_underscores.
Class names: PascalCase.

### Architecture Patterns

[Document your architectural decisions and patterns]
Architecture:
Small, single-repo project with clear separation: data/, src/, notebooks/, reports/.
`src/` contains reusable modules: data processing, feature extraction, model, evaluation.
Keep experiments reproducible via fixed random seeds and a `config/` folder.

### Testing Strategy

[Explain your testing approach and requirements]
Testing:
Use `pytest` for unit tests covering preprocessing, feature extraction, and evaluation metrics.
Add small integration tests for end-to-end training pipeline on a tiny sample dataset.
CI should run tests, linting, and a reproducible evaluation script.

### Git Workflow

[Describe your branching strategy and commit conventions]
Git workflow:
`main` is protected and always green.
Feature branches: `feat/<short-desc>`; bugfix branches: `fix/<short-desc>`.
Commit messages follow Conventional Commits (type(scope): summary).

## Domain Context

[Add domain-specific knowledge that AI assistants need to understand]
Domain context:
Input: raw email text (headers + body) – primary features include token counts, sender metadata, and simple heuristics.
Labels: binary `spam` / `ham`.
Evaluation metrics: precision, recall, F1, ROC-AUC; emphasis on precision at high recall for some use cases.

## Important Constraints

[List any technical, business, or regulatory constraints]
Constraints:
Data privacy: email content should not be exfiltrated; treat sample data as sensitive.
Models should be small and fast to evaluate for homework / teaching constraints.

## External Dependencies

[Document key external services, APIs, or systems]
External dependencies:
No production external services expected. Optional: cloud storage (S3) for datasets.
Local Python environment with packages described in `requirements.txt`.
