from typing import List, Tuple
from collections import Counter
import re
from pathlib import Path
import joblib


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d[\d\-\s]{7,}\d)\b")


def normalize_text(text: str, keep_numbers: bool = False) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = text.lower()
    t = URL_RE.sub("<URL>", t)
    t = EMAIL_RE.sub("<EMAIL>", t)
    t = PHONE_RE.sub("<PHONE>", t)
    if not keep_numbers:
        t = re.sub(r"\d+", "<NUM>", t)
    t = re.sub(r"[^\w\s<>]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_pipeline_from_paths() -> object:
    """Try to load a pipeline from models/ or artifacts/"""
    # Prefer models/ if exists
    mdir = Path('models')
    a_path = Path('artifacts/baseline_model.joblib')
    cand = None
    if mdir.exists():
        # try common filenames
        for name in ('spam_pipeline.joblib', 'spam_logreg_model.joblib', 'pipeline.joblib'):
            p = mdir / name
            if p.exists():
                cand = p
                break
    if cand is None and a_path.exists():
        cand = a_path
    if cand is None:
        raise FileNotFoundError('No pipeline found in models/ or artifacts/')
    return joblib.load(cand)


def get_vectorizer_and_model(pipeline):
    """Given a sklearn Pipeline or estimator, try to extract vectorizer and estimator."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    if hasattr(pipeline, 'named_steps'):
        vec = None
        clf = None
        for name, step in pipeline.named_steps.items():
            if isinstance(step, TfidfVectorizer):
                vec = step
            else:
                clf = step
        if vec is None:
            # first transformer
            for step in pipeline.steps:
                if isinstance(step[1], TfidfVectorizer):
                    vec = step[1]
                    break
        # fallback: assume first step transforms
        if vec is None:
            vec = pipeline.steps[0][1]
        if clf is None:
            clf = pipeline.steps[-1][1]
        return vec, clf
    else:
        # not a pipeline, unknown
        return None, pipeline


def token_importance_by_removal(text: str, pipeline, top_k: int = 10) -> List[Tuple[str, float]]:
    """Compute token importance by removing each token and measuring drop in spam probability.

    Returns list of (token, delta_prob) sorted by absolute importance desc.
    """
    vec, clf = get_vectorizer_and_model(pipeline)
    cleaned = normalize_text(text)
    tokens = [t for t in cleaned.split() if t]
    if not tokens:
        return []
    base_prob = None
    try:
        base_prob = float(pipeline.predict_proba([cleaned])[0][1])
    except Exception:
        try:
            base_prob = float(clf.predict_proba(vec.transform([cleaned]))[0][1])
        except Exception:
            base_prob = None
    results = []
    uniq = list(dict.fromkeys(tokens))
    for tok in uniq:
        modified_tokens = [t for t in tokens if t != tok]
        modified_text = " ".join(modified_tokens) if modified_tokens else ""
        try:
            if base_prob is None:
                prob = None
            else:
                prob = float(pipeline.predict_proba([modified_text])[0][1])
        except Exception:
            try:
                prob = float(clf.predict_proba(vec.transform([modified_text]))[0][1])
            except Exception:
                prob = None
        if base_prob is None or prob is None:
            delta = 0.0
        else:
            delta = base_prob - prob
        results.append((tok, delta))
    # sort by absolute impact
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    return results[:top_k]


def try_shap_explain(texts: List[str], pipeline, nsamples: int = 50):
    """Attempt to compute SHAP values for given texts. Returns (tokens, shap_values) or None on failure."""
    try:
        import shap
        import numpy as np
    except Exception:
        return None
    vec, clf = get_vectorizer_and_model(pipeline)
    # prepare background
    # use first nsamples from data if pipeline has training data? we don't have it, so use random small set of texts
    # Build wrapper that maps raw texts -> model probabilities
    def f(x_texts):
        # x_texts: list of raw texts
        return pipeline.predict_proba(x_texts)[:, 1]

    try:
        explainer = shap.KernelExplainer(f, texts[:min(len(texts), max(1, nsamples))])
        shap_values = explainer.shap_values(texts)
        return shap_values
    except Exception:
        return None
