"""Utilities to compute visualization data for the Streamlit app.

These functions compute ROC/PR curve points, confusion matrix counts,
label distribution, and top words for spam/ham.
"""
from pathlib import Path
import json
from collections import Counter
import re
from typing import Dict, Any

import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    df = df.rename(columns={0: 'label', 1: 'text'})
    return df


def label_distribution(df: pd.DataFrame) -> Dict[str, int]:
    return df['label'].value_counts().to_dict()


def top_words(df: pd.DataFrame, n=20) -> Dict[str, Any]:
    # simple tokenization and counts per label
    token_re = re.compile(r"\b\w+\b")
    counts = {'spam': Counter(), 'ham': Counter()}
    for _, row in df.iterrows():
        label = row['label']
        text = str(row['text']).lower()
        tokens = token_re.findall(text)
        counts[label].update(tokens)
    return {
        'spam': counts['spam'].most_common(n),
        'ham': counts['ham'].most_common(n),
    }


def compute_curves(model, X, y):
    # model should support predict_proba
    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception:
        # fallback to decision_function
        try:
            probs = model.decision_function(X)
        except Exception:
            probs = None
    preds = model.predict(X)
    result = {}
    if probs is not None:
        fpr, tpr, _ = roc_curve(y, probs)
        prec, rec, _ = precision_recall_curve(y, probs)
        result['roc'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        result['pr'] = {'precision': prec.tolist(), 'recall': rec.tolist()}
    else:
        result['roc'] = {'fpr': [], 'tpr': []}
        result['pr'] = {'precision': [], 'recall': []}
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    result['confusion_matrix'] = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    return result


def generate_visuals(model_path: str, data_path: str, out_path: str):
    df = load_dataset(Path(data_path))
    X = df['text'].astype(str)
    y = df['label'].map({'ham': 0, 'spam': 1})
    import joblib

    model = joblib.load(model_path)
    curves = compute_curves(model, X, y)
    visuals = {
        'label_distribution': label_distribution(df),
        'top_words': top_words(df, n=30),
        'curves': curves,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(visuals))
    return visuals
from typing import Tuple
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
import plotly.graph_objects as go
import pandas as pd


def compute_roc_pr(model, X, y) -> Tuple[dict, dict]:
    # model: sklearn pipeline with predict_proba
    probs = None
    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception:
        # fallback to decision_function
        try:
            probs = model.decision_function(X)
        except Exception:
            raise RuntimeError('Model does not provide probabilities or decision_function')
    fpr, tpr, _ = roc_curve(y, probs)
    precision, recall, _ = precision_recall_curve(y, probs)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)
    return {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'roc_auc': float(roc_auc),
    }, {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'pr_auc': float(pr_auc),
    }


def plot_roc(fpr, tpr, roc_auc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={roc_auc:.3f})'))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
    fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    return fig


def plot_pr(precision, recall, pr_auc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR (AUC={pr_auc:.3f})'))
    fig.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision')
    return fig


def plot_confusion_matrix(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    z = [[tn, fp], [fn, tp]]
    labels = [['TN', 'FP'], ['FN', 'TP']]
    fig = go.Figure(data=go.Heatmap(z=z, x=['Pred Ham','Pred Spam'], y=['Actual Ham','Actual Spam'], colorscale='Blues', showscale=True))
    fig.update_layout(title='Confusion Matrix')
    return fig


def plot_label_distribution(df: pd.DataFrame, label_col='label'):
    counts = df[label_col].value_counts().reset_index()
    counts.columns = [label_col, 'count']
    fig = go.Figure(go.Bar(x=counts[label_col], y=counts['count']))
    fig.update_layout(title='Label Distribution', xaxis_title='Label', yaxis_title='Count')
    return fig


def top_n_words(df, text_col='text', label_col='label', n=20, label='spam'):
    texts = df[df[label_col] == label][text_col].astype(str)
    from collections import Counter
    import re
    words = Counter()
    for t in texts:
        tokens = re.findall(r"\w+", t.lower())
        words.update(tokens)
    most = words.most_common(n)
    labels = [w for w,c in most]
    counts = [c for w,c in most]
    fig = go.Figure(go.Bar(x=labels, y=counts))
    fig.update_layout(title=f'Top {n} words for {label}', xaxis_title='Word', yaxis_title='Count')
    return fig
