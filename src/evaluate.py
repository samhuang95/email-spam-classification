"""Evaluation script to compute metrics for a trained model and produce JSON + Markdown report."""
from pathlib import Path
import argparse
import json
import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def load_model(path: Path):
    return joblib.load(path)


def load_data(path: Path):
    df = pd.read_csv(path, header=None)
    df = df.rename(columns={0: 'label', 1: 'text'})
    y = df['label'].map({'ham': 0, 'spam': 1})
    X = df['text'].astype(str)
    return X, y


def evaluate(model, X, y):
    preds = model.predict(X)
    probs = None
    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception:
        pass
    metrics = {
        'precision': float(precision_score(y, preds)),
        'recall': float(recall_score(y, preds)),
        'f1': float(f1_score(y, preds)),
    }
    if probs is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y, probs))
        except Exception:
            metrics['roc_auc'] = None
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    metrics['confusion_matrix'] = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    return metrics


def write_report(metrics: dict, out_json: Path, out_md: Path):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2))
    md = ["# Evaluation Report", "", "## Metrics", ""]
    for k, v in metrics.items():
        if k != 'confusion_matrix':
            md.append(f"- **{k}**: {v}")
    md.append("")
    cm = metrics.get('confusion_matrix')
    if cm:
        md.append("## Confusion Matrix")
        md.append(f"- TN: {cm['tn']}")
        md.append(f"- FP: {cm['fp']}")
        md.append(f"- FN: {cm['fn']}")
        md.append(f"- TP: {cm['tp']}")
    out_md.write_text("\n".join(md))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='artifacts/baseline_model.joblib')
    p.add_argument('--data', default='data/sms_spam.csv')
    p.add_argument('--out-json', default='artifacts/metrics.json')
    p.add_argument('--out-md', default='artifacts/report.md')
    args = p.parse_args()
    model = load_model(Path(args.model))
    X, y = load_data(Path(args.data))
    metrics = evaluate(model, X, y)
    write_report(metrics, Path(args.out_json), Path(args.out_md))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
