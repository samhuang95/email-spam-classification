#!/usr/bin/env python3
"""Generate a PDF summary report from artifacts.

Reads `artifacts/metrics.json` and `artifacts/visuals.json` (if present) and writes
`artifacts/report.pdf` with a short summary (metrics, ROC/PR charts, confusion matrix and top tokens).

This script uses reportlab and matplotlib. Install into your venv before running:

.venv\Scripts\pip.exe install reportlab matplotlib

Usage:
  .venv\Scripts\python.exe scripts/generate_report_pdf.py --metrics artifacts/metrics.json --visuals artifacts/visuals.json --out artifacts/report.pdf
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf8'))
    except Exception:
        return None


def make_pdf(metrics, visuals, out_path: Path, model_path: str | None = None):
    # We'll embed matplotlib charts (ROC, PR, confusion matrix) into the PDF via in-memory images.
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    import matplotlib.pyplot as plt
    import io
    import numpy as np

    doc = SimpleDocTemplate(str(out_path), pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    title = Paragraph('Spam Classification — Summary Report', styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))

    meta_text = f'Report generated: {datetime.utcnow().isoformat()} UTC'
    if model_path:
        meta_text += f' | Model: {model_path}'
    story.append(Paragraph(meta_text, styles['Normal']))
    story.append(Spacer(1, 12))

    # Metrics table
    if metrics:
        story.append(Paragraph('Evaluation metrics', styles['Heading2']))
        data = [['metric', 'value']]
        for k in ('precision', 'recall', 'f1', 'roc_auc'):
            if k in metrics:
                data.append([k, f"{metrics[k]:.4f}" if isinstance(metrics[k], (int,float)) else str(metrics[k])])
        t = Table(data, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ]))
        story.append(t)
        story.append(Spacer(1,12))

    # Prepare and embed ROC and PR plots if available
    if visuals and 'curves' in visuals:
        curves = visuals['curves']
        # ROC
        roc = curves.get('roc') or {}
        pr = curves.get('pr') or {}
        if roc and roc.get('fpr') and roc.get('tpr'):
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(roc['fpr'], roc['tpr'], label=f"ROC AUC {metrics.get('roc_auc', '') if metrics else ''}")
            ax.plot([0,1],[0,1], linestyle='--', color='gray')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            story.append(Paragraph('ROC Curve', styles['Heading2']))
            story.append(Image(buf, width=400, height=300))
            story.append(Spacer(1,12))

        # PR
        if pr and pr.get('precision') and pr.get('recall'):
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(pr['recall'], pr['precision'])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            story.append(Paragraph('Precision-Recall Curve', styles['Heading2']))
            story.append(Image(buf, width=400, height=300))
            story.append(Spacer(1,12))

        # Confusion matrix heatmap
        cm = curves.get('confusion_matrix')
        if cm:
            try:
                cm_arr = np.array([[cm.get('tn',0), cm.get('fp',0)],[cm.get('fn',0), cm.get('tp',0)]], dtype=int)
                fig, ax = plt.subplots(figsize=(4,3))
                im = ax.imshow(cm_arr, cmap='Blues')
                ax.set_xticks([0,1])
                ax.set_yticks([0,1])
                ax.set_xticklabels(['pred_ham','pred_spam'])
                ax.set_yticklabels(['actual_ham','actual_spam'])
                for (i,j), val in np.ndenumerate(cm_arr):
                    ax.text(j, i, str(val), ha='center', va='center', color='black')
                ax.set_title('Confusion Matrix')
                buf = io.BytesIO()
                fig.tight_layout()
                fig.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                story.append(Paragraph('Confusion Matrix', styles['Heading2']))
                story.append(Image(buf, width=300, height=240))
                story.append(Spacer(1,12))
            except Exception:
                pass

    # Top tokens (text tables)
    if visuals and 'top_words' in visuals:
        story.append(Paragraph('Top tokens (spam)', styles['Heading2']))
        spam = visuals['top_words'].get('spam', [])[:20]
        if spam:
            data = [['token','count']] + [[str(w[0]), str(w[1])] for w in spam]
            t = Table(data, hAlign='LEFT')
            t.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(1,0), colors.lightgrey),
                ('GRID',(0,0),(-1,-1),0.25,colors.grey),
            ]))
            story.append(t)
            story.append(Spacer(1,12))

        story.append(Paragraph('Top tokens (ham)', styles['Heading2']))
        ham = visuals['top_words'].get('ham', [])[:20]
        if ham:
            data = [['token','count']] + [[str(w[0]), str(w[1])] for w in ham]
            t = Table(data, hAlign='LEFT')
            t.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(1,0), colors.lightgrey),
                ('GRID',(0,0),(-1,-1),0.25,colors.grey),
            ]))
            story.append(t)
            story.append(Spacer(1,12))

    if not metrics and not visuals:
        story.append(Paragraph('No metrics or visuals found to include in the report.', styles['Normal']))

    doc.build(story)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', default='artifacts/metrics.json')
    parser.add_argument('--visuals', default='artifacts/visuals.json')
    parser.add_argument('--model', default=None)
    parser.add_argument('--out', default='artifacts/report.pdf')
    args = parser.parse_args()

    metrics = load_json(Path(args.metrics))
    visuals = load_json(Path(args.visuals))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    make_pdf(metrics, visuals, out_path, model_path=args.model)
    print(f'Wrote PDF report to {out_path}')


if __name__ == '__main__':
    main()
