"""Simple reporting helpers for evaluation reports."""
def render_summary(metrics: dict) -> str:
    lines = ["# Evaluation Report", "", "## Metrics", ""]
    for k, v in metrics.items():
        if k != 'confusion_matrix':
            lines.append(f"- **{k}**: {v}")
    cm = metrics.get('confusion_matrix')
    if cm:
        lines.append("")
        lines.append("## Confusion Matrix")
        lines.append(f"- TN: {cm['tn']}")
        lines.append(f"- FP: {cm['fp']}")
        lines.append(f"- FN: {cm['fn']}")
        lines.append(f"- TP: {cm['tp']}")
    return "\n".join(lines)
