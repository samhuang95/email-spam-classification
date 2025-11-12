"""Generate visualization data (ROC/PR/CM/top-words) and write to artifacts/visuals.json

Usage:
  python scripts/generate_visuals.py --model artifacts/baseline_model.joblib --data data/sms_spam.csv --out artifacts/visuals.json
"""
import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `src` can be imported when run from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.visualization import generate_visuals


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='artifacts/baseline_model.joblib')
    p.add_argument('--data', default='data/sms_spam.csv')
    p.add_argument('--out', default='artifacts/visuals.json')
    args = p.parse_args()
    visuals = generate_visuals(args.model, args.data, args.out)
    print('Wrote visuals to', args.out)


if __name__ == '__main__':
    main()
