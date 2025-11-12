"""Train a baseline spam classifier using scikit-learn.

This script trains a simple Logistic Regression model on the downloaded CSV and
saves the trained model to `artifacts/`.
"""
from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import joblib


def load_data(path: Path):
    df = pd.read_csv(path, header=None)
    # dataset: first column = label (ham/spam), second column = text
    df = df.rename(columns={0: 'label', 1: 'text'})
    return df


def train(df: pd.DataFrame, out: Path):
    X = df['text'].astype(str)
    y = df['label'].map({'ham': 0, 'spam': 1})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Use SVM as the phase1 baseline (probability=True to allow probability output)
    pipeline = make_pipeline(TfidfVectorizer(), SVC(probability=True))
    pipeline.fit(X_train, y_train)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out)
    print(f"Saved model to {out}")
    return pipeline, (X_test, y_test)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/sms_spam.csv')
    p.add_argument('--out', default='artifacts/baseline_model.joblib')
    args = p.parse_args()
    df = load_data(Path(args.data))
    train(df, Path(args.out))


if __name__ == '__main__':
    main()
