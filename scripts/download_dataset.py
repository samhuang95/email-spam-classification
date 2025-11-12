#!/usr/bin/env python3
"""Download the SMS spam dataset and save to data/ directory.

Usage:
    python scripts/download_dataset.py --url <CSV_URL> --out data/sms_spam.csv
"""
import argparse
from pathlib import Path
import requests


def download(url: str, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url)
    r.raise_for_status()
    out.write_bytes(r.content)
    print(f"Downloaded {len(r.content)} bytes to {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=(
        "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
    ))
    p.add_argument("--out", default="data/sms_spam.csv")
    args = p.parse_args()
    download(args.url, Path(args.out))


if __name__ == "__main__":
    main()
