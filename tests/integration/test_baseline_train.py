import subprocess
from pathlib import Path


def test_baseline_train_creates_model(tmp_path):
    out = Path('artifacts/test_model.joblib')
    if out.exists():
        out.unlink()
    cmd = [
        ".venv\Scripts\python.exe",
        "src/train_baseline.py",
        "--data",
        "tests/fixtures/sample_sms.csv",
        "--out",
        str(out),
    ]
    subprocess.check_call(cmd)
    assert out.exists()
