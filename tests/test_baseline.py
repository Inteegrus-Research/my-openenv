from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_baseline_script_smoke() -> None:
    script = ROOT / "inference.py"
    if not script.exists():
        pytest.skip("inference.py not present yet")
    result = subprocess.run(
        ["python", str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    assert "[START]" in result.stdout
    assert "[END]" in result.stdout
