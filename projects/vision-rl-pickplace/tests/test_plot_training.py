import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_training.py"


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_writes_light_and_dark_charts_and_table(tmp_path):
    eval_dir = tmp_path / "run" / "eval"
    eval_dir.mkdir(parents=True)
    np.savez(
        eval_dir / "evaluations.npz",
        timesteps=np.array([1000, 2000, 3000, 4000]),
        results=np.array([[-100.0] * 4, [-20.0] * 4, [-1.0] * 4, [-2.0] * 4]),   # best reward at 3000
        ep_lengths=np.array([[100] * 4, [30] * 4, [12] * 4, [13] * 4]),
        successes=np.array([[0] * 4, [1, 1, 0, 0], [1] * 4, [1] * 4], dtype=bool),
    )
    out = tmp_path / "assets"
    result = _run("--run", str(tmp_path / "run"), "--out-dir", str(out))
    assert result.returncode == 0, result.stderr
    assert "saved checkpoint at 3000 steps" in result.stdout

    for name in ("training_curve.png", "training_curve_dark.png"):
        assert (out / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    rows = list(csv.DictReader((out / "training_curve.csv").open()))
    assert [r["saved_checkpoint"] for r in rows] == ["0", "0", "1", "0"]
    assert rows[1]["success_rate"] == "0.50"


def test_missing_log_fails_with_a_clear_message(tmp_path):
    result = _run("--run", str(tmp_path))
    assert result.returncode != 0
    assert "no evaluation log" in result.stderr


def test_place_task_writes_separate_files(tmp_path):
    eval_dir = tmp_path / "run" / "eval"
    eval_dir.mkdir(parents=True)
    np.savez(
        eval_dir / "evaluations.npz",
        timesteps=np.array([50000, 100000]),
        results=np.array([[-20.0] * 2, [20.0] * 2]),
        ep_lengths=np.array([[200] * 2, [40] * 2]),
        successes=np.array([[0, 0], [1, 1]], dtype=bool),
    )
    out = tmp_path / "assets"
    result = _run("--run", str(tmp_path / "run"), "--out-dir", str(out), "--task", "place")
    assert result.returncode == 0, result.stderr
    assert {f.name for f in out.iterdir()} == {
        "training_curve_place.png", "training_curve_place_dark.png", "training_curve_place.csv"}
