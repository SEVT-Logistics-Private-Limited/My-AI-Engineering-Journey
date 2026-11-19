import re
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_perception.py"


def test_renders_light_and_dark_perception_figures(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), "--out-dir", str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for name in ("perception.png", "perception_dark.png"):
        assert (tmp_path / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # The figure must show a real, accurate detection: error under 5 mm.
    errors = [float(e) for e in re.findall(r"localisation error ([\d.]+) mm", result.stdout)]
    assert len(errors) == 2 and all(e < 5.0 for e in errors)
