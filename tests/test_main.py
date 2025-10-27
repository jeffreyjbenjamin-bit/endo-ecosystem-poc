import subprocess
import sys
from pathlib import Path


import subprocess
import sys
from pathlib import Path

def test_main_runs_and_prints_greeting():
    proc = subprocess.run(
        [sys.executable, str(Path("src") / "main.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    out = proc.stdout
    assert "Hello from the Endo Ecosystem PoC!" in out
    assert "Python version:" in out
