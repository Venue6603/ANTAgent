import subprocess, os, sys
from typing import Dict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ALLOWED_CMDS = [
    [sys.executable, "-m", "black", "--check", "--diff", "AntAgent"],
    [sys.executable, "-m", "isort", "--check-only", "--diff", "AntAgent"],
    [sys.executable, "-m", "ruff", "AntAgent"],
    [sys.executable, "-m", "mypy", "AntAgent"],
    [sys.executable, "-m", "pytest", "-q"],
    [sys.executable, "-m", "coverage", "run", "-m", "pytest", "-q"],
    [sys.executable, "-m", "bandit", "-q", "-r", "AntAgent"],
]

def run_checks() -> Dict[str, Dict[str, str]]:
    results: Dict[str, Dict[str, str]] = {}
    for cmd in ALLOWED_CMDS:
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, cwd=ROOT, timeout=300)
            results[" ".join(cmd)] = {"status": "ok", "output": out.decode("utf-8", "ignore")}
        except subprocess.CalledProcessError as e:
            results[" ".join(cmd)] = {"status": "fail", "output": e.output.decode("utf-8", "ignore")}
        except Exception as e:
            results[" ".join(cmd)] = {"status": "error", "output": str(e)}
    return results
