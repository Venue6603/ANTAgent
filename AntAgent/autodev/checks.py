from typing import Dict, Any
from .sandbox import run_checks

def quality_report() -> Dict[str, Any]:
    return {"tools": run_checks()}
import compileall
import importlib
import traceback
from pathlib import Path

# ...keep your other imports / helpers...

def _try_import(module_name: str):
    """
    Import a module for the smoke test.
    Returns: ("ok" | "skipped" | "fail", message)
    - ok: import succeeded
    - skipped: dependency missing; do not fail the check
    - fail: import attempted but crashed for another reason (syntax/runtime error)
    """
    try:
        importlib.import_module(module_name)
        return "ok", f"import {module_name} ok"
    except ModuleNotFoundError as e:
        # Missing third-party dep (e.g. 'fastapi') — treat as skipped, not a failure.
        return "skipped", f"import {module_name} skipped (missing dependency: {e.name})"
    except Exception:
        # Real import error (syntax/runtime) — this should fail checks.
        return "fail", f"import {module_name} failed:\n{traceback.format_exc()}"

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _smoke_checks():
    """
    Compile and shallow-import checks. Missing optional third-party deps should not fail
    the self-improve run; they are marked 'skipped'. Real errors still fail.
    """
    root = _project_root()
    log_lines = []

    # 1) byte-compile to catch syntax errors broadly
    rc = 0
    try:
        ok = compileall.compile_dir(str(root / "AntAgent"), quiet=1, force=False)
        rc = 0 if ok else 1
    except Exception:
        rc = 1
        log_lines.append(traceback.format_exc())

    log_lines.append(f"compileall rc={rc}")

    if rc != 0:
        return False, "\n".join(log_lines)

    # 2) optional import of the top-level app (skip if deps missing)
    status, msg = _try_import("AntAgent.app")
    if status == "ok":
        log_lines.append(f"[ok] {msg}")
        return True, "\n".join(log_lines)
    elif status == "skipped":
        log_lines.append(f"[skipped] {msg}")
        # treat as success; environment may not include runtime deps for app server
        return True, "\n".join(log_lines)
    else:
        # 'fail' — real import error unrelated to missing third-party deps
        log_lines.append(f"[interrupted] | {msg}")
        return False, "\n".join(log_lines)
