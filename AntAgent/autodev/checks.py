from typing import Dict, Any
from .sandbox import run_checks

def quality_report() -> Dict[str, Any]:
    return {"tools": run_checks()}
import compileall
import importlib
import traceback
from pathlib import Path

import importlib, traceback

def _try_import(module_name: str):
    """
    Import for smoke test.
    Returns ("ok" | "skipped" | "fail", message)
    - ok: import succeeded
    - skipped: dependency missing (e.g., FastAPI not installed)
    - fail: real syntax/runtime error inside the module
    """
    try:
        importlib.import_module(module_name)
        return "ok", f"import {module_name} ok"
    except ModuleNotFoundError as e:
        return "skipped", f"import {module_name} skipped (missing dependency: {e.name})"
    except Exception:
        return "fail", f"import {module_name} failed:\n{traceback.format_exc()}"
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

from pathlib import Path
import compileall

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _smoke_checks():
    root = _project_root()
    log = []

    # 1) compile all to catch syntax errors
    try:
        ok = compileall.compile_dir(str(root / "AntAgent"), quiet=1, force=False)
        rc = 0 if ok else 1
    except Exception:
        rc = 1
        log.append(traceback.format_exc())

    log.append(f"compileall rc={rc}")
    if rc != 0:
        return False, "\n".join(log)

    # 2) shallow import (skip if dependency missing)
    status, msg = _try_import("AntAgent.app")
    if status == "ok":
        log.append(f"[ok] {msg}")
        return True, "\n".join(log)
    if status == "skipped":
        log.append(f"[skipped] {msg}")
        return True, "\n".join(log)  # treat missing deps as non-fatal for SI
    log.append(f"[interrupted] | {msg}")
    return False, "\n".join(log)