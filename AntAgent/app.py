import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

try:
    from ollama_adapter import Llama
except Exception:
    Llama = None  # type: ignore

from AntAgent.models import (
    Project, PlanDraft, EvidenceTable, SequenceCheckReport,
    BOMSpec, BOMTable, ComplianceChecklist, TimelineSpec, Timeline,
    ReportRequest, ReportBundle
)
from AntAgent.planning.planner import plan_from_goal
from AntAgent.rag.ingest import ingest_pdfs
from AntAgent.rag.extract import extract_facts
from AntAgent.sequence.checks import check_sequences
from AntAgent.costing.bom import build_bom
from AntAgent.planning.compliance import build_compliance
from AntAgent.planning.timeline import build_timeline
from AntAgent.reporting.reporter import package_report
from AntAgent.guards import enforce_non_operational

_LAST_SELF_IMPROVE_DIAG: dict | None = None




# Storage for chat transcripts
_CHAT_DIR = Path(".antagent")
_CHAT_DIR.mkdir(parents=True, exist_ok=True)
_CHAT_PATH = _CHAT_DIR / "chat_history.jsonl"


# Global toggles (define once)
try:
    _AUTOSI_ENABLED
except NameError:
    _AUTOSI_ENABLED = bool(int(os.getenv("ANT_AUTOSI", "0")))

try:
    _AUTOSI_PERIOD_SEC
except NameError:
    _AUTOSI_PERIOD_SEC = int(os.getenv("ANT_AUTOSI_PERIOD_SEC", "300"))


_AUTOSI_ENABLED = bool(int(os.getenv("ANT_AUTOSI", "0")))   # 1 to enable at startup
_AUTOSI_PERIOD_SEC = int(os.getenv("ANT_AUTOSI_PERIOD_SEC", "300"))  # 5 min default

model_path = os.getenv("ANT_LLAMA_MODEL_PATH")

# Tune these via env vars without touching code
ctx        = int(os.getenv("ANT_LLAMA_CTX", "4096"))      # keep <= model's train ctx unless you know what you're doing
batch      = int(os.getenv("ANT_LLAMA_N_BATCH", "656"))   # smaller uses less VRAM
gpu_layers = int(os.getenv("ANT_LLAMA_N_GPU_LAYERS", "28"))  # try 20 first on a 3080 if ~3.5–4 GB VRAM free

llm = Llama(
    model_path=model_path,
    n_ctx=ctx,
    n_batch=batch,
    n_gpu_layers=gpu_layers,  # <-- this is the key
    verbose=True,
)


# self-improvement
app = FastAPI(title="AntAgent — Planning & Protocol Assistant")


_DIFF_HUNK = re.compile(r"^@@\s*-\d+(?:,\d+)?\s*\+\d+(?:,\d+)?\s*@@", re.M)

def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()


def _normalize_targets_to_allowlist(targets: list[str], allowed: list[str]) -> list[str]:
    """
    Map short/basename or mixed-style targets to exact allowlisted paths.
    Keeps only items that resolve uniquely.
    """
    allowed_norm = [p.replace("\\", "/") for p in allowed]
    by_base: dict[str, list[str]] = {}
    for p in allowed_norm:
        base = p.rsplit("/", 1)[-1]
        by_base.setdefault(base, []).append(p)

    resolved: list[str] = []
    for t in (targets or []):
        t_norm = t.replace("\\", "/")
        if t_norm in allowed_norm:
            resolved.append(t_norm)
            continue
        base = t_norm.rsplit("/", 1)[-1]
        cand = by_base.get(base, [])
        if len(cand) == 1:   # only accept unique basename matches
            resolved.append(cand[0])
        # if 0 or >1, drop it (ambiguous or unknown)
    return resolved

def _all_targets_allowed(diff_text: str) -> list[str]:
    targets = _diff_targets(diff_text)
    if not targets:
        raise ValueError("No valid targets in diff")

    # NEW: allow writes to any file when ANT_SI_WRITE_ALL=1
    if bool(int(os.getenv("ANT_SI_WRITE_ALL", "0"))):
        # normalize slashes only; no allowlist gating
        return [t.replace("\\", "/") for t in targets]

    # default: enforce allowlist (use local _allowed_paths)
    allowed = _allowed_paths()
    allowed_norm = {p.replace("\\", "/") for p in (allowed or [])}
    targets_norm = [t.replace("\\", "/") for t in targets]
    if not set(targets_norm).issubset(allowed_norm):
        bad = [t for t in targets_norm if t not in allowed_norm]
        raise ValueError(f"Path not allowed for self-edit: {', '.join(bad)}")
    return targets_norm

def _validate_diff_paths(diff_text: str, allowed_paths: list[str]) -> bool:
    """Allow changes only to the whitelisted paths."""
    allowed = set(allowed_paths or [])
    if not allowed:
        return False
    touched = set()
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            # format: diff --git a/<path> b/<path>
            parts = line.split()
            if len(parts) >= 4 and parts[2].startswith("a/") and parts[3].startswith("b/"):
                p = parts[3][2:]  # strip 'b/'
                touched.add(p)
    return bool(touched) and touched.issubset(allowed)

def _diff_has_top_insert(diff_text: str) -> bool:
    """Heuristic: '+' lines appear before any context/hunk in the header area."""
    if not _DIFF_HUNK.search(diff_text):
        return False
    pre_hunk = []
    for ln in diff_text.splitlines():
        if ln.startswith("@@"):
            break
        pre_hunk.append(ln)
    return any(ln.startswith("+") for ln in pre_hunk)

def _diff_has_required_anchors(diff_text: str, anchors: list[str]) -> bool:
    """Require that at least one anchor appears as a context line in some hunk."""
    if not anchors:
        return False
    ok = False
    in_hunk = False
    for ln in diff_text.splitlines():
        if ln.startswith("@@"):
            in_hunk = True
            continue
        if in_hunk and ln.startswith(" "):  # context
            raw = ln[1:]
            if any(a in raw for a in anchors):
                ok = True
    return ok

def _file_hashes(paths: list[str]) -> dict:
    out = {}
    for p in paths:
        try:
            txt = Path(p).read_text(encoding="utf-8", errors="replace")
            out[p] = {"sha256": _sha256_text(txt), "size": len(txt)}
        except Exception:
            out[p] = {"sha256": None, "size": None}
    return out

AUTOLOG_DIR = Path(".autodev/_self_improve")
AUTOLOG_DIR.mkdir(parents=True, exist_ok=True)

def _autolog_start(payload: dict) -> Dict[str, Any]:
    run = {
        "ts": time.time(),
        "goal": payload.get("goal"),
        "constraints": payload.get("constraints", {}),
        "rounds": payload.get("rounds", 1),
        "do_apply": payload.get("do_apply", False),
        "steps": [],
        "status": "started",
    }
    return run

def _autolog_step(run: dict, **kv):
    run["steps"].append(kv)

def _autolog_finish(run: dict, status: str, extra: Optional[dict] = None):
    run["status"] = status
    if extra:
        run.update(extra)
    p = AUTOLOG_DIR / f"run_{int(run['ts'])}.json"
    p.write_text(json.dumps(run, indent=2), encoding="utf-8")

def _file_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""

def _infer_si_anchor(file_text: str) -> Dict[str, str]:
    """
    Heuristics to extract anchors for the classic SI-docstring edit without precise prompt.
    Falls back to generic-safe anchors.
    """
    anchors = {}
    # try to find list_paths definition
    m = re.search(r"^\s*def\s+list_paths\s*\(", file_text, flags=re.M)
    if m:
        anchors["near_def"] = "def list_paths():"
    # try to find current SI version string
    s = re.search(r'Llama test successful\.\s*\(SI v(\d+)\)', file_text)
    if s:
        anchors["current_line"] = f"\"\"\"Llama test successful. (SI v{s.group(1)})\"\"\""
    # robust generic anchors
    if not anchors:
        anchors["module"] = "\"\"\""
    return anchors

def _enrich_constraints(payload: dict) -> dict:
    c = dict(payload.get("constraints") or {})
    # hard safety defaults
    c.setdefault("paths", [])
    c["allow_lenient"] = False
    c["allow_top_insert"] = False
    c.setdefault("require_context_lines", 2)
    # add anchors if missing
    paths: List[str] = c.get("paths") or []
    if paths:
        p = Path(paths[0])
        txt = _file_text(p)
        anchors = _infer_si_anchor(txt)
        # only add if caller didn't specify
        c.setdefault("must_anchor_any", [])
        for a in anchors.values():
            if a not in c["must_anchor_any"]:
                c["must_anchor_any"].append(a)
    return c
# --- Lightweight code browsing helpers (no manager.py dependency) ---

try:
    from ollama_adapter import Llama  # noqa: F401
    _LLAMA_IMPORT_OK = True
except Exception:
    _LLAMA_IMPORT_OK = False

import time
from fastapi import Request

@app.middleware("http")
async def _log_requests(request: Request, call_next):
    t0 = time.time()
    print(f"[HTTP] {request.method} {request.url.path}", flush=True)
    try:
        resp = await call_next(request)
        dt = (time.time() - t0)*1000
        print(f"[HTTP] -> {resp.status_code} in {dt:.1f} ms", flush=True)
        return resp
    except Exception as e:
        dt = (time.time() - t0)*1000
        print(f"[HTTP] !! ERROR after {dt:.1f} ms: {type(e).__name__}: {e}", flush=True)
        raise

@app.get("/debug/llama")
async def debug_llama():
    """
    Quick check for local LLaMA availability.
    Returns: {available, llama_lib_imported, model_path, reason}
    """
    model_path = os.getenv("ANT_LLAMA_MODEL_PATH")
    out = {
        "available": False,
        "llama_lib_imported": _LLAMA_IMPORT_OK,
        "model_path": model_path,
        "reason": None,
    }

    if not _LLAMA_IMPORT_OK:
        out["reason"] = "llama_cpp not importable"
        return out

    if not model_path or not os.path.exists(model_path):
        out["reason"] = "ANT_LLAMA_MODEL_PATH not set or file not found"
        return out

    # Light sanity-check: try creating a small context LLaMA instance.
    try:
        Llama(model_path=model_path, n_ctx=512, n_threads=4)  # loads model
        out["available"] = True
        return out
    except Exception as e:
        out["reason"] = str(e)
        return out


import sys, importlib, site


@app.get("/debug/python")
def debug_python():
    info = {
        "python_executable": sys.executable,
        "sys_path_head": sys.path[:5],
        "site_packages": site.getsitepackages() if hasattr(site, "getsitepackages") else [],
        "venv": os.environ.get("VIRTUAL_ENV"),
    }
    try:
        m = importlib.import_module("llama_cpp")
        info["llama_cpp_imported"] = True
        info["llama_cpp_file"] = getattr(m, "__file__", None)
        info["llama_cpp_libdir"] = os.path.join(os.path.dirname(m.__file__), "lib")
        info["lib_contents"] = os.listdir(info["llama_cpp_libdir"]) if os.path.isdir(info["llama_cpp_libdir"]) else []
    except Exception as e:
        info["llama_cpp_imported"] = False
        info["llama_cpp_error"] = str(e)
        info["llama_cpp_traceback"] = traceback.format_exc()
    return info

@app.get("/debug/llama2")
async def debug_llama2():
    """Shows whether the wheel actually shipped a llama.dll and where it is."""
    import os, sys, glob
    site = next(p for p in sys.path if p.endswith("\\site-packages"))
    lib_dir = os.path.join(site, "llama_cpp", "lib")
    dlls = glob.glob(os.path.join(lib_dir, "*.dll"))
    return {
        "site_packages": site,
        "lib_dir": lib_dir,
        "dlls": dlls,
        "exists": os.path.isdir(lib_dir),
    }

@app.get("/debug/env")
async def debug_env():
    import sys, os, platform
    return {
        "python_executable": sys.executable,
        "venv_is_.venv311": sys.executable.lower().endswith(r"\.venv311\scripts\python.exe"),
        "cwd": os.getcwd(),
        "platform": platform.platform(),
        "first_sys_path": sys.path[:5],
        "env": {
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV"),
            "PATH_head": os.environ.get("PATH", "").split(";")[:6],
        },
    }

# ---------------------------------------------------------------------------
def _project_root() -> Path:
    # .../ANTAgent (repo root)
    return Path(__file__).resolve().parents[1]

def _allowlist_path() -> Path:
    return _project_root() / "AntAgent" / "autodev" / "allowlist.txt"

def _allowed_paths() -> list[str]:
    p = _allowlist_path()
    if not p.exists():
        return []
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]

@app.get("/code/list_paths")
def code_list_paths():
    """
            Returns allowlisted project paths that currently exist on disk.



    """
    repo_root = _project_root()
    existing: list[str] = []
    for rel in _allowed_paths():
        abs_path = (repo_root / rel).resolve()
        try:
            if abs_path.exists():
                existing.append(rel.replace("\\", "/"))
        except Exception:
            # ignore unreadable entries
            pass
    return JSONResponse({"paths": existing})

@app.post("/code/read_files")
async def code_read_files(request: Request):
    """
    Body: { "paths": ["AntAgent/autodev/manager.py", ...] }
    Returns { "files": { "path": "content", ... } }
    Only files that are both allowlisted and present on disk are returned.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    paths = payload.get("paths")
    if not isinstance(paths, list) or not paths:
        raise HTTPException(status_code=400, detail="paths must be a non-empty list")

    allowed = set(_allowed_paths())
    repo_root = _project_root()
    out: dict[str, str] = {}
    for rel in paths:
        norm = str(rel).replace("\\", "/")
        if norm not in allowed:
            # silently skip non-allowed paths
            continue
        abs_path = (repo_root / norm).resolve()
        try:
            text = abs_path.read_text(encoding="utf-8")
            if len(text) > 500_000:
                text = text[:500_000] + "\n\n# [TRUNCATED]"
            out[norm] = text
        except Exception:
            # skip unreadable files
            pass

    return JSONResponse({"files": out})

@app.post("/plan/draft", response_model=PlanDraft)
def draft_plan(project: Project) -> PlanDraft:
    enforce_non_operational(project.objective)
    return plan_from_goal(project)

@app.post("/literature/ingest")
def literature_ingest(files: List[UploadFile] = File(...)) -> dict:
    return {"ingested": ingest_pdfs(files)}

@app.post("/literature/extract", response_model=EvidenceTable)
def literature_extract(spec: dict) -> EvidenceTable:
    return extract_facts(spec)

@app.post("/sequence/check", response_model=SequenceCheckReport)
def sequence_check(files: List[UploadFile] = File(...)) -> SequenceCheckReport:
    return check_sequences(files)

@app.post("/bom/build", response_model=BOMTable)
def bom_build(spec: BOMSpec) -> BOMTable:
    return build_bom(spec)

@app.post("/compliance/check", response_model=ComplianceChecklist)
def compliance_check(project: Project) -> ComplianceChecklist:
    return build_compliance(project)

@app.post("/timeline/build", response_model=Timeline)
def timeline_build(spec: TimelineSpec) -> Timeline:
    return build_timeline(spec)

@app.post("/report/package", response_model=ReportBundle)
def report_package(req: ReportRequest) -> ReportBundle:
    enforce_non_operational(req.objective)
    return package_report(req)

# ===== Self-Improvement Endpoints =====
@app.post("/code/list")
def code_list() -> dict:
    return {"paths": list_paths()}

@app.post("/code/read")
def code_read(payload: dict) -> dict:
    return {"files": read_files(payload.get("paths", []))}

@app.post("/code/propose_patch")
def code_propose_patch(payload: dict) -> dict:
    patch_txt = propose_patch(payload.get("goal", ""), payload.get("constraints", {}))
    return {"unified_diff": patch_txt}

@app.post("/code/run_checks")
def code_run_checks() -> dict:
    return run_quality_checks()

from hashlib import sha256
import os, re

def _diff_targets(udiff_text: str) -> list[str]:
    # Accept both "diff --git a/X b/X" and "diff --git X X"
    pats = re.findall(r"^diff --git\s+(?:a/)?([^\s]+)\s+(?:b/)?\1", udiff_text, flags=re.M)
    # De-dup and keep only repo-relative paths that actually exist (best-effort)
    seen = set()
    out = []
    for p in pats:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out or []  # may be empty; we still proceed

def _file_sha256(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return sha256(f.read()).hexdigest()
    except Exception:
        return None

@app.post("/code/apply_patch")
def code_apply_patch(payload: dict) -> dict:
    # ---- safe event emitter (no-op if _emit_event is not available) ----
    _emit = globals().get("_emit_event")
    if not callable(_emit):
        def _emit_event(_evt: dict):  # type: ignore
            return None
    else:
        _emit_event = _emit  # type: ignore

    import time  # local import to avoid top-level changes

    udiff = payload.get("unified_diff", "")
    if not udiff:
        _emit_event({
            "type": "code.apply_patch.error",
            "error": "No diff provided",
            "ts": time.time(),
        })
        return {"applied": False, "error": "No diff provided"}

    # Normalize and ensure bytes for the underlying applier
    if isinstance(udiff, (bytes, bytearray)):
        udiff_b = bytes(udiff)
        udiff_text = udiff.decode("utf-8", errors="replace")
    else:
        udiff_text = str(udiff).replace("\r\n", "\n")
        udiff_b = udiff_text.encode("utf-8")

    # Parse targets & snapshot hashes before
    targets = _diff_targets(udiff_text)
    allowed = _allowed_paths() or []
    targets = _normalize_targets_to_allowlist(targets, allowed)
    before = {p: _file_sha256(p) for p in targets}

    _emit_event({
        "type": "code.apply_patch.start",
        "targets": targets,
        "diff_len": len(udiff_b),
        "ts": time.time(),
    })

    # Apply (STRICT – no fallback here)
    try:
        from AntAgent.autodev.manager import apply_patch as _mgr_apply
        _emit_event({
            "type": "code.apply_patch.attempt",
            "targets": targets,
            "ts": time.time(),
        })
        res = _mgr_apply(udiff)  # bytes or str is fine; manager normalizes
    except Exception as e:
        msg = f"Patch apply failed: {e}"
        _emit_event({
            "type": "code.apply_patch.error",
            "error": msg,
            "ts": time.time(),
        })
        return {"applied": False, "error": msg}

    # If manager reports not applied, surface its reason
    if not res.get("applied", False):
        why = res.get("message", "Patch apply failed")
        _emit_event({
            "type": "code.apply_patch.result",
            "applied": False,
            "reason": why,
            "ts": time.time(),
        })
        return {"applied": False, "error": why}

    # Verify content actually changed
    after = {p: _file_sha256(p) for p in targets}
    changed = False
    for p in targets:
        if p in before or os.path.exists(p):
            if before.get(p) != after.get(p):
                changed = True
                break

    if not changed:
        _emit_event({
            "type": "code.apply_patch.result",
            "applied": False,
            "reason": "Patch reported success but no file changed",
            "ts": time.time(),
        })
        return {"applied": False, "error": "Patch reported success but no file changed"}

    _emit_event({
        "type": "code.apply_patch.result",
        "applied": True,
        "changed_targets": targets,
        "ts": time.time(),
    })

    return {"applied": True, "error": None, "message": "Patch applied"}

@app.post("/code/preview_patch")
def code_preview_patch(payload: dict) -> dict:
    goal = payload.get("goal", "")
    constraints = payload.get("constraints", {}) or {}
    # Reuse the same engine the system uses to propose a patch
    from AntAgent.autodev.manager import propose_patch
    udiff = (propose_patch(goal, constraints) or "").replace("\r\n", "\n")
    return {"unified_diff": udiff}


import traceback

# Update the import section in app.py to include the new functions:
from AntAgent.autodev.manager import (
    propose_patch,
    self_improve_round,
    self_improve_with_retry,  # Add this
    validate_diff_safety,  # Add this
    run_quality_checks,
    list_paths,
    read_files,
)


# Replace the existing /self_improve endpoint with this enhanced version:

@app.post("/self_improve")
async def self_improve(request: Request):
    """
    Enhanced self-improvement endpoint with explanations and retry logic.

    Body: {
        "goal": "specific change to make",
        "constraints": {
            "paths": ["file1.py", "file2.py"],
            "require_context_lines": 3,
            "must_anchor_any": ["def function_name", "class ClassName"]
        },
        "max_attempts": 3,
        "use_retry": true  # Set to true to use the new retry system
    }
    """
    _emit = globals().get("_emit_event")
    if not callable(_emit):
        def _emit_event(_):
            return None
    else:
        _emit_event = _emit

    try:
        payload = await request.json()
        goal = (payload.get("goal") or "").strip()
        constraints = (payload.get("constraints") or {})
        max_attempts = int(payload.get("max_attempts", 3))
        use_retry = bool(payload.get("use_retry", True))  # Default to new system

        if not goal:
            raise HTTPException(status_code=400, detail="Field 'goal' is required")

        # Ensure paths are specified
        paths = constraints.get("paths") or []
        if isinstance(paths, str):
            paths = [paths]
        if not paths:
            raise HTTPException(
                status_code=400,
                detail="'constraints.paths' must specify target files"
            )
        constraints["paths"] = paths

        # Add safety defaults
        constraints.setdefault("require_context_lines", 3)
        constraints.setdefault("no_top_insert", True)
        constraints.setdefault("allow_lenient", False)
        constraints.setdefault("allow_top_insert", False)

        _emit_event({
            "type": "self_improve.start",
            "goal": goal,
            "constraints": constraints,
            "max_attempts": max_attempts,
            "use_retry": use_retry,
            "ts": time.time()
        })

        if use_retry:
            # Use the new enhanced retry system with explanations
            result = self_improve_with_retry(goal, constraints, max_attempts)

            _emit_event({
                "type": "self_improve.complete",
                "success": result["success"],
                "attempts": len(result.get("attempts", [])),
                "explanation": result.get("explanation", ""),
                "ts": time.time()
            })

            return JSONResponse(result)
        else:
            # Use the original single-attempt system (backward compatibility)
            before = _file_hashes(paths)

            res = self_improve_round(goal, constraints, do_apply=True)

            # Normalize the response
            if isinstance(res, tuple):
                summary, diff = res
                res = {
                    "summary": summary,
                    "unified_diff": diff or "",
                    "applied": bool(diff),
                    "message": "Patch applied" if diff else "No diff produced"
                }

            after = _file_hashes(paths)
            changed = (before != after)

            _emit_event({
                "type": "self_improve.done",
                "rounds_completed": 1,
                "changed": changed,
                "ts": time.time()
            })

            # Format response to match new structure
            return JSONResponse({
                "success": res.get("applied", False),
                "attempts": [{
                    "attempt": 1,
                    "status": "applied" if res.get("applied") else "failed",
                    "explanation": res.get("summary", ""),
                    "diff": res.get("unified_diff", "")
                }],
                "final_diff": res.get("unified_diff", "") if res.get("applied") else None,
                "explanation": res.get("message", ""),
                "diagnostics": res.get("diagnostics", {})
            })

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "trace": traceback.format_exc(),
                "where": "self_improve"
            }
        )


@app.post("/self_improve/validate_diff")
async def validate_diff_endpoint(request: Request):
    """
    Validate a diff before applying it.

    Body: {
        "diff": "unified diff text",
        "constraints": {...}
    }

    Returns: {
        "safe": bool,
        "issues": [...]
    }
    """
    try:
        payload = await request.json()
        diff = payload.get("diff", "")
        constraints = payload.get("constraints", {})

        result = validate_diff_safety(diff, constraints)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/self_improve/explain_last")
async def explain_last_improvement():
    """
    Get explanation for the last self-improvement attempt.
    """
    from AntAgent.autodev import manager
    diag = manager.get_last_self_improve_diag()

    # Try to get explanation from diagnostics or return helpful message
    explanation = diag.get("explanation", None)
    if not explanation and diag.get("message"):
        explanation = f"Last operation: {diag['message']}"
    elif not explanation:
        explanation = "No explanation available for the last operation"

    return JSONResponse({
        "diagnostics": diag,
        "explanation": explanation
    })



from pathlib import Path  # if not already imported
from fastapi import HTTPException  # if not already imported
from fastapi.responses import JSONResponse  # if not already imported

@app.get("/status")
async def status():
    import os
    # Check OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_available = bool(openai_key)

    # Check llama-cpp model path + importability
    llama_model_path = os.getenv("ANT_LLAMA_MODEL_PATH")
    try:
        from ollama_adapter import Llama  # may raise if not installed
        llama_lib_imported = True
    except Exception:
        llama_lib_imported = False

    return {
        "openai_available": openai_available,
        "llama_lib_imported": llama_lib_imported,
        "llama_model_path": llama_model_path,
        "cwd": str(Path.cwd()),
    }


@app.get("/debug/openai")
async def debug_openai():
    import os, json, requests
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return {"ok": False, "reason": "OPENAI_API_KEY not set in server process"}
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            data=json.dumps({
                "model": "gpt-5",
                "messages": [
                    {"role": "system", "content": "You only reply with 'pong'."},
                    {"role": "user", "content": "ping"}
                ],
                "temperature": 0
            }),
            timeout=30
        )
        return {
            "ok": resp.ok,
            "status": resp.status_code,
            "text": (resp.json().get("choices",[{}])[0].get("message",{}).get("content","") if resp.headers.get("content-type","").startswith("application/json") else resp.text)
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/code/replace_once")
def code_replace_once(payload: dict) -> dict:
    """
    Safely replace the first occurrence of old_substr with new_substr in a single allowlisted file.
    Body: {"path": "...", "old_substr": "...", "new_substr": "..."}
    Returns: {"changed": bool, "count": int}
    """
    path = payload.get("path")
    old = payload.get("old_substr")
    new = payload.get("new_substr")

    if not isinstance(path, str) or not isinstance(old, str) or not isinstance(new, str):
        raise HTTPException(status_code=400, detail="path, old_substr, new_substr are required strings")

    # allowlist enforcement (reuse your existing helpers)
    repo_root = _project_root()
    allowed = set(_allowed_paths())
    # Normalize slashes for comparison
    norm_rel = path.replace("\\", "/")
    if norm_rel not in allowed:
        raise HTTPException(status_code=400, detail=f"path not allowlisted: {norm_rel}")

    abs_path = (repo_root / norm_rel).resolve()
    if not abs_path.exists():
        raise HTTPException(status_code=404, detail=f"file not found: {abs_path}")

    text = abs_path.read_text(encoding="utf-8", errors="replace")
    idx = text.find(old)
    if idx < 0:
        return {"changed": False, "count": 0}

    new_text = text[:idx] + new + text[idx+len(old):]
    abs_path.write_text(new_text, encoding="utf-8")
    return {"changed": True, "count": 1}
@app.get("/debug/self_improve_last")
def debug_self_improve_last():
    # import inside the function so we don't alter module-level imports
    from .autodev import manager
    return manager.get_last_self_improve_diag()

# --- Debug: analyze a diff payload without applying ---------------------------
@app.post("/debug/diagnose_diff")
async def debug_diagnose_diff(payload: dict):
    """
    Body: { "unified_diff": "<diff text>", "paths": ["AntAgent/autodev/manager.py", ...] }
    Returns the same diagnostics diagnose_diff_failure() produces.
    """
    from .autodev import manager
    diff_text = (payload or {}).get("unified_diff") or ""
    paths = (payload or {}).get("paths") or []
    return manager.diagnose_diff_failure(diff_text, paths)

@app.get("/debug/dashboard", response_class=HTMLResponse)
def debug_dashboard():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>ANT Debug Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root { --bg:#0f172a; --panel:#111827; --fg:#e5e7eb; --muted:#94a3b8; --ok:#22c55e; --warn:#f59e0b; --err:#ef4444; --accent:#60a5fa; }
  body { margin:0; font:14px/1.45 system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background:var(--bg); color:var(--fg); }
  header { padding:12px 16px; background:#0b1220; border-bottom:1px solid #1f2937; display:flex; gap:10px; align-items:center; }
  h1 { font-size:16px; margin:0; color:var(--fg); }
  .badge { font-size:12px; padding:2px 8px; border-radius:999px; background:#1f2937; color:var(--muted); border:1px solid #263244; }
  main { padding:16px; display:grid; gap:16px; grid-template-columns: 1.2fr 1fr; }
  section { background:var(--panel); border:1px solid #1f2937; border-radius:12px; overflow:hidden; }
  section header { background:#0f1626; border-bottom:1px solid #1f2937; }
  section header h2 { margin:0; font-size:14px; color:var(--accent); }
  .pad { padding:12px 14px; }
  .kvs { display:grid; grid-template-columns: 140px 1fr; gap:6px 12px; }
  .kvs div.key { color:var(--muted); }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
  .log { max-height: 380px; overflow:auto; font-size:13px; }
  .row { padding:8px 10px; border-bottom:1px solid #1f2937; display:flex; gap:10px; align-items:flex-start; }
  .row:last-child { border-bottom:none; }
  .ts { color:var(--muted); min-width: 130px; font-variant-numeric: tabular-nums; }
  .tag { font-size:11px; padding:2px 6px; border-radius:6px; border:1px solid #2a364a; color:#cbd5e1; background:#172035; }
  .ok { color:var(--ok); border-color:#1f5132; background:#0f2b1b; }
  .err { color:var(--err); border-color:#4d1f27; background:#2a1013; }
  .warn { color:var(--warn); border-color:#4a3a19; background:#261d0a; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  .muted { color:var(--muted); }
  .small { font-size:12px; }
  code { background:#0b1220; padding:1px 4px; border-radius:6px; border:1px solid #1f2937; }
</style>
</head>
<body>
  <header>
    <h1>ANT Debug Dashboard</h1>
    <span class="badge" id="conn">connecting…</span>
    <span class="badge">/events</span>
  </header>
  <main>
    <section>
      <header class="pad"><h2>Self-Improve (live)</h2></header>
      <div class="pad">
        <div class="kvs small">
          <div class="key">Rounds</div><div id="siRounds">0</div>
          <div class="key">Last Goal</div><div id="siGoal" class="mono muted">—</div>
          <div class="key">Last Engine</div><div id="siEngine" class="mono muted">—</div>
          <div class="key">Last Result</div><div id="siResult" class="mono muted">—</div>
          <div class="key">Last Diff Bytes</div><div id="siBytes" class="mono muted">—</div>
        </div>
      </div>
    </section>

    <section>
      <header class="pad"><h2>Patch Apply (live)</h2></header>
      <div class="pad">
        <div class="kvs small">
          <div class="key">Targets</div><div id="paTargets" class="mono muted">—</div>
          <div class="key">Diff Bytes</div><div id="paBytes" class="mono muted">—</div>
          <div class="key">Last Result</div><div id="paResult" class="mono muted">—</div>
          <div class="key">Reason</div><div id="paReason" class="mono muted">—</div>
        </div>
      </div>
    </section>

    <section style="grid-column: 1 / -1;">
      <header class="pad"><h2>Event Timeline</h2></header>
      <div id="log" class="log"></div>
    </section>
  </main>

<script>
(function () {
  const conn = document.getElementById('conn');
  const log  = document.getElementById('log');

  // Self-improve widgets
  const siRounds = document.getElementById('siRounds');
  const siGoal   = document.getElementById('siGoal');
  const siEngine = document.getElementById('siEngine');
  const siResult = document.getElementById('siResult');
  const siBytes  = document.getElementById('siBytes');

  // Patch-apply widgets
  const paTargets = document.getElementById('paTargets');
  const paBytes   = document.getElementById('paBytes');
  const paResult  = document.getElementById('paResult');
  const paReason  = document.getElementById('paReason');

  function ts() {
    const d = new Date();
    return d.toLocaleTimeString([], {hour12:false}) + '.' + String(d.getMilliseconds()).padStart(3,'0');
  }

  function addRow({ tag, text, tone }) {
    const row = document.createElement('div');
    row.className = 'row';
    const t = document.createElement('div');
    t.className = 'ts'; t.textContent = ts();

    const b = document.createElement('div');
    const badge = document.createElement('span');
    badge.className = 'tag ' + (tone || '');
    badge.textContent = tag;
    const span = document.createElement('span');
    span.style.marginLeft = '8px';
    span.innerHTML = text;
    b.appendChild(badge); b.appendChild(span);

    row.appendChild(t); row.appendChild(b);
    log.appendChild(row);
    log.scrollTop = log.scrollHeight;
  }

  function fmtJSON(obj) {
    try { return '<code class="mono">' + JSON.stringify(obj) + '</code>'; }
    catch { return '<code class="mono">[unserializable]</code>'; }
  }

  function handleEvent(evt) {
    let data;
    try { data = JSON.parse(evt.data); } catch { return; }
    const type = data.type || 'unknown';

    // ---- Self improve events ----
    if (type === 'self_improve.start') {
      siGoal.textContent = data.goal || '—';
      siEngine.textContent = '—';
      siResult.textContent = 'running…';
      siBytes.textContent  = '—';
      addRow({ tag: 'SELF', text: 'start ' + fmtJSON({goal: data.goal}), tone: '' });
    }
    else if (type === 'self_improve.progress') {
      // data.stage: 'prompt', 'llama', 'openai', etc.
      if (data.engine) siEngine.textContent = data.engine;
      addRow({ tag: 'SELF', text: 'progress ' + fmtJSON(data), tone: '' });
    }
    else if (type === 'self_improve.result') {
      const ok = !!data.applied;
      siResult.textContent = ok ? 'applied' : 'not applied';
      siEngine.textContent = data.engine || siEngine.textContent;
      siBytes.textContent  = (data.diff_bytes != null) ? String(data.diff_bytes) : '—';
      siRounds.textContent = String((parseInt(siRounds.textContent||'0',10) || 0) + 1);
      addRow({ tag: 'SELF', text: (ok ? '✅ ' : '❌ ') + (data.message || ''), tone: ok ? 'ok' : 'err' });
    }
    else if (type === 'self_improve.error') {
      siResult.textContent = 'error';
      addRow({ tag: 'SELF', text: 'error ' + fmtJSON({error: data.error}), tone: 'err' });
    }

    // ---- Code apply patch events ----
    else if (type === 'code.apply_patch.start') {
      paTargets.textContent = (data.targets || []).join(', ') || '—';
      paBytes.textContent   = (data.diff_len != null) ? String(data.diff_len) : '—';
      paResult.textContent  = 'running…';
      paReason.textContent  = '—';
      addRow({ tag: 'PATCH', text: 'start ' + fmtJSON({targets: data.targets, bytes: data.diff_len}), tone: '' });
    }
    else if (type === 'code.apply_patch.attempt') {
      addRow({ tag: 'PATCH', text: 'attempt', tone: '' });
    }
    else if (type === 'code.apply_patch.result') {
      const ok = !!data.applied;
      paResult.textContent = ok ? 'applied' : 'not applied';
      paReason.textContent = data.reason ? String(data.reason) : '—';
      addRow({ tag: 'PATCH', text: (ok ? '✅ ' : '❌ ') + (data.reason || ''), tone: ok ? 'ok' : 'warn' });
    }
    else if (type === 'code.apply_patch.error') {
      paResult.textContent = 'error';
      paReason.textContent = data.error || '—';
      addRow({ tag: 'PATCH', text: 'error ' + fmtJSON({error: data.error}), tone: 'err' });
    }

    // ---- Anything else ----
    else {
      addRow({ tag: type, text: fmtJSON(data), tone: '' });
    }
  }

  // Connect
  const es = new EventSource("/events");
  es.onopen = () => { conn.textContent = "connected"; };
  es.onerror = () => { conn.textContent = "disconnected"; };
  es.onmessage = handleEvent;
})();
</script>
</body>
</html>
"""

@app.get("/self_improve/lessons")
async def self_improve_lessons():
    from AntAgent.autodev.manager_learning import get_lessons
    return JSONResponse(get_lessons())

@app.get("/self_improve/history_tail")
async def self_improve_history_tail(n: int = 20):
    from pathlib import Path
    p = Path(".antagent") / "self_improve_history.jsonl"
    if not p.exists():
        return JSONResponse({"history": []})
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return JSONResponse({"history": lines[-max(1,min(500,n)):]})

_EXAMPLES = {
    "json_schema_extra": {
        "examples": [
            {"objective": "Remove duplicate imports and tighten error handling", "rounds": 1},
            {"objective": "Improve propose_patch return-shape handling and add diff validator", "rounds": 2},
        ]
    }
}

class SelfImproveRequest(BaseModel):
    objective: str = Field(..., description="Short instruction for what to improve.")
    rounds: int = Field(1, ge=1, le=10, description="How many iterations to attempt.")
    model_config = _EXAMPLES  # v2 only

class SelfImproveRoundResult(BaseModel):
    round: int
    summary: Optional[str] = Field(None, description="Short summary from the model.")
    explanation: Optional[str] = Field(None, description="Model's reasoning/explanation string.")
    unified_diff: Optional[str] = Field(None, description="Unified diff (may be truncated).")
    diff_truncated: bool = Field(False, description="True if diff returned here was truncated.")
    applied: bool = Field(..., description="True if patch was applied to the worktree.")
    kept: bool = Field(..., description="True if checks passed and commit was kept.")
    message: str = Field(..., description="Outcome message.")
    paths: List[str] = Field(default_factory=list, description="Paths touched by the diff.")
    checks_log: Optional[str] = Field(None, description="Output of compile/import/pytest checks.")

class SelfImproveResult(BaseModel):
    branch: str
    base: str
    objective: str
    rounds: int
    results: List[SelfImproveRoundResult]

@app.post(
    "/si",
    response_model=SelfImproveResult,
    summary="Self-improve (POST)",
    description="Trigger autonomous self-improvement with reasoning and diff.",
    tags=["Self-Improve"],
)
def si_post(req: SelfImproveRequest = Body(...)):
    from AntAgent.autodev.manager_learning import auto_self_improve
    objective = req.objective.strip()
    if not objective:
        raise HTTPException(status_code=400, detail="objective is required")
    result = auto_self_improve(objective, rounds=req.rounds)
    return JSONResponse(content=result)
def _route_exists(path: str, method: str = "GET") -> bool:
    method = method.upper()
    for r in app.router.routes:
        try:
            if getattr(r, "path", None) == path and method in getattr(r, "methods", set()):
                return True
        except Exception:
            continue
    return False

# Background loop (define before any scheduling)
async def _autosi_loop():
    from AntAgent.autodev.manager_learning import (
        self_improve_once, queue_len, enqueue_objective, ObjectiveItem, generate_objectives
    )
    # initial seed if queue is empty
    try:
        if queue_len() == 0:
            for obj in generate_objectives(max_items=3):
                enqueue_objective(obj)
    except Exception as se:
        print("[AUTOSI] seed error:", se, flush=True)

    while True:
        try:
            res = self_improve_once(prompt=None, rounds=1)
            print("[AUTOSI] step:", (res.get("objective"), len(res.get("results", []))), flush=True)
        except Exception as e:
            print("[AUTOSI] error:", e, flush=True)
        await asyncio.sleep(max(15, int(_AUTOSI_PERIOD_SEC)))

# POST /si (only if not already registered elsewhere)
if not _route_exists("/si", "POST"):
    @app.post(
        "/si",
        response_model=SelfImproveResult,
        summary="Self-improve (POST)",
        description="Trigger autonomous self-improvement with reasoning and diff.",
        tags=["Self-Improve"],
    )
    def si_post(req: SelfImproveRequest = Body(...)):
        from AntAgent.autodev.manager_learning import auto_self_improve
        objective = (req.objective or "").strip()
        if not objective:
            raise HTTPException(status_code=400, detail="objective is required")
        result = auto_self_improve(objective, rounds=req.rounds)
        return JSONResponse(content=result)

# GET /si (only if not already registered elsewhere)
if not _route_exists("/si", "GET"):
    @app.get(
        "/si",
        response_model=SelfImproveResult,
        summary="Self-improve (GET)",
        description="Browser/URL-friendly: /si?objective=...&rounds=1 (reasoning included).",
        tags=["Self-Improve"],
    )
    def si_get(objective: str, rounds: int = 1):
        from AntAgent.autodev.manager_learning import auto_self_improve
        objective = (objective or "").strip()
        if not objective:
            raise HTTPException(status_code=400, detail="objective is required")
        result = auto_self_improve(objective, rounds=rounds)
        return JSONResponse(content=result)

# Prompt-driven one-shot step
if not _route_exists("/si/prompt", "POST"):
    @app.post("/si/prompt")
    async def si_prompt(request: Request):
        """
        Body: { "objective": "string", "rounds": 1 }
        Enqueue and run a single self-improve step with this objective immediately.
        """
        from AntAgent.autodev.manager_learning import self_improve_once
        try:
            payload = await request.json()
            obj = (payload.get("objective") or "").strip()
            rounds = int(payload.get("rounds", 1))
            if not obj:
                raise HTTPException(status_code=400, detail="objective is required")
            result = self_improve_once(obj, rounds=rounds)
            return JSONResponse(result)
        except HTTPException:
            raise
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

# Toggle autonomous mode at runtime
if not _route_exists("/si/autonomous", "POST"):
    @app.post("/si/autonomous")
    async def si_autonomous_toggle(request: Request):
        """
        Body: {"enable": true/false, "period_sec": 300}
        """
        global _AUTOSI_ENABLED, _AUTOSI_PERIOD_SEC
        try:
            payload = await request.json()
            enable = bool(payload.get("enable", False))
            period = int(payload.get("period_sec", _AUTOSI_PERIOD_SEC))
            _AUTOSI_ENABLED = enable
            _AUTOSI_PERIOD_SEC = max(15, period)
            if enable:
                # Start a new background task (it's okay to spawn multiple; loop is idempotent per tick)
                asyncio.create_task(_autosi_loop())
            return {"enabled": _AUTOSI_ENABLED, "period_sec": _AUTOSI_PERIOD_SEC}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

# Start background loop on startup if enabled
@app.on_event("startup")
async def _maybe_start_autosi():
    if _AUTOSI_ENABLED:
        asyncio.create_task(_autosi_loop())


def _chat_append(thread_id: str, role: str, content: str) -> None:
    rec = {
        "ts": time.time(),
        "thread_id": thread_id,
        "role": role,
        "content": content,
    }
    with _CHAT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _chat_load(thread_id: str, limit: int = 50) -> List[Dict]:
    if not _CHAT_PATH.exists():
        return []
    rows = []
    with _CHAT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("thread_id") == thread_id:
                    rows.append(rec)
            except Exception:
                continue
    rows.sort(key=lambda r: r.get("ts", 0.0))
    return rows[-max(1, min(500, limit)):]

def _chat_clear(thread_id: str) -> int:
    if not _CHAT_PATH.exists():
        return 0
    kept = []
    removed = 0
    with _CHAT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("thread_id") != thread_id:
                    kept.append(line.rstrip("\n"))
                else:
                    removed += 1
            except Exception:
                kept.append(line.rstrip("\n"))
    _CHAT_PATH.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return removed

# Model chooser: OpenAI if key present; fallback to local llama_cpp `llm`
def _chat_complete(messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
    """
    Chat completion that always uses the local Qwen model via llama_cpp.
    Ignores any OpenAI API key or model parameter.
    """
    try:
        result = llm.create_chat_completion(
            messages=messages,
            temperature=0.2,
            top_p=0.9,
            max_tokens=1024,
        )
        return (result["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:
        return f"[local chat error] {type(e).__name__}: {e}"

# Default system prompt for the agent
_DEFAULT_CHAT_SYSTEM = (
    "You are ANT Agent: a focused, pragmatic coding and planning assistant. "
    "Be concise, actionable, and specific. When the user asks about self-improvement, "
    "explain exact steps (constraints, anchors, diff rules). Avoid background work; "
    "perform actions only when explicitly invoked via available endpoints."
)

@app.post("/chat/send")
async def chat_send(request: Request):
    """
    Body: {
      "message": "text",
      "thread_id": "optional-string (default: 'default')",
      "model": "optional model name (e.g., gpt-4o-mini)"
    }
    Returns: { "thread_id": ..., "messages": [..., {"role":"assistant","content":"..."}] }
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    user_msg = (payload.get("message") or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="'message' is required")

    thread_id = (payload.get("thread_id") or "default").strip() or "default"
    model = payload.get("model")

    # Load recent history
    history = _chat_load(thread_id, limit=30)

    # Build message list for the model
    msgs = [{"role": "system", "content": _DEFAULT_CHAT_SYSTEM}]
    for rec in history:
        msgs.append({"role": rec["role"], "content": rec["content"]})
    msgs.append({"role": "user", "content": user_msg})

    # Append user message to persistent log
    _chat_append(thread_id, "user", user_msg)

    # Get assistant reply
    assistant_text = _chat_complete(msgs, model=model)
    _chat_append(thread_id, "assistant", assistant_text)

    # Return updated conversation
    updated = _chat_load(thread_id, limit=50)
    return JSONResponse({
        "thread_id": thread_id,
        "messages": updated
    })

@app.get("/chat/history")
def chat_history(thread_id: str = "default", limit: int = 50):
    """
    Query params:
      thread_id: which thread to load (default: 'default')
      limit: max messages to return (default: 50)
    """
    return JSONResponse({
        "thread_id": thread_id,
        "messages": _chat_load(thread_id, limit=limit)
    })

@app.post("/chat/clear")
async def chat_clear(request: Request):
    """
    Body: { "thread_id": "id" }
    Clears the specified chat thread.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    thread_id = (payload.get("thread_id") or "default").strip() or "default"
    removed = _chat_clear(thread_id)
    return {"thread_id": thread_id, "removed": removed}

@app.get("/chat/ui", response_class=HTMLResponse)
def chat_ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>ANT Agent Chat</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root { --bg:#0f172a; --panel:#111827; --fg:#e5e7eb; --muted:#94a3b8; --accent:#60a5fa; }
    body { margin:0; font:14px/1.45 system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background:var(--bg); color:var(--fg); }
    header { padding:12px 16px; background:#0b1220; border-bottom:1px solid #1f2937; display:flex; gap:10px; align-items:center; }
    h1 { font-size:16px; margin:0; color:var(--fg); }
    main { display:grid; grid-template-rows: 1fr auto; height: calc(100vh - 48px); }
    #log { padding:12px; overflow:auto; }
    .msg { padding:10px 12px; margin:8px 0; border-radius:10px; max-width: 900px; white-space:pre-wrap; }
    .user { background:#1d283a; border:1px solid #253049; }
    .assistant { background:#152237; border:1px solid #23314c; }
    form { display:flex; gap:8px; padding:12px; background:#0b1220; border-top:1px solid #1f2937; }
    input, textarea { background:#0b1220; color:var(--fg); border:1px solid #263244; border-radius:8px; padding:8px; width:100%; }
    button { background:#1b2a45; color:#dbeafe; border:1px solid #2a3a5b; border-radius:8px; padding:8px 12px; }
    .meta { color:var(--muted); font-size:12px; margin-bottom:4px; }
  </style>
</head>
<body>
  <header><h1>ANT Agent — Chat</h1></header>
  <main>
    <div id="log"></div>
    <form id="f">
      <input type="text" id="thread" value="default" style="max-width:220px" />
      <input type="text" id="msg" placeholder="Type a message…" autofocus />
      <button type="submit">Send</button>
      <button type="button" id="clear">Clear</button>
    </form>
  </main>
<script>
  const log = document.getElementById('log');
  const f = document.getElementById('f');
  const threadEl = document.getElementById('thread');
  const msgEl = document.getElementById('msg');
  const clearBtn = document.getElementById('clear');

  function render(messages){
    log.innerHTML = '';
    messages.forEach(m => {
      const wrap = document.createElement('div');
      wrap.className = 'msg ' + (m.role === 'user' ? 'user' : 'assistant');
      const meta = document.createElement('div');
      meta.className = 'meta';
      const d = new Date((m.ts||0)*1000);
      meta.textContent = (m.role||'?') + ' · ' + (isNaN(d) ? '' : d.toLocaleString());
      const pre = document.createElement('div');
      pre.textContent = m.content || '';
      wrap.appendChild(meta);
      wrap.appendChild(pre);
      log.appendChild(wrap);
    });
    log.scrollTop = log.scrollHeight;
  }

  async function load(){
    const tid = threadEl.value || 'default';
    const r = await fetch('/chat/history?thread_id=' + encodeURIComponent(tid) + '&limit=100');
    const j = await r.json();
    render(j.messages || []);
  }

  f.addEventListener('submit', async (e) => {
    e.preventDefault();
    const tid = threadEl.value || 'default';
    const text = msgEl.value || '';
    if(!text.trim()) return;
    msgEl.value = '';
    const r = await fetch('/chat/send', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({thread_id: tid, message: text})
    });
    const j = await r.json();
    render(j.messages || []);
  });

  clearBtn.addEventListener('click', async () => {
    const tid = threadEl.value || 'default';
    await fetch('/chat/clear', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({thread_id: tid})
    });
    load();
  });

  load();
</script>
</body>
</html>
"""