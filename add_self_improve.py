"""
Adds a self-improvement endpoint (/self_improve) and patch-explanation logic
to ANTAgent so it can modify its own code safely.
"""

from pathlib import Path
import requests, json

PATCH = r"""
diff --git a/AntAgent/autodev/manager.py b/AntAgent/autodev/manager.py
index 1111111..2222222 100644
--- a/AntAgent/autodev/manager.py
+++ b/AntAgent/autodev/manager.py
@@
import os
from typing import Dict, List, Tuple

SYSTEM_POLICY = (
    "You are a careful code refactoring assistant. "
    "Produce concise explanations and safe, minimal diffs. "
    "Never invent files; do not break public APIs unless instructed."
)
MODEL_PATH = os.environ.get("ANT_LLAMA_MODEL_PATH")

def _llama():
    if MODEL_PATH:
        from ollama_adapter import Llama
        return Llama(model_path=MODEL_PATH, n_ctx=8192, n_gpu_layers=-1, verbose=False)
    return None

def _openai_completion(prompt: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI()
        res = client.chat.completions.create(
            model="gpt-5-turbo",
            messages=[{"role":"system","content":SYSTEM_POLICY},
                      {"role":"user","content":prompt}],
            temperature=0.2, top_p=0.9, max_tokens=2400
        )
        return res.choices[0].message.content or ""
    except Exception:
        return ""

def _build_prompt(goal: str, constraints: Dict) -> str:
    paths: List[str] = constraints.get("paths") or []
    no_net_new_deps = bool(constraints.get("no_net_new_deps"))
    extra_rules: List[str] = constraints.get("rules") or []

    rules = [
        "After analysis, output two sections in this order:",
        "---SUMMARY---",
        "(<=120 words human-readable summary)",
        "---DIFF---",
        "(pure git diff)",
        "Preserve APIs unless requested."
    ]
    if no_net_new_deps:
        rules.append("Do not add new third-party dependencies.")
    rules.extend(extra_rules)

    path_text = "\n".join(f"- {p}" for p in paths) if paths else "(no path restriction)"
    return f"Goal:\n{goal}\n\nTarget paths:\n{path_text}\n\nRules:\n- " + "\n- ".join(rules)

def _parse_summary_and_diff(text: str) -> Tuple[str, str]:
    t = (text or "").replace("```", "").strip()
    if "---DIFF---" in t:
        parts = t.split("---DIFF---", 1)
        summary = parts[0].split("---SUMMARY---")[-1].strip()
        diff = parts[1].strip()
    else:
        k = t.find("diff --git ")
        summary, diff = (t[:k].strip(), t[k:].strip()) if k >= 0 else (t, "")
    return summary[:800], diff

def propose_patch_with_explanation(goal: str, constraints: Dict) -> Tuple[str, str]:
    prompt = _build_prompt(goal, constraints)
    try:
        llm = _llama()
        if llm:
            out = llm.create_chat_completion(
                messages=[{"role":"system","content":SYSTEM_POLICY},
                          {"role":"user","content":prompt}],
                temperature=0.2, top_p=0.9, max_tokens=2400
            )["choices"][0]["message"]["content"]
            return _parse_summary_and_diff(out)
    except Exception:
        pass
    return _parse_summary_and_diff(_openai_completion(prompt))

def propose_patch(goal: str, constraints: Dict) -> str:
    _, diff = propose_patch_with_explanation(goal, constraints)
    return diff

def self_improve_round(goal: str, constraints: Dict) -> Dict:
    from .patch import apply_unified_diff
    summary, diff = propose_patch_with_explanation(goal, constraints)
    if not diff:
        return {"summary": summary, "unified_diff": "", "applied": False, "message": "No diff produced"}
    try:
        apply_unified_diff(diff)
        return {"summary": summary, "unified_diff": diff, "applied": True, "message": "Patch applied"}
    except Exception as e:
        return {"summary": summary, "unified_diff": diff, "applied": False, "message": f"Apply failed: {e}"}

diff --git a/AntAgent/app.py b/AntAgent/app.py
index 3333333..4444444 100644
--- a/AntAgent/app.py
+++ b/AntAgent/app.py
@@
from AntAgent.autodev.manager import (
    propose_patch, propose_patch_with_explanation, self_improve_round,
    run_quality_checks, apply_patch, list_paths, read_files
)

@app.post("/code/propose_patch")
def code_propose_patch(request: Request):
    payload = request.json()
    goal = payload.get("goal", "")
    constraints = payload.get("constraints", {})
    summary, patch_txt = propose_patch_with_explanation(goal, constraints)
    return JSONResponse({"summary": summary, "unified_diff": patch_txt})

@app.post("/self_improve")
def self_improve(request: Request):
    payload = request.json()
    goal = payload.get("goal", "")
    constraints = payload.get("constraints", {}) or {}
    rounds = int(payload.get("rounds", 1))
    results = []
    for i in range(max(1, rounds)):
        res = self_improve_round(goal, constraints)
        results.append({"round": i + 1, **res})
        if not res.get("unified_diff"):
            break
    return JSONResponse({"results": results})

diff --git a/AntAgent/autodev/allowlist.txt b/AntAgent/autodev/allowlist.txt
index 5555555..6666666 100644
--- a/AntAgent/autodev/allowlist.txt
+++ b/AntAgent/autodev/allowlist.txt
@@
AntAgent/planning/planner.py
AntAgent/autodev/manager.py
AntAgent/app.py
"""

# --- Write the patch to disk ---
patch_path = Path("self_improve_patch.diff")
patch_path.write_text(PATCH)
print(f"Patch saved to {patch_path.resolve()}")

# --- Try to apply via local server if running ---
try:
    response = requests.post(
        "http://127.0.0.1:8008/code/apply_patch",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"unified_diff": PATCH})
    )
    print("Server response:", response.status_code)
    print(response.text[:400])
except Exception as e:
    print("Server not reachable. You can manually upload self_improve_patch.diff in /code/apply_patch.")
