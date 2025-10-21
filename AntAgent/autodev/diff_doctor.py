from __future__ import annotations
import re
from typing import Tuple

HUNK_RE = re.compile(r"^@@\s*-\d+(?:,\d+)?\s*\+\d+(?:,\d+)?\s*@@", re.M)

def has_top_insert(diff_text: str) -> bool:
    """
    Heuristic: a hunk with + lines and zero context at file start often indicates top insert.
    """
    # if first hunk starts at +1 with near-zero context and begins with '+' lines => risky
    # Relaxed check: any diff that begins immediately with '+' before any context markers
    lines = diff_text.splitlines()
    # quick negative checks
    if not HUNK_RE.search(diff_text):
        return False
    # crude heuristic
    plus_first = 0
    for ln in lines:
        if ln.startswith("diff --git"):
            continue
        if ln.startswith("@@"):
            break
        if ln.startswith("+"):
            plus_first += 1
        elif ln.startswith(" "):
            # context exists
            return False
        elif ln.startswith("-"):
            return False
    return plus_first > 0

def enforce_context(diff_text: str, min_context: int = 1) -> Tuple[bool, str]:
    """
    Ensure each hunk contains at least `min_context` context lines.
    Returns (ok, maybe_fixed_diff).
    """
    if min_context <= 0:
        return True, diff_text
    # soft check: require at least one ' ' line after each @@ header
    parts = diff_text.splitlines()
    out, ok = [], True
    pending_hunk = False
    ctx_count = 0
    for ln in parts:
        if ln.startswith("@@"):
            if pending_hunk and ctx_count < min_context:
                ok = False
            pending_hunk = True
            ctx_count = 0
            out.append(ln)
            continue
        if pending_hunk:
            if ln.startswith(" "):
                ctx_count += 1
            if ln.startswith("@@"):
                # will be handled next iteration; keep flow
                pass
        out.append(ln)
    if pending_hunk and ctx_count < min_context:
        ok = False
    return ok, "\n".join(out) + ("\n" if diff_text.endswith("\n") else "")
