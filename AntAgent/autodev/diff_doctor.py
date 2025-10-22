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

# Add this helper (near your other helpers)
def _reject_append_without_remove_random_animal(diff_text: str) -> tuple[bool, str]:
    """
    If a hunk *adds* a line that matches '^# Random animal:' it must also *remove*
    a line that matches '^# Random animal:' in the SAME hunk. This blocks diffs
    that merely append a second comment instead of replacing the original.
    """
    in_hunk = False
    minus_has_ra = False
    plus_has_ra = False

    def flush_check():
        if plus_has_ra and not minus_has_ra:
            return (False, "Rejecting diff: Random-animal comment was added without removing the original (append-only).")
        return (True, "")

    for ln in diff_text.splitlines():
        if ln.startswith("@@"):
            # New hunk: check the last one, then reset
            ok, why = flush_check()
            if not ok:
                return ok, why
            in_hunk = True
            minus_has_ra = False
            plus_has_ra = False
            continue
        if not in_hunk:
            continue
        if ln.startswith("-") and ln[1:].lstrip().startswith("# Random animal:"):
            minus_has_ra = True
        elif ln.startswith("+") and ln[1:].lstrip().startswith("# Random animal:"):
            plus_has_ra = True

    # final hunk
    ok, why = flush_check()
    if not ok:
        return ok, why
    return (True, "")

_HUNK_RE = re.compile(r"^@@\s*-\d+(?:,\d+)?\s*\+\d+(?:,\d+)?\s*@@", re.M)
_IMPORT_RE = re.compile(r"^\s*(from\s+\S+\s+import\b|import\s+\S+)")

def _has_top_insert(diff_text: str) -> bool:
    """
    Heuristic: first hunk has only additions and zero context/deletions.
    """
    lines = diff_text.splitlines()
    in_hunk = False
    saw_plus = saw_minus = saw_ctx = False
    for ln in lines:
        if ln.startswith("@@"):
            # consider only first hunk
            if in_hunk:
                break
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if ln.startswith("+"): saw_plus = True
        elif ln.startswith("-"): saw_minus = True
        elif ln.startswith(" "): saw_ctx = True
    return in_hunk and saw_plus and not saw_minus and not saw_ctx

def _validate_min_context(diff_text: str, min_context: int = 1) -> Tuple[bool, str]:
    if min_context <= 0:
        return True, ""
    lines = diff_text.splitlines()
    in_hunk = False
    ctx = 0
    any_hunk = False
    for ln in lines:
        if ln.startswith("@@"):
            any_hunk = True
            if in_hunk and ctx < min_context:
                return False, f"Hunk has only {ctx} context lines (< {min_context})."
            in_hunk = True
            ctx = 0
            continue
        if not in_hunk:
            continue
        if ln.startswith(" "):
            ctx += 1
    if not any_hunk:
        return False, "No @@ hunks found."
    if in_hunk and ctx < min_context:
        return False, f"Final hunk has only {ctx} context lines (< {min_context})."
    return True, ""

def _reject_import_inline_comment(diff_text: str) -> Tuple[bool, str]:
    """
    Forbid adding inline comments to import lines unless the import line is also
    removed in the same hunk (proper replacement).
    """
    lines = diff_text.splitlines()
    in_hunk = False
    minus_seen = False
    for ln in lines:
        if ln.startswith("@@"):
            in_hunk = True
            minus_seen = False
            continue
        if not in_hunk:
            continue
        if ln.startswith("-") and _IMPORT_RE.search(ln[1:]):
            minus_seen = True
        if ln.startswith("+") and _IMPORT_RE.search(ln[1:]):
            if "#" in ln and not minus_seen:
                return False, "Inline comment added to import line without matching '-' removal."
    return True, ""

def _require_add_and_remove_each_hunk(diff_text: str) -> Tuple[bool, str]:
    """
    Universal rule: every hunk must include at least one '+' and at least one '-'.
    Prevents append-only or delete-only diffs that often fail to anchor.
    
    EXCEPTION: Pure insertions are allowed if they have sufficient context lines
    to anchor properly (at least 2 context lines).
    """
    lines = diff_text.splitlines()
    in_hunk = False
    plus = minus = False
    context_count = 0
    saw_any = False
    def flush() -> Tuple[bool, str]:
        if not in_hunk:
            return True, ""
        # Allow pure insertions if they have sufficient context
        if plus and not minus and context_count >= 2:
            return True, ""
        # Allow pure deletions if they have sufficient context  
        if minus and not plus and context_count >= 2:
            return True, ""
        # Require both + and - for hunks with insufficient context
        if not plus or not minus:
            return False, "Each hunk must contain at least one added ('+') and one removed ('-') line, unless it has sufficient context (2+ lines)."
        return True, ""
    for ln in lines:
        if ln.startswith("@@"):
            ok, why = flush()
            if not ok: return ok, why
            in_hunk = True; saw_any = True; plus = minus = False; context_count = 0
            continue
        if not in_hunk: continue
        if ln.startswith("+"): plus = True
        elif ln.startswith("-"): minus = True
        elif ln.startswith(" "): context_count += 1
    ok, why = flush()
    if not ok: return ok, why
    if not saw_any: return False, "No @@ hunks found."
    return True, ""

def vet_diff(diff_text: str, min_context: int = 1) -> Tuple[bool, str]:
    """
    Universal diff vetting:
      - Must start with a 'diff --git' block and at least one @@ hunk
      - Reject top-of-file blind inserts
      - Require per-hunk minimum context lines
      - Forbid import inline-comment append
      - Require each hunk to have at least one '+' and one '-'
    """
    if "diff --git " not in diff_text:
        return False, "Missing 'diff --git' header."
    if not _HUNK_RE.search(diff_text):
        return False, "No @@ hunks found."
    if _has_top_insert(diff_text):
        return False, "Suspicious top-of-file insertion."
    ok, why = _validate_min_context(diff_text, min_context=min_context)
    if not ok: return False, why
    ok, why = _reject_import_inline_comment(diff_text)
    if not ok: return False, why
    ok, why = _require_add_and_remove_each_hunk(diff_text)
    if not ok: return False, why
    return True, ""