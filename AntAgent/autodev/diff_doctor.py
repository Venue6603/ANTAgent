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

from __future__ import annotations
import re
from typing import List, Tuple

HUNK_RE = re.compile(r"^@@\s*-\d+(?:,\d+)?\s*\+\d+(?:,\d+)?\s*@@", re.M)
IMPORT_RE = re.compile(r"^\s*(from\s+\S+\s+import\b|import\s+\S+)")

def has_top_insert(diff_text: str) -> bool:
    """
    Heuristic: reject diffs whose first hunk adds lines without any surrounding
    context or deletions (common 'top insert' failure).
    """
    lines = diff_text.splitlines()
    in_hunk = False
    saw_context = False
    saw_minus = False
    saw_plus = False

    for ln in lines:
        if ln.startswith("@@"):
            # reset for first hunk only; after we saw a proper mix, stop checking
            if in_hunk:
                break
            in_hunk = True
            saw_context = False
            saw_minus = False
            saw_plus = False
            continue
        if not in_hunk:
            continue
        if ln.startswith(" "):
            saw_context = True
        elif ln.startswith("-"):
            saw_minus = True
        elif ln.startswith("+"):
            saw_plus = True

    # "Top insert" signature: only additions in first hunk and no context/deletions
    return in_hunk and saw_plus and not saw_context and not saw_minus


def validate_min_context(diff_text: str, min_context: int = 1) -> Tuple[bool, str]:
    """
    Ensure each hunk has at least `min_context` lines that start with a space.
    """
    if min_context <= 0:
        return True, ""

    lines = diff_text.splitlines()
    in_hunk = False
    context_count = 0
    any_hunks = False

    for ln in lines:
        if ln.startswith("@@"):
            any_hunks = True
            # check previous hunk
            if in_hunk and context_count < min_context:
                return False, f"Hunk has only {context_count} context lines (< {min_context})."
            # start new hunk
            in_hunk = True
            context_count = 0
            continue
        if not in_hunk:
            continue
        if ln.startswith(" "):
            context_count += 1

    if not any_hunks:
        return False, "No @@ hunks found."
    if in_hunk and context_count < min_context:
        return False, f"Final hunk has only {context_count} context lines (< {min_context})."

    return True, ""


def reject_import_inline_comment(diff_text: str) -> Tuple[bool, str]:
    """
    Reject diffs that *add* inline comments to import lines without removing
    the original import line. Prevents things like:
        +from x import y  # Random animal: Giraffe
    without a matching '-' removal.
    """
    lines = diff_text.splitlines()
    in_hunk = False
    minus_imports: List[str] = []
    plus_imports: List[str] = []

    for ln in lines:
        if ln.startswith("@@"):
            # new hunk
            in_hunk = True
            minus_imports.clear()
            plus_imports.clear()
            continue
        if not in_hunk:
            continue

        if ln.startswith("-") and IMPORT_RE.search(ln[1:]):
            minus_imports.append(ln[1:])
        elif ln.startswith("+"):
            body = ln[1:]
            if IMPORT_RE.search(body):
                plus_imports.append(body)
                if "#" in body and not minus_imports:
                    return False, "Inline comment added to import line without proper replacement."
    return True, ""


def vet_diff(diff_text: str, min_context: int = 1) -> Tuple[bool, str]:
    """
    Combined gate used by the patch applier:
    - Reject 'top insert' patterns.
    - Enforce minimal context per hunk.
    - Reject import-line inline comments.
    - Require Random-animal change to be a true replacement (not append-only).
    """
    if has_top_insert(diff_text):
        return False, "Rejecting diff: suspicious top-of-file insertion."

    ok, msg = validate_min_context(diff_text, min_context=min_context)
    if not ok:
        return False, f"Rejecting diff: insufficient hunk context (min {min_context}). {msg}"

    ok, msg = reject_import_inline_comment(diff_text)
    if not ok:
        return False, f"Rejecting diff: {msg}"

    ok, msg = _reject_append_without_remove_random_animal(diff_text)
    if not ok:
        return False, f"Rejecting diff: {msg}"

    return True, ""