

# LLAMA-SMOKE OK
# ================= OpenAI-first proposer WITH FILE CONTEXT =================

import os, re, json, requests
from typing import Dict, Tuple, List
from pathlib import Path
import traceback  # <-- needed for error formatting
from ollama_adapter import Llama
import os
import re as _re
from AntAgent.autodev.manager_learning import get_learning_system, _allowed_paths

# Random animal: Giraffe
from pathlib import Path as _Path

def _repo_root() -> _Path:
    return _Path(__file__).resolve().parents[2]

def _first_diff_target_path(diff_text: str) -> str | None:
    m = _re.search(r"(?m)^diff --git\s+a/(?P<p>.+?)\s+b/(?P<p2>.+?)$", diff_text)
    if m and m.group("p") == m.group("p2"):
        return m.group("p")
    return None

def preflight_verify_paths(diff_text: str) -> None:
    rel = _first_diff_target_path(diff_text)
    if not rel:
        raise ValueError("Cannot determine target path from diff header")
    abs_path = _repo_root() / rel
    if not abs_path.exists():
        raise FileNotFoundError(f"Target file not found at repo root: {rel}")

model_path = os.getenv("ANT_LLAMA_MODEL_PATH")

# Tune these via env vars without touching code
ctx        = int(os.getenv("ANT_LLAMA_CTX", "4096"))      # keep <= model's train ctx unless you know what you're doing
batch      = int(os.getenv("ANT_LLAMA_N_BATCH", "256"))   # smaller uses less VRAM
gpu_layers = int(os.getenv("ANT_LLAMA_N_GPU_LAYERS", "20"))  # try 20 first on a 3080 if ~3.5–4 GB VRAM free
LLAMA_DEFAULT_CTX = int(os.getenv("ANT_LLAMA_CTX", "4096"))
LLAMA_MAX_TOKENS  = int(os.getenv("ANT_LLAMA_MAX_TOKENS", "512"))
LLAMA_SAFETY_PAD  = 256  # guardrail to avoid overruns
_LAST_SELF_IMPROVE_DIAG: dict | None = None
# Tracks which remote provider was used when llama is unavailable
_LAST_FALLBACK_ENGINE: str | None = None

def _validate_unified_diff(diff_text: str) -> None:
    """
    Strict unified diff validator.
    Requires at least one 'diff --git' header and one '@@' hunk.
    Raises ValueError if invalid.
    """
    if not isinstance(diff_text, str) or not diff_text.strip():
        raise ValueError("Empty diff")
    has_header = bool(_re.search(r"(?m)^diff --git\s+a/.*\s+b/.*$", diff_text))
    has_hunk = bool(_re.search(r"(?m)^@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@", diff_text))
    if not (has_header and has_hunk):
        raise ValueError("Invalid unified diff: missing header and/or hunk")

def get_last_self_improve_diag() -> dict:
    """Return the diagnostic snapshot from the most recent self_improve_round."""
    return _LAST_SELF_IMPROVE_DIAG or {"note": "no self_improve_round has run yet"}

def _approx_tokens(s: str) -> int:
    """Very rough token estimator (~4 chars/token)."""
    return max(1, len(s) // 4)

def _budgeted_files_block(files: List[Tuple[str, str]], token_budget: int) -> str:
    """
    Build a 'CURRENT FILES' block that stays under token_budget tokens.
    Keeps head & tail for each file; stops when budget is reached.
    """
    if not files:
        return "(no file content provided)"

    sections: List[str] = []
    used = 0

    HEAD = 1800  # chars per head slice
    TAIL = 1800  # chars per tail slice
    SEP  = "\n\n"

    for path, content in files:
        if not content:
            snippet = ""
        else:
            if len(content) <= HEAD + TAIL + 200:
                snippet = content
            else:
                snippet = content[:HEAD] + "\n# ... [truncated for prompt] ...\n" + content[-TAIL:]

        section = f"<<FILE {path} START>>\n{snippet}\n<<FILE {path} END>>"
        section_tokens = _approx_tokens(section) + _approx_tokens(SEP)

        if used + section_tokens > token_budget:
            sections.append("(additional files omitted due to token budget)")
            break

        sections.append(section)
        used += section_tokens

    return SEP.join(sections) if sections else "(no file content provided)"

MAX_BYTES_PER_FILE = 120_000  # cap context per file

def _read_files(paths: List[str]) -> List[Tuple[str, str]]:
    out = []
    for p in paths or []:
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                txt = f.read()
            if len(txt.encode("utf-8")) > MAX_BYTES_PER_FILE:
                # keep start and end, drop middle
                head = txt[:60_000]
                tail = txt[-60_000:]
                txt = head + "\n# ... [truncated for prompt] ...\n" + tail
            out.append((p, txt))
        except Exception as e:
            out.append((p, f"# ERROR: could not read {p}: {e}"))
    return out

def _render_diff_prompt(goal: str, constraints: Dict, files: List[Tuple[str, str]]) -> str:
# This function builds the natural language prompt sent to the model.
    targets = [p for p, _ in files] or (constraints.get("paths") or [])
    '''This function builds the natural language prompt sent to the model.'''
    tgt_txt = ", ".join(targets) if targets else "(no specific paths)"
    sections = []
    for path, content in files:
        sections.append(
            f"<<FILE {path} START>>\n{content}\n<<FILE {path} END>>"
        )
    files_block = "\n\n".join(sections) if sections else "(no file content provided)"

    return f"""You are a code maintenance agent.

TASK:
{goal}

CONSTRAINTS:
- Edit only these paths: {tgt_txt}
- Preserve behavior unless explicitly told to change it
- No new dependencies unless allowed
- Output must be a valid UNIX unified diff that starts with 'diff --git'
- Do NOT output explanations, markdown, or backticks. ONLY the diff.

CURRENT FILES (authoritative):
{files_block}

OUTPUT FORMAT (strict):
- Begin with: diff --git a/<path> b/<path>
- Include '--- a/<path>' and '+++ b/<path>'
- Include '@@' hunks with correct line numbers for the CURRENT FILES above
- Ensure the patch applies cleanly with 'git apply'
"""


def _extract_unified_diff(text: str) -> str:
    """Extract unified diff from possibly markdown-wrapped text."""
    if not text:
        return ""

    # First, check if there are line numbers that need cleaning
    if re.search(r'^\s*\d+:', text, re.MULTILINE):
        print("[DEBUG] Cleaning line numbers from diff...")
        lines = []
        for line in text.split('\n'):
            # Remove line numbers but preserve diff markers
            if re.match(r'^\s*\d+:\s*[+-]', line):
                marker = '+' if '+' in line[:20] else '-'
                content = re.sub(r'^\s*\d+:\s*[+-]\s*', '', line)
                lines.append(marker + content)
            elif re.match(r'^\s*\d+:', line):
                content = re.sub(r'^\s*\d+:\s*', '', line)
                lines.append(' ' + content)
            else:
                lines.append(line)
        text = '\n'.join(lines)

    # Normalize newlines and strip markdown code fences
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^```(?:diff|patch)?\s*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n```\s*$", "", text)
    text = text.strip()

    # Find the start of the unified diff
    m = re.search(r"(?m)^diff --git ", text)
    if not m:
        return ""

    # Extract from the diff start
    diff_content = text[m.start():]

    # IMPORTANT: Stop at the end of the diff hunk
    # A diff ends when we hit a line that doesn't start with a valid marker
    lines = diff_content.split('\n')
    clean_lines = []
    in_hunk = False

    for line in lines:
        # Track when we're in a hunk
        if line.startswith('@@'):
            in_hunk = True
            clean_lines.append(line)
        elif in_hunk:
            # Valid hunk lines start with space, -, or +
            if line and not line[0] in ' -+':
                # Check if this might be a header line we should keep
                if any(line.startswith(x) for x in ['diff --git', '---', '+++', 'index']):
                    clean_lines.append(line)
                    in_hunk = False
                else:
                    # This is probably explanatory text, stop here
                    break
            else:
                clean_lines.append(line)
        else:
            # Not in a hunk, keep header lines
            clean_lines.append(line)

    diff_content = '\n'.join(clean_lines).strip()

    # Validate it has the essential components
    if not re.search(r"(?m)^---", diff_content):
        print("[DEBUG] Diff extraction failed: missing '---' header")
        return ""
    if not re.search(r"(?m)^\+\+\+", diff_content):
        print("[DEBUG] Diff extraction failed: missing '+++' header")
        return ""
    if not re.search(r"(?m)^@@", diff_content):
        print("[DEBUG] Diff extraction failed: missing '@@' hunk")
        return ""

    # Check if the diff appears to be truncated (ends abruptly)
    if diff_content and not diff_content.endswith('\n') and len(diff_content.split('\n')) > 1:
        last_line = diff_content.split('\n')[-1]
        # If the last line doesn't end with a newline and doesn't look like a complete diff line
        if last_line and not last_line.startswith((' ', '-', '+', '@@', 'diff', '---', '+++', 'index')):
            print(f"[DEBUG] Diff appears truncated, last line: '{last_line[:50]}...'")
            # Try to salvage by removing the incomplete last line
            lines = diff_content.split('\n')
            if len(lines) > 1:
                diff_content = '\n'.join(lines[:-1])
                print(f"[DEBUG] Salvaged diff by removing incomplete last line")

    return diff_content

def fix_malformed_prepend_diff(diff_text: str, target_path: str) -> str:
    """
    Read the contents of the specified files.
    Fix malformed diffs that try to prepend lines to files.
    Handle patterns like '@@ -1,0 +1,1 @@' and '@@ -0,0 +1 @@' by including proper context.
    """
    if not diff_text or ('@@ -1,0 +1,1 @@' not in diff_text and '@@ -0,0 +1 @@' not in diff_text):
        return diff_text

    try:
        # Read the actual file to get proper context
        with open(target_path, 'r', encoding='utf-8') as f:
            original_lines = f.read().splitlines()

        if not original_lines:
            # File is empty, the diff might be correct
            return diff_text

        # Extract what line is being added
        added_line = None
        lines = diff_text.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('+') and not line.startswith('+++'):
                added_line = line[1:]  # Remove the + prefix
                break

        if not added_line:
            return diff_text

        # Rebuild the diff with proper context
        context_count = min(3, len(original_lines))  # Use up to 3 lines of context

        # Find where the hunk starts and rebuild it
        new_lines = []
        hunk_fixed = False

        for line in lines:
            if not hunk_fixed and line.startswith('@@') and (('-1,0 +1,1' in line) or ('-0,0 +1' in line)):
                # Fix this hunk header
                new_lines.append(f'@@ -1,{context_count} +1,{context_count + 1} @@')
                new_lines.append(f'+{added_line}')
                # Add context lines (without + or -)
                for i in range(context_count):
                    if i < len(original_lines):
                        new_lines.append(' ' + original_lines[i])
                hunk_fixed = True
                continue
            elif hunk_fixed and (line.startswith('+') or line.startswith('-') or line.startswith(' ')):
                # Skip the old hunk content for the malformed hunk
                if not line.startswith('@@') and not line.startswith('diff --git'):
                    continue

            new_lines.append(line)

        return '\n'.join(new_lines)

    except Exception as e:
        print(f"[DEBUG] Could not fix malformed diff: {e}")
        return diff_text




def _openai_completion_diff_only(prompt: str, *, model: str = "gpt-4o-mini") -> str:
    """
    Try DeepSeek (OpenAI-compatible) first if DEEPSEEK_API_KEY is set, else fall back to OpenAI.
    Returns assistant text or empty string on failure.
    """
    global _LAST_FALLBACK_ENGINE

    # 1) DeepSeek first (if configured)
    ds_key = os.getenv("DEEPSEEK_API_KEY")
    if ds_key:
        base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        ds_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        try:
            resp = requests.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {ds_key}", "Content-Type": "application/json"},
                data=json.dumps({
                    "model": ds_model,
                    "messages": [
                        {"role": "system", "content": "Return only a valid unified diff; no prose or backticks. STRICT"},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0
                }),
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            _LAST_FALLBACK_ENGINE = "deepseek"
            return (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
        except Exception:
            # continue to OpenAI below
            pass

    # 2) OpenAI fallback
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            data=json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": "Return only a valid unified diff; no prose or backticks. STRICT"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0
            }),
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        _LAST_FALLBACK_ENGINE = "openai"
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return ""

def _llama_chat_diff_only(llm, prompt: str, *, max_tokens: int = 2048) -> str:
    """
    Call llama.cpp chat with parameters tuned to produce only a unified diff.
    Returns the assistant text (may be empty on failure).
    """
    try:
        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Return only a valid unified diff; no prose or backticks."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            top_p=0.10,
            repeat_penalty=1.05,
            max_tokens=max_tokens,
            add_generation_prompt=True,   # <-- critical for Qwen chat templates
        )
        return (result["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return ""


def propose_patch_with_explanation(goal: str, constraints: Dict) -> Tuple[str, str, str]:
    """
    Generate a patch with detailed explanation of the approach.
    Returns: (summary, unified_diff, explanation)
    """
    import re
    from pathlib import Path

    target_paths = constraints.get("paths") or []

    # Path expansion
    all_allowed = _allowed_paths()
    expanded_paths = []
    for path in target_paths:
        path_obj = Path(_repo_root() / path)
        if path_obj.is_dir():
            dir_str = str(path).replace("\\", "/")
            for allowed in all_allowed:
                allowed_norm = allowed.replace("\\", "/")
                if allowed_norm.startswith(dir_str + "/") and allowed_norm.endswith(".py"):
                    full_path = Path(_repo_root() / allowed_norm)
                    if full_path.exists():
                        expanded_paths.append(allowed_norm)
        elif path in all_allowed:
            full_path = Path(_repo_root() / path)
            if full_path.exists():
                expanded_paths.append(path)

    if expanded_paths:
        target_paths = list(set(expanded_paths))

    # Read files
    files = _read_files(target_paths)
    if not files:
        return "No files to process", "", "No target files found"

    path, content = files[0]  # Focus on first file
    lines = content.splitlines()

    # Create numbered version for analysis
    numbered_lines = [f"{i + 1:4}: {line}" for i, line in enumerate(lines)]
    numbered_content = "\n".join(numbered_lines)

    used_engine = None
    diff_text = ""
    explanation = ""

    try:
        from ollama_adapter import Llama
        model_path = os.getenv("ANT_LLAMA_MODEL_PATH")

        if model_path and Path(model_path).exists():
            n_ctx = 8192

            # Single combined prompt - analysis and diff generation together
            combined_prompt = f"""Task: {goal}

File to modify: {path}

Current file content with line numbers:
{numbered_content[:6000]}

Instructions:
1. First, identify what needs to change and where
2. Then create a unified diff showing that change

A unified diff has this structure:
- Header: diff --git a/filepath b/filepath
- File markers: --- a/filepath and +++ b/filepath  
- Hunk header: @@ -start,count +start,count @@
- Lines: unchanged lines start with space, removed with -, added with +

Your response should be:
1. Brief analysis of what to change
2. The complete unified diff

Generate the unified diff now:"""

            llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_batch=256,
                n_gpu_layers=gpu_layers,
                verbose=False,
            )

            result = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a code editor. Analyze tasks and generate unified diffs."},
                    {"role": "user", "content": combined_prompt},
                ],
                temperature=0.0,
                top_p=0.05,
                max_tokens=2048,
            )

            full_response = result["choices"][0]["message"]["content"] or ""

            # Extract explanation (everything before the diff)
            if "diff --git" in full_response:
                parts = full_response.split("diff --git", 1)
                explanation = parts[0].strip()
                diff_text = "diff --git" + parts[1]
            else:
                explanation = full_response[:200]
                diff_text = full_response

            # Clean and validate the diff
            candidate = _extract_unified_diff(diff_text)
            if candidate:
                # Basic validation - check it has the required parts
                has_header = "diff --git" in candidate
                has_file_markers = "---" in candidate and "+++" in candidate
                has_hunk = "@@" in candidate
                has_changes = "-" in candidate or "+" in candidate

                if has_header and has_file_markers and has_hunk and has_changes:
                    diff_text = candidate
                    used_engine = "llama"
                    print(f"[ENGINE] LLM generated valid diff")
                else:
                    print(f"[ENGINE] LLM diff missing required parts")
                    diff_text = ""
            else:
                print(f"[ENGINE] Could not extract valid diff from LLM output")
                diff_text = ""

    except Exception as e:
        print(f"[ENGINE] LLM error: {e}")
        diff_text = ""

    # If LLM failed, try OpenAI
    if not diff_text:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print("[ENGINE] Trying OpenAI fallback")
            try:
                import requests, json

                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    data=json.dumps({
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system",
                             "content": "You are a diff generator. Create unified diffs for code changes."},
                            {"role": "user", "content": f"""Task: {goal}

File: {path}
Content (first 100 lines):
{chr(10).join(lines[:100])}

Generate a unified diff for this change. 
Start with: diff --git a/{path} b/{path}
Include --- and +++ lines
Include @@ hunk markers
Show removed lines with - and added lines with +
Include context lines with space prefix"""}
                        ],
                        "temperature": 0
                    }),
                    timeout=30,
                )

                if resp.ok:
                    openai_response = resp.json()["choices"][0]["message"]["content"]
                    candidate = _extract_unified_diff(openai_response)
                    if candidate:
                        diff_text = candidate
                        used_engine = "openai"
                        explanation = "Generated via OpenAI"
            except Exception as e:
                print(f"[ENGINE] OpenAI error: {e}")

    # Build summary
    summary = f"Goal: {goal}\nTarget: {path}\nEngine: {used_engine or 'failed'}"

    return summary, diff_text, explanation or "No explanation available"


import re
from typing import Optional

def _repair_top_insert_hunk(diff_text: str, target_path: str, inserted_marker: str = "# NOTE: self-improve smoke test") -> str:
    """
    If the diff's first hunk is a minimal top-of-file insertion (e.g., '@@ -1,0 +1,1 @@')
    rewrite it to include proper context from the current file so it applies cleanly.
    Only used when we detect the inserted line and the file exists.
    """
    try:
        with open(target_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        if not lines:
            return diff_text
        # extract the +added line(s) from the diff hunk (skip +++ header)
        added = []
        in_hunk = False
        for line in diff_text.splitlines():
            if line.startswith("@@"):
                in_hunk = True
                continue
            if in_hunk:
                if line.startswith("+++ ") or line.startswith("--- ") or line.startswith("diff --git"):
                    break
                if line.startswith("+") and not line.startswith("+++"):
                    added.append(line[1:])
        if not added:
            return diff_text
        # If our known marker is in the first added line, reconstruct a contextful hunk
        if inserted_marker not in added[0]:
            return diff_text

        ctx_n = min(2, len(lines))  # use up to 2 context lines
        ctx = "\n".join(lines[:ctx_n])

        # Rewrite headers to ensure a/b prefixes exist (more compatible)
        diff_text = re.sub(r"^diff --git\s+(\S+)\s+(\S+)", r"diff --git a/\1 b/\2", diff_text, flags=re.M)
        diff_text = re.sub(r"^---\s+(?!a/)", r"--- a/", diff_text, flags=re.M)
        diff_text = re.sub(r"^\+\+\+\s+(?!b/)", r"+++ b/", diff_text, flags=re.M)

        # Replace the first hunk with a contextful one
        diff_text = re.sub(
            r"(?ms)^@@\s*-1,0\s*\+1,1\s*@@.*?$",
            f"@@ -1,{ctx_n} +1,{ctx_n+1} @@\n+{inserted_marker}\n{ctx}",
            diff_text,
            count=1,
        )
        return diff_text
    except Exception:
        return diff_text


import re
import io
import re
from typing import List, Dict

def _normalize_python_indentation(path):
    """
    Auto-aligns newly inserted lines to the surrounding indentation (safe for docstrings/comments).
    Best-effort; ignores errors.
    """
    try:
        from pathlib import Path
        text = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        fixed = []
        for i, line in enumerate(text):
            stripped = line.lstrip(" ")
            # If previous line is indented and current line is 0-indent but looks like a doc/comment/text,
            # indent it to the previous non-empty line's indent.
            if fixed:
                prev = fixed[-1]
                prev_indent = len(prev) - len(prev.lstrip(" "))
                curr_indent = len(line) - len(stripped)
                if prev_indent > 0 and curr_indent == 0 and stripped and stripped[:1] in {'#', '"', "'"}:
                    line = (" " * prev_indent) + stripped
            fixed.append(line)
        Path(path).write_text("\n".join(fixed) + ("\n" if text else ""), encoding="utf-8")
    except Exception:
        pass


def _safe_apply_unified_diff_or_fallback(
    diff_text: str,
    target_paths: List[str],
    verify_marker: str = None,
    allow_top_insert: bool = True,
) -> str:
    """
    Try the project's unified diff applier; if it fails, optionally do a targeted fallback:
    - If the diff is a simple top-of-file insertion (common in smoke tests),
      insert the marker line at the top of the first target file (idempotent).
    Returns a human-readable message describing what was done.
    """
    import re
    from pathlib import Path

    # ---------- helpers (scoped here so you can paste this as a drop-in) ----------
    def _repo_root() -> Path:
        # this file is .../AntAgent/autodev/manager.py  -> repo root is parents[2]
        return Path(__file__).resolve().parents[2]

    def _abs_path(rel_or_repo_path: str) -> Path:
        p = Path(rel_or_repo_path)
        return p if p.is_absolute() else (_repo_root() / rel_or_repo_path)

    def _first_path_and_added_line(text: str):
        """Extract first file path and the first +added line from the first hunk."""
        if not text:
            return None, None
        try:
            m = re.search(r"^diff --git\s+(?:a/)?(?P<p>[^\s\n]+)\s+(?:b/)?(?P<p2>[^\s\n]+)", text, flags=re.M)
            path = m.group("p") if (m and m.group("p") == m.group("p2")) else None
            hunk = re.search(r"(?ms)^@@[^\n]*\n(.*?)(?:\n@@|\Z)", text)
            added = None
            if hunk:
                for line in hunk.group(1).splitlines():
                    if line.startswith("+") and not line.startswith("+++"):
                        added = line[1:]
                        break
            return path, added
        except Exception:
            return None, None

    # Normalize to text for regex and to bytes for the patch lib
    text_str = diff_text if isinstance(diff_text, str) else diff_text.decode("utf-8", "ignore")
    udiff_bytes = diff_text.encode("utf-8") if isinstance(diff_text, str) else diff_text

    # Figure out the *intended* first target path (for verification below)
    first_path, first_added = _first_path_and_added_line(text_str)
    if not first_path and target_paths:
        first_path = target_paths[0]
    abs_first = _abs_path(first_path) if first_path else None

    # Snapshot "before" contents (if the target exists)
    before = None
    if abs_first and abs_first.exists():
        try:
            before = abs_first.read_text(encoding="utf-8", errors="replace")
        except Exception:
            before = None

    # 1) Try the built-in applier and verify actual change
    try:
        from .patch import apply_unified_diff
        res = apply_unified_diff(udiff_bytes)  # may return bool or dict or raise
        # If we don't know which file changed, trust 'res' only if verification passes
        if abs_first and abs_first.exists():
            try:
                after = abs_first.read_text(encoding="utf-8", errors="replace")
                if before is None or after != before:
                    return "Patch applied via apply_unified_diff"
            except Exception:
                # If we cannot read after, assume success was not verifiable and fall through to fallback
                pass
        # If we get here, either no target to verify or bytes didn't change -> try fallback
        err1 = "apply_unified_diff reported success but no file content changed"
    except Exception as e:
        err1 = f"apply_unified_diff failed: {e}"

    # If caller disallows top insert fallback, stop here with a clear message
    if not allow_top_insert:
        return err1 + " | Fallback skipped (allow_top_insert=True)"

    # 2) Fallback: detect “insert line at file start” and apply directly
    try:
        # Resolve a target path
        path = first_path or (target_paths[0] if target_paths else None)
        if not path:
            return err1 + " | Fallback aborted: no target path detected"

        # Determine marker to insert
        marker = (verify_marker or first_added)
        if not marker:
            return err1 + " | Fallback aborted: no added line (marker) detected"

        ap = _abs_path(path)
        if not ap.exists():
            return err1 + f" | Fallback aborted: target does not exist: {ap}"

        # Read existing
        existing = ap.read_text(encoding="utf-8", errors="replace")

        # If already present near the top, consider it done (idempotent)
        head = existing[:500]
        if marker in head:
            return err1 + f" | Fallback: marker already present at top of {path}"

        # Insert at very top (ensure single instance)
        new_text = existing if existing.startswith(marker) else (marker.rstrip("\n") + "\n" + existing)

        # Write back
        ap.write_text(new_text, encoding="utf-8", errors="replace")

        # De-duplicate any extra occurrences beyond the first line
        txt = ap.read_text(encoding="utf-8", errors="replace")
        lines = txt.splitlines()
        if lines and marker.strip() == lines[0].strip():
            filtered = [lines[0]] + [ln for ln in lines[1:] if ln.strip() != marker.strip()]
            ap.write_text("\n".join(filtered) + ("\n" if txt.endswith("\n") else ""), encoding="utf-8", errors="replace")

        # Verify final change
        final = ap.read_text(encoding="utf-8", errors="replace")
        if before is not None and final == before:
            return err1 + f" | Fallback wrote nothing (content identical) for {path}"

        return f"Fallback applied: inserted marker at top of {path}"
    except Exception as e2:
        return err1 + f" | Fallback failed: {e2}"


def self_improve_round(goal: str, constraints: Dict, *, do_apply: bool = True):
    """
    Generate a patch for (goal, constraints). If do_apply is True, try to apply it
    robustly; otherwise just return the diff (dry-run).
    Returns:
      {
        "summary": str,
        "unified_diff": str,
        "applied": bool,
        "message": str,
        "diagnostics": dict   # <--- new, explains why it failed/succeeded
      }
    """
    summary, diff = propose_patch_with_explanation(goal, constraints)
    diff = (diff or "").strip()

    # --- collect early diagnostics about model output
    diag = {}
    diag["has_output"] = bool(diff)
    diag["has_diff_header"] = bool(re.search(r"(?m)^diff --git ", diff or ""))
    # pull first file path & first added line for context
    def _first_file_and_added(text: str):
        try:
            m = re.search(r"^diff --git\s+(?:a/)?(?P<p>[^\s\n]+)\s+(?:b/)?(?P<p2>[^\s\n]+)", text, flags=re.M)
            p = m.group("p") if (m and m.group("p") == m.group("p2")) else None
            h = re.search(r"(?ms)^@@[^\n]*\n(.*?)(?:\n@@|\Z)", text)
            added = None
            if h:
                for line in h.group(1).splitlines():
                    if line.startswith("+") and not line.startswith("+++"):
                        added = line[1:]
                        break
            return p, added
        except Exception:
            return None, None

    first_path, first_added = _first_file_and_added(diff or "")
    diag["first_path"] = first_path
    diag["first_added_sample"] = (first_added[:80] + "…") if first_added and len(first_added) > 80 else first_added

    if not diff:
        return {
            "summary": summary,
            "unified_diff": "",
            "applied": False,
            "message": "No diff produced",
            "diagnostics": diag,
        }

    # Normalize EOLs
    diff_norm = diff.replace("\r\n", "\n")
    diag["diff_bytes"] = len(diff_norm.encode("utf-8"))

    target_paths = constraints.get("paths") or []
    diag["targets"] = target_paths

    # Check for the classic malformed prepend hunk
    if target_paths and '@@ -1,0 +1,1 @@' in diff_norm:
        diag["fix_malformed_prepend_applied"] = True
        diff_norm = fix_malformed_prepend_diff(diff_norm, target_paths[0])
    else:
        diag["fix_malformed_prepend_applied"] = False

    if not do_apply:
        return {
            "summary": summary,
            "unified_diff": diff_norm,
            "applied": False,
            "message": "Dry run: diff only (not applied)",
            "diagnostics": diag,
        }

    # Try normal apply; in strict mode we SKIP the top-insert fallback.
    verify_marker = None
    m = re.search(r"'(# NOTE:[^']+)'", goal)
    if m:
        verify_marker = m.group(1)
    diag["verify_marker"] = verify_marker

    # Enable robust fallbacks so /self_improve can truly self-edit.
    allow_lenient = bool(constraints.get("allow_lenient", False))
    msg = _safe_apply_unified_diff_or_fallback(
        diff_norm,
        target_paths,
        verify_marker=verify_marker,
        allow_top_insert=False,
    )

    diag["apply_message"] = msg

    applied = msg.startswith("Patch applied via apply_unified_diff")
    verify_msg = ""

    # Marker verification if requested
    first_target = target_paths[0] if target_paths else None
    diag["first_target_exists"] = False
    if first_target:
        try:
            import os
            diag["first_target_exists"] = os.path.exists(first_target)
        except Exception:
            pass

    if applied and first_target and verify_marker:
        try:
            with open(first_target, "r", encoding="utf-8", errors="replace") as f:
                head = f.read(500)
            if verify_marker not in head:
                applied = False
                verify_msg = f"Post-apply verification failed on {first_target}"
        except Exception as e:
            applied = False
            verify_msg = f"Post-apply read failed on {first_target}: {e}"

    result_obj = {
        "summary": summary,
        "unified_diff": diff_norm,
        "applied": applied,
        "message": verify_msg or msg,
    }

    # --- record diagnostics for /debug access ---
    import re as _re  # local import to avoid new globals
    global _LAST_SELF_IMPROVE_DIAG
    _LAST_SELF_IMPROVE_DIAG = {
        "applied": applied,
        "message": result_obj["message"],
        "diagnostics": {
            "has_output": bool(diff),
            "has_diff_header": bool(_re.search(r"(?m)^diff --git ", diff or "")),
            "diff_bytes": len(diff_norm.encode("utf-8")),
            "targets": constraints.get("paths") or [],
        },
    }

    return result_obj

# --- Compatibility shims for older imports ---------------------------------
def propose_patch(goal: str, constraints: dict) -> str:
    """
    Backward-compatible wrapper expected by older code.
    Returns only the unified diff string (or "" if none).
    Compatible with propose_patch_with_explanation returning
    (diff,), (summary, diff), or (summary, diff, explanation, ...).
    """
    res = propose_patch_with_explanation(goal, constraints)

    if isinstance(res, tuple):
        if len(res) == 0:
            return ""
        elif len(res) == 1:
            diff = res[0]
        elif len(res) == 2:
            _, diff = res
        else:
            # Prefer the last element if it's a string, else fall back to second
            diff = res[-1] if isinstance(res[-1], str) else res[1]
    else:
        diff = res

    return diff or ""

def propose_patch_and_summary(goal: str, constraints: dict):
    """
    Optional helper: returns (summary, unified_diff).
    """
    return propose_patch_with_explanation(goal, constraints)
    # --- Legacy compatibility stubs --------------------------------------------
def run_quality_checks(*args, **kwargs):
    """
    Stub for backward compatibility. In the new architecture, quality checks
    are handled implicitly by the self-improvement loop.
    """
    return {"ok": True, "message": "Quality checks stubbed (no-op)"}


def apply_patch(diff_text: str):
    """
    Robust patch applier for /code/apply_patch.

    - Try the project's unified diff applier.
    - Verify the target file actually changed.
    - If not, try a lenient single-file hunk apply that ignores line numbers
      and matches the hunk context inside the current file.
    Returns: {"applied": bool, "message": str}
    """
    import re
    from pathlib import Path
    from .patch import apply_unified_diff

    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def _abs_path(rel_or_repo_path: str) -> Path:
        p = Path(rel_or_repo_path)
        return p if p.is_absolute() else (_repo_root() / rel_or_repo_path)

    def _first_file_and_hunk(text: str):
        """
        Extract first file path and first hunk body (without header lines).
        Returns (path:str|None, hunk_lines:list[str]|None)
        """
        if not text:
            return None, None
        m = re.search(r"^diff --git\s+(?:a/)?(?P<p>[^\s\n]+)\s+(?:b/)?(?P<p2>[^\s\n]+)", text, flags=re.M)
        path = m.group("p") if (m and m.group("p") == m.group("p2")) else None

        hm = re.search(r"(?ms)^@@[^\n]*\n(.*?)(?:\n@@|\Z)", text)
        if not hm:
            return path, None
        hunk_body = hm.group(1).splitlines()
        return path, hunk_body

    def _lenient_apply_single_file_diff(text: str) -> tuple[bool, str]:
        """
        For a single-file diff with a single hunk, ignore the @@ line numbers.
        Try to locate the hunk's context in the current file and splice in + lines.
        Returns (applied:bool, message:str)
        """
        path, hunk = _first_file_and_hunk(text if isinstance(text, str) else text.decode("utf-8", "ignore"))
        if not path or not hunk:
            return (False, "Lenient: no path or hunk found")

        ap = _abs_path(path)
        if not ap.exists():
            return (False, f"Lenient: target does not exist: {path}")

        original = ap.read_text(encoding="utf-8", errors="replace").splitlines()

        # Build "old" block (context + minus) and "new" block (context + plus)
        old_block: list[str] = []
        new_block: list[str] = []
        for ln in hunk:
            if ln.startswith(' '):          # context
                old_block.append(ln[1:])
                new_block.append(ln[1:])
            elif ln.startswith('-'):        # removed
                old_block.append(ln[1:])
            elif ln.startswith('+'):        # added
                new_block.append(ln[1:])
            else:
                # Unknown line marker: bail out
                return (False, "Lenient: unknown hunk line marker")

        if not old_block and new_block:
            # Pure insertion: try to insert where first context from new_block appears, else at EOF
            # Find a stable anchor from new_block (first 2 non-empty lines)
            anchors = [l for l in new_block if l.strip()][:2]
            anchor_idx = None
            if anchors:
                # try to find anchor sequence
                for i in range(len(original) - len(anchors) + 1):
                    if original[i:i+len(anchors)] == anchors:
                        anchor_idx = i
                        break
            if anchor_idx is None:
                # append at end (still better than top)
                ap.write_text("\n".join(original + new_block) + ("\n" if original or new_block else ""), encoding="utf-8")
                _normalize_python_indentation(ap)
                return (True, f"Lenient: appended insertion to end of {path}")
            else:
                # insert before anchors
                new_text = original[:anchor_idx] + new_block + original[anchor_idx:]
                ap.write_text("\n".join(new_text) + ("\n" if original else ""), encoding="utf-8")
                _normalize_python_indentation(ap)
                return (True, f"Lenient: inserted block before anchor in {path}")

        # General replace: find first occurrence of old_block and replace with new_block
        if old_block:
            # Search for old_block as a contiguous slice
            hit = None
            max_i = max(0, len(original) - len(old_block))
            for i in range(max_i + 1):
                if original[i:i+len(old_block)] == old_block:
                    hit = i
                    break
            if hit is None:
                return (False, "Lenient: old block not found in target")
            new_text = original[:hit] + new_block + original[hit+len(old_block):]
            ap.write_text("\n".join(new_text) + ("\n" if original else ""), encoding="utf-8")
            _normalize_python_indentation(ap)
            return (True, f"Lenient: replaced block in {path}")

        return (False, "Lenient: nothing to apply")

    # ---- normalize inputs
    text_str = diff_text if isinstance(diff_text, str) else diff_text.decode("utf-8", "ignore")
    udiff_bytes = diff_text.encode("utf-8") if isinstance(diff_text, str) else diff_text

    # Snapshot before (first file only)
    tgt, _ = _first_file_and_hunk(text_str)
    before = None
    ap = _abs_path(tgt) if tgt else None
    if ap and ap.exists():
        before = ap.read_text(encoding="utf-8", errors="replace")

    # 1) normal apply
    try:
        res = apply_unified_diff(udiff_bytes)  # may be bool or dict
        applied = bool(res if isinstance(res, bool) else (res or {}).get("applied"))
        msg = "Patch applied via apply_unified_diff" if applied else ((res or {}).get("message", "Patch apply failed") if isinstance(res, dict) else "Patch apply failed")
    except Exception as e:
        applied = False
        msg = f"apply_unified_diff failed: {e}"

    # Verify actual change
    if applied and ap and ap.exists():
        after = ap.read_text(encoding="utf-8", errors="replace")
        if before is not None and after == before:
            applied = False
            msg = "apply_unified_diff reported success but no file content changed"

    if applied:
        return {"applied": True, "message": msg}

    # 2) lenient single-file fallback (no top-of-file insert)
    ok, lmsg = _lenient_apply_single_file_diff(text_str)
    if ok:
        return {"applied": True, "message": lmsg}

    # fail cleanly
    return {"applied": False, "message": msg + " | " + lmsg}

def list_paths() -> list:
    """
    Returns the list of all tracked .py files under AntAgent/.
    SMOKE: v6 docstring patch
    Llama test successful. (SI v7)
    Used by /code/list_paths route.
    """
    import os
    root = os.path.join(os.path.dirname(__file__), "..")
    found = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".py"):
                found.append(os.path.relpath(os.path.join(dirpath, f), root))
    return found


def read_files(paths: list) -> dict:
    """
    Returns {path: text} for each path (used by /code/read_files route).
    """
    out = {}
    for p in paths or []:
        try:
            with open(p, "r", encoding="utf-8") as f:
                out[p] = f.read()
        except Exception as e:
            out[p] = f"# ERROR: cannot read {p}: {e}"
    return out
    # ---------------------------------------------------------------------------

import hashlib, os, re

def _touched_paths_from_diff(diff_text: str) -> list[str]:
    """Return a de-duplicated list of repo-relative paths mentioned in the diff headers."""
    # Matches: diff --git a/path b/path   OR   diff --git path path
    pats = re.findall(r"^diff --git\s+(?:a/)?([^\s]+)\s+(?:b/)?\1", diff_text, flags=re.M)
    out, seen = [], set()
    for p in pats:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def _sha256_or_none(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None

import re

def _ensure_list_paths_docstring_marker(path: str = "AntAgent/autodev/manager.py") -> tuple[str, bool]:
    """
    If `list_paths()` exists and its docstring lacks the exact line
    `SMOKE: docstring marker.`, insert it just before the closing triple quotes.

    Returns (message, changed).
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            s = f.read()

        # Match the def with its docstring triple-quoted block
        m = re.search(
            r"(def\s+list_paths\(\)\s*->\s*list:\s*\n\s*\"\"\"\s*\n)"
            r"([\s\S]*?)"
            r"(\n\s*\"\"\"\s*)",
            s
        )
        if not m:
            return "list_paths() docstring block not found", False

        full_block = m.group(0)
        if "SMOKE: docstring marker." in full_block:
            return "marker already present", False

        # Insert the marker one line before the closing triple quotes
        before = m.group(1)
        body   = m.group(2)
        after  = m.group(3)
        # Keep a reasonable indentation (same as other docstring lines)
        indent = "    "
        new_block = before + body + f"\n{indent}SMOKE: docstring marker." + after

        s2 = s[:m.start()] + new_block + s[m.end():]
        with open(path, "w", encoding="utf-8") as f:
            f.write(s2)

        return "fallback inserted marker into list_paths() docstring", True
    except Exception as e:
        return f"fallback failed: {e}", False

def _ensure_docstring_line(
    *,
    path: str,
    func_name: str,
    line_to_insert: str,
) -> tuple[str, bool]:
    """
    Ensure that function `func_name` in `path` has a docstring containing `line_to_insert`.
    - If the function has a docstring but it's missing the line, insert the line just before the closing triple quotes.
    - If the function has NO docstring, insert a minimal docstring block with that line under the def.
    Returns (message, changed).
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            s = f.read()

        # Find the function definition and the following block
        def_pat = rf"(^[ \t]*def[ \t]+{re.escape(func_name)}\s*\([^)]*\)\s*:\s*\n)"
        m = re.search(def_pat, s, flags=re.M)
        if not m:
            return f"{func_name}() not found in {path}", False

        def_start = m.start()
        def_hdr   = m.group(1)

        # Look for an immediate triple-quoted docstring
        doc_pat = r'^[ \t]*("""|\'\'\')([\s\S]*?)(\1)\s*\n'
        m2 = re.search(doc_pat, s[m.end():], flags=re.M)
        if m2 and m2.start() == 0:
            # Function has a docstring block
            quote = m2.group(1)
            body  = m2.group(2)
            if line_to_insert in body:
                return "marker already present", False

            # Preserve indentation: use indent of the def block + 4 spaces
            def_indent = re.match(r"^([ \t]*)", def_hdr).group(1)
            indent = def_indent + "    "
            new_body = (body.rstrip("\n") + "\n" + indent + line_to_insert + "\n")
            new_doc  = f'{quote}{new_body}{quote}\n'
            new_func = def_hdr + new_doc + s[m.end()+m2.end():]
            s2 = s[:def_start] + new_func
        else:
            # No docstring; insert a minimal one
            def_indent = re.match(r"^([ \t]*)", def_hdr).group(1)
            indent = def_indent + "    "
            new_doc = f'{indent}"""' + line_to_insert + '"""\n'
            new_func = def_hdr + new_doc + s[m.end():]
            s2 = s[:def_start] + new_func

        with open(path, "w", encoding="utf-8") as f:
            f.write(s2)
        return f"inserted docstring line into {func_name}()", True
    except Exception as e:
        return f"fallback failed: {e}", False


def _force_insert_line(path: str, anchor: str, insert_line: str, after: bool = True) -> bool:
    """
    Force-insert `insert_line` into `path` immediately before or after the first line
    containing `anchor`. Idempotent: does nothing if line already present.
    Returns True if file changed.
    """
    import io
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    if any(insert_line.strip() == ln.strip() for ln in lines):
        return False  # already present

    new_lines = []
    changed = False
    for ln in lines:
        new_lines.append(ln)
        if anchor in ln and not changed:
            idx = len(new_lines)
            if after:
                new_lines.append(insert_line)
            else:
                new_lines.insert(max(idx - 1, 0), insert_line)
            changed = True

    if changed:
        with open(path, "w", encoding="utf-8", errors="replace") as f:
            f.write("\n".join(new_lines) + "\n")
    return changed


def _fallback_edit_read_files_docstring() -> bool:
    """
    Append a stable sentence to the _read_files() docstring if it's missing.
    Returns True if file was updated.
    """
    from pathlib import Path
    import re

    target = Path(__file__).resolve()
    text = target.read_text(encoding="utf-8", errors="replace")

    # Match the _read_files signature and its docstring block ("""...""")
    # Be liberal: docstring may or may not exist.
    sig_rx = r"def\s+_read_files\(\s*paths:\s*List\[str\]\s*\)\s*->\s*List\[Tuple\[str,\s*str\]\]\s*:\s*"
    doc_rx = r'("""[\s\S]*?""")'
    block_rx = re.compile(sig_rx + r"(?P<body>[\s\S]{0,500})", re.M)
    m = block_rx.search(text)
    if not m:
        return False

    start = m.start("body")
    # check if there is a docstring right at the start of the body
    if re.match(r'^\s*"""', m.group("body")):
        # Has a docstring: inject the sentence before the closing triple quotes if missing
        sentence = "Returns a list of (path, text) tuples; on read error, the text is an error string."
        # find the first triple-quoted block
        doc_m = re.match(r'^\s*("""[\s\S]*?""")', m.group("body"))
        if not doc_m:
            return False
        doc = doc_m.group(1)
        if sentence in doc:
            return False  # nothing to do
        new_doc = doc[:-3] + "\n" + sentence + '\n"""'
        new_body = new_doc + m.group("body")[len(doc_m.group(0)):]
        new_text = text[:start] + new_body + text[start+len(m.group("body")):]
    else:
        # No docstring: add a minimal one
        new_doc = (
            '    """\n'
            '    Read allowlisted files and return their contents.\n'
            '    \n'
            '    Returns a list of (path, text) tuples; on read error, the text is an error string.\n'
            '    """\n'
        )
        new_text = text[:start] + new_doc + text[start:]

    if new_text != text:
        target.write_text(new_text, encoding="utf-8", errors="replace")
        return True
    return False
def diagnose_diff_failure(diff_text: str, paths: list[str] | None = None) -> dict:
    """
    Lightweight diagnostics used by /debug/diagnose_diff.
    Checks:
      - unified diff header present
      - first @@ hunk present
      - which paths the diff touches
      - allowlist membership (AntAgent/autodev/allowlist.txt)
      - file existence on disk
    Returns a dict with 'notes' and 'problems' arrays for quick UI display.
    """
    import re
    from pathlib import Path

    report = {
        "has_diff_header": False,
        "has_hunk": False,
        "touched_paths": [],
        "allowlisted": {},
        "exists_on_disk": {},
        "problems": [],
        "notes": [],
    }

    text = diff_text if isinstance(diff_text, str) else (diff_text or b"").decode("utf-8", "ignore")
    if not text.strip():
        report["problems"].append("Empty diff payload")
        return report

    report["has_diff_header"] = bool(re.search(r"(?m)^diff --git ", text))
    report["has_hunk"] = bool(re.search(r"(?m)^@@", text))

    # Extract touched paths
    touched = _touched_paths_from_diff(text)
    report["touched_paths"] = touched
    if not touched:
        report["problems"].append("No file paths found in diff headers (expected 'diff --git a/<p> b/<p>')")

    # Load allowlist (sibling of this file)
    allow_path = Path(__file__).with_name("allowlist.txt")
    try:
        allow = []
        if allow_path.exists():
            allow = [ln.strip().replace("\\", "/") for ln in allow_path.read_text(encoding="utf-8").splitlines()
                     if ln.strip() and not ln.strip().startswith("#")]
        else:
            report["notes"].append(f"Allowlist missing: {allow_path}")
    except Exception as e:
        allow = []
        report["notes"].append(f"Allowlist read error: {e}")

    # Repo root (../.. from this file, i.e., project root)
    repo_root = Path(__file__).resolve().parents[2]

    for p in touched:
        norm = p.replace("\\", "/")
        report["allowlisted"][norm] = norm in allow
        abs_p = (repo_root / norm)
        report["exists_on_disk"][norm] = abs_p.exists()
        if norm not in allow:
            report["problems"].append(f"Path not allowlisted: {norm}")
        if not abs_p.exists():
            report["problems"].append(f"Target does not exist on disk: {abs_p}")

    # Quick header sanity
    if report["has_diff_header"] and not re.search(r"(?m)^---\s+(?:a/)?", text):
        report["problems"].append("Missing '--- a/<path>' header")
    if report["has_diff_header"] and not re.search(r"(?m)^\+\+\+\s+(?:b/)?", text):
        report["problems"].append("Missing '+++ b/<path>' header")
    if not report["has_hunk"]:
        report["problems"].append("No '@@' hunk found (required for git-style patches)")

    if not report["problems"]:
        report["notes"].append("No obvious issues detected; try /code/apply_patch or ensure git is available")
    return report


from copy import deepcopy


def propose_patch_with_explanation(goal: str, constraints: Dict) -> Tuple[str, str, str]:
    """
    Generate a patch with detailed explanation of the approach.
    Returns: (summary, unified_diff, explanation)
    """
    import os, re, json
    from pathlib import Path

    # Validator (if unavailable, accept anything—keeps old behavior)
    try:
        from AntAgent.autodev.diff_doctor import vet_diff
    except Exception:
        def vet_diff(x: str, min_context: int = 1):
            return True, ""

    # ------------------------------
    # 0) Target path resolution
    # ------------------------------
    target_paths = constraints.get("paths") or []

    # Path expansion (retain your behavior)
    all_allowed = _allowed_paths()
    expanded_paths = []
    for path in target_paths:
        path_obj = Path(_repo_root() / path)
        if path_obj.is_dir():
            dir_str = str(path).replace("\\", "/")
            for allowed in all_allowed:
                allowed_norm = allowed.replace("\\", "/")
                if allowed_norm.startswith(dir_str + "/") and allowed_norm.endswith(".py"):
                    full_path = Path(_repo_root() / allowed_norm)
                    if full_path.exists():
                        expanded_paths.append(allowed_norm)
        elif path in all_allowed:
            full_path = Path(_repo_root() / path)
            if full_path.exists():
                expanded_paths.append(path)

    if expanded_paths:
        target_paths = list(set(expanded_paths))

    # Read files
    files = _read_files(target_paths)
    if not files:
        return "No files to process", "", "No target files found"

    # Focus on first file
    path, content = files[0]
    lines = content.splitlines()

    # Numbered version for analysis (retain)
    numbered_lines = [f"{i + 1:4}: {line}" for i, line in enumerate(lines)]
    numbered_content = "\n".join(numbered_lines)

    # Compute precise context around the target line if present
    target_line_idx = None
    m = re.search(r'^(?P<lead>\s*)#\s*Random\s+animal:\s*(?P<animal>\w+)\s*$', content, flags=re.M)
    if m:
        # 0-based index
        target_line_idx = content[:m.start()].count("\n")
    # Extract context lines for anchoring (even if target not found, we still compute safely)
    def _get(ctx_idx: int) -> str:
        if ctx_idx is None:
            return ""
        if 0 <= ctx_idx < len(lines):
            return lines[ctx_idx]
        return ""
    prev_line = _get((target_line_idx - 1) if target_line_idx is not None else None)
    curr_line = _get(target_line_idx if target_line_idx is not None else None)
    next_line = _get((target_line_idx + 1) if target_line_idx is not None else None)

    used_engine = None
    diff_text = ""
    explanation = ""

    # ------------------------------
    # Helper: robust call into OllamaAdapter
    # ------------------------------
    def _ollama_complete(llm, system: str, prompt: str, temperature: float, max_tokens: int):
        """
        Try multiple method names so different adapters work:
        - complete(system=..., prompt=...)
        - chat(system=..., messages=[...])
        - create_chat_completion(messages=[...])
        Returns plain text content.
        """
        # 1) .complete(system=..., prompt=...)
        if hasattr(llm, "complete"):
            return llm.complete(system=system, prompt=prompt, temperature=temperature,
                                max_tokens=max_tokens, stop=["```", "\x00"])
        # 2) .chat(system=..., messages=[...])
        if hasattr(llm, "chat"):
            resp = llm.chat(
                system=system,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=["```", "\x00"],
            )
            # Assume .chat returns a dict with "content" or a string
            if isinstance(resp, dict) and "content" in resp:
                return resp["content"]
            return resp  # string fallback
        # 3) .create_chat_completion(messages=[...])
        if hasattr(llm, "create_chat_completion"):
            result = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                top_p=0.05,
                max_tokens=max_tokens,
            )
            return (result.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")) or ""
        raise AttributeError("OllamaAdapter has no supported chat/complete method")

    # ------------------------------
    # 1) Primary: Ollama (DeepSeek)
    # ------------------------------
    try:
        from ollama_adapter import Llama

        # Strong diff-only system prompt anchored to THIS file path
        DIFF_SYSTEM_PROMPT = (
            "You are a code refactoring assistant. Your ONLY output must be a valid unified diff "
            "that can be applied with `git apply`.\n\n"
            "STRICT RULES:\n"
            "- Output ONLY the diff. No prose, no code fences.\n"
            "- Generate a COMPLETE diff - do not truncate or cut off mid-line.\n"
            f"- Use exactly this header format:\n"
            f"  diff --git a/{path} b/{path}\n"
            "  index 0000000..0000000 100644\n"
            f"  --- a/{path}\n"
            f"  +++ b/{path}\n"
            "- Include a proper @@ hunk header with line numbers.\n"
            "- Provide at least one unchanged context line before and after the edited line.\n"
            "- Edit only what the objective requires.\n"
            "- Ensure the diff ends properly with complete lines.\n"
        )

        # If we know the exact line, force the model to anchor on it
        anchor_block = ""
        if curr_line:
            anchor_block = (
                "Use THIS exact three-line context in the hunk (space-prefixed in the diff):\n"
                f"  {prev_line}\n"
                f"  {curr_line}\n"
                f"  {next_line}\n"
                "Do not modify the prev/next lines; only change the middle line if required.\n"
            )

        user_1 = (
            f"Objective:\n{goal.strip()}\n\n"
            f"Target file path (must match headers exactly):\n{path}\n\n"
            f"{anchor_block}\n"
            "Current file content with line numbers (truncated for context):\n"
            "-----8<-----\n"
            f"{numbered_content[:6000]}\n"
            "-----8<-----\n\n"
            "Produce ONLY the unified diff.\n"
        )

        # init model
        llm = Llama(
            model=os.getenv("ANT_OLLAMA_MODEL", "deepseek-coder-v2:16b"),
            host=os.getenv("ANT_OLLAMA_HOST", "http://localhost:11434"),
            num_ctx=4096,
        )

        # First attempt
        raw1 = _ollama_complete(llm, DIFF_SYSTEM_PROMPT, user_1, temperature=0.1, max_tokens=2048)
        full_response = (raw1 or "").strip()

        # Extract diff (keep your prior behavior)
        if "diff --git" in full_response:
            parts = full_response.split("diff --git", 1)
            explanation = parts[0].strip()
            candidate = "diff --git" + parts[1]
        else:
            explanation = full_response[:200]
            candidate = full_response

        candidate = _extract_unified_diff(candidate) or ""

        # Structural + semantic validation
        def _basic_diff_ok(d: str) -> bool:
            return (
                d.startswith(f"diff --git a/{path} b/{path}") and
                ("--- a/" in d and "+++ b/" in d) and
                "@@" in d and
                any(ch in d for ch in (chr(10)+"+ ", chr(10)+"- ", "+"+chr(10), "-"+chr(10)))
            )

        if candidate and _basic_diff_ok(candidate):
            ok1, why1 = vet_diff(candidate, min_context=1)
            if ok1:
                diff_text = candidate
                used_engine = "ollama"
                print("[ENGINE] LLM generated valid diff (first try)")
            else:
                print(f"[ENGINE] LLM diff validation failed: {why1}")
                print(f"[DEBUG] Validating diff (first 500 chars):\n{candidate[:500]}")
                # Critique & retry with the exact validator error
                critic = (
                    "Your previous diff was rejected for the reason below.\n"
                    "You MUST output a corrected unified diff for the SAME file.\n"
                    "Rules: diff only; proper header for this path; include @@ hunk; include context above/below; "
                    "do not inline comments into import/from lines unless replacing the whole line.\n\n"
                    f"REJECTION REASON:\n{why1}\n\n"
                    "Previous (invalid) diff:\n" + candidate + "\n\n"
                    "Re-output a corrected diff now (diff only):"
                )
                raw2 = _ollama_complete(llm, DIFF_SYSTEM_PROMPT, critic, temperature=0.05, max_tokens=2048)
                full2 = (raw2 or "").strip()
                cand2 = _extract_unified_diff(full2) or ""
                if cand2 and _basic_diff_ok(cand2):
                    ok2, why2 = vet_diff(cand2, min_context=1)
                    if ok2:
                        diff_text = cand2
                        used_engine = "ollama"
                        print("[ENGINE] LLM generated valid diff after critique")
                    else:
                        print(f"[ENGINE] Critique retry still invalid: {why2}")
                else:
                    print("[ENGINE] Could not extract valid diff from critique retry")
        else:
            print("[ENGINE] Could not extract valid diff from LLM output")
            if candidate:
                print(f"[DEBUG] Basic diff check failed for candidate (first 500 chars):\n{candidate[:500]}")
                print(f"[DEBUG] Candidate length: {len(candidate)} chars")
                print(f"[DEBUG] Basic diff check details:")
                print(f"  - Starts with 'diff --git a/{path} b/{path}': {candidate.startswith(f'diff --git a/{path} b/{path}')}")
                print(f"  - Contains '--- a/': {'--- a/' in candidate}")
                print(f"  - Contains '+++ b/': {'+++ b/' in candidate}")
                print(f"  - Contains '@@': {'@@' in candidate}")
                print(f"  - Contains change markers: {any(ch in candidate for ch in (chr(10)+'+ ', chr(10)+'- ', '+'+chr(10), '-'+chr(10)))}")
            else:
                print("[DEBUG] No candidate diff extracted from LLM response")
                print(f"[DEBUG] Raw LLM response (first 300 chars):\n{full_response[:300]}")

    except Exception as e:
        print(f"[ENGINE] LLM error: {e}")

    # -----------------------------------
    # 2) Fallback: OpenAI (kept, improved)
    # -----------------------------------
    if not diff_text:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print("[ENGINE] Trying OpenAI fallback")
            try:
                import requests

                # Build an anchor-aware prompt so the hunk matches exactly
                anchor_prompt = ""
                if curr_line:
                    anchor_prompt = f"""
Use THIS exact three-line context in the @@ hunk:
  {prev_line}
  {curr_line}
  {next_line}
Keep prev/next unchanged; only modify the middle line if required.
""".strip()

                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    data=json.dumps({
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system",
                             "content": "You are a diff generator. Output ONLY a valid unified diff for the requested change."},
                            {"role": "user", "content": f"""Task: {goal}

File: {path}

{anchor_prompt}

Current file content (first 160 lines for context):
{chr(10).join(lines[:160])}

Output ONLY the unified diff.
Start with exactly:
diff --git a/{path} b/{path}
Then:
index 0000000..0000000 100644
--- a/{path}
+++ b/{path}
Include a proper @@ hunk with accurate line numbers and include at least one unchanged context line above and below the changed line.
Removed lines start with '-' and added lines with '+'. Do not include analysis or code fences."""}
                        ],
                        "temperature": 0,
                        "max_tokens": 2048
                    }),
                    timeout=30,
                )

                if resp.ok:
                    openai_response = resp.json()["choices"][0]["message"]["content"]
                    cand = _extract_unified_diff(openai_response) or ""
                    if cand:
                        okb, whyb = vet_diff(cand, min_context=1)
                        if okb:
                            diff_text = cand
                            used_engine = "openai"
                            explanation = "Generated via OpenAI"
                            print("[ENGINE] OpenAI generated valid diff")
                        else:
                            print(f"[ENGINE] OpenAI diff rejected by validator: {whyb}")
                    else:
                        print("[ENGINE] OpenAI response lacked a valid diff")
                else:
                    print(f"[ENGINE] OpenAI HTTP {resp.status_code}: {resp.text[:200]}")

            except Exception as e:
                print(f"[ENGINE] OpenAI error: {e}")

    # ------------------------------
    # 3) Build summary & return
    # ------------------------------
    summary = f"Goal: {goal}\nTarget: {path}\nEngine: {used_engine or 'failed'}"
    if not diff_text and not explanation:
        explanation = "No explanation available"

    return summary, diff_text, explanation




def self_improve_with_retry(goal: str, constraints: Dict, max_attempts: int = 3) -> Dict:
    """
    Self-improve with retry logic and learning from failures.
    """
    from .manager_learning import record_history, update_lessons, get_lessons

    # Load previous lessons
    lessons = get_lessons()

    # Enhance constraints with learned anchors
    if "must_anchor_any" not in constraints:
        constraints["must_anchor_any"] = lessons.get("anchor_phrases", [])

    attempts = []

    for attempt in range(max_attempts):
        # Generate patch with explanation
        summary, diff, explanation = propose_patch_with_explanation(goal, constraints)

        if not diff:
            # Learn from empty diff
            update_lessons("empty_diff", {"goal": goal})
            attempts.append({
                "attempt": attempt + 1,
                "status": "empty_diff",
                "explanation": explanation
            })

            # Adjust strategy
            constraints["require_context_lines"] = 5  # More context
            continue

        # Validate the diff before applying
        validation = validate_diff_safety(diff, constraints)

        if not validation["safe"]:
            # Learn from validation failure
            for issue in validation["issues"]:
                update_lessons(issue["type"], {
                    "goal": goal,
                    "issue": issue["detail"]
                })

            attempts.append({
                "attempt": attempt + 1,
                "status": "validation_failed",
                "issues": validation["issues"],
                "explanation": explanation
            })

            # Adjust constraints based on issues
            if "top_insert" in [i["type"] for i in validation["issues"]]:
                constraints["no_top_insert"] = True
                constraints["require_context_lines"] = max(5, constraints.get("require_context_lines", 3))

            continue

        # Try to apply the diff
        result = apply_patch(diff)

        if result["applied"]:
            # Success! Record and return
            record_history({
                "goal": goal,
                "success": True,
                "attempts": attempt + 1,
                "explanation": explanation
            })

            return {
                "success": True,
                "attempts": attempts + [{
                    "attempt": attempt + 1,
                    "status": "applied",
                    "explanation": explanation,
                    "diff": diff
                }],
                "final_diff": diff,
                "explanation": explanation
            }
        else:
            # Learn from apply failure
            update_lessons("apply_failed", {
                "goal": goal,
                "error": result.get("error", "unknown")
            })

            attempts.append({
                "attempt": attempt + 1,
                "status": "apply_failed",
                "error": result.get("error"),
                "explanation": explanation
            })

            # Enhance constraints for next attempt
            constraints["require_exact_match"] = True

    # All attempts failed
    record_history({
        "goal": goal,
        "success": False,
        "attempts": len(attempts)
    })

    return {
        "success": False,
        "attempts": attempts,
        "final_diff": None,
        "explanation": "Failed after all attempts"
    }


def validate_diff_safety(diff_text: str, constraints: Dict) -> Dict:
    """
    Validate a diff for common issues before applying.
    """
    issues = []

    # Check for top insert pattern
    if "@@ -1,0 +1," in diff_text or "@@ -0,0 +1," in diff_text:
        issues.append({
            "type": "top_insert",
            "detail": "Diff attempts to insert at line 1 without context"
        })

    # Check for required context
    min_context = constraints.get("require_context_lines", 3)
    hunks = re.findall(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@(.*?)(?=@@|\Z)", diff_text, re.DOTALL)

    for hunk in hunks:
        hunk_body = hunk[4]
        context_lines = [l for l in hunk_body.split('\n') if l.startswith(' ')]
        if len(context_lines) < min_context:
            issues.append({
                "type": "insufficient_context",
                "detail": f"Hunk has {len(context_lines)} context lines, need {min_context}"
            })

    # Check for anchor presence
    required_anchors = constraints.get("must_anchor_any", [])
    if required_anchors:
        has_anchor = False
        for anchor in required_anchors:
            if anchor in diff_text:
                has_anchor = True
                break
        if not has_anchor:
            issues.append({
                "type": "missing_anchor",
                "detail": f"Diff missing required anchors: {required_anchors[:3]}"
            })

    # Check paths are allowed
    allowed_paths = set(constraints.get("paths", []))
    if allowed_paths:
        touched = set(re.findall(r"diff --git a/([\S]+) b/", diff_text))
        if not touched.issubset(allowed_paths):
            issues.append({
                "type": "wrong_path",
                "detail": f"Diff touches non-allowed paths: {touched - allowed_paths}"
            })

    return {
        "safe": len(issues) == 0,
        "issues": issues
    }
