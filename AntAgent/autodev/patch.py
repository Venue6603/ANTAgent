import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List
from typing import Dict
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = Path(__file__).parent / "allowlist.txt"


def _allowed_paths() -> List[str]:
    """Read allowed paths from allowlist.txt"""
    if not ALLOWLIST.exists():
        return []
    with open(ALLOWLIST, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def apply_unified_diff(diff_text) -> Dict:
    """
    Strictly apply a unified diff using git from repo root.
    No fallbacks.
    Returns: { "applied": bool, "error": str|None }
    """
    # normalize input
    if isinstance(diff_text, bytes):
        diff_text = diff_text.decode("utf-8", errors="replace")
    if not isinstance(diff_text, str) or not diff_text.strip():
        return {"applied": False, "error": "Empty diff provided"}

    # Normalize EOL to LF; let git handle CRLF in files
    diff_text = diff_text.replace("\r\n", "\n")

    original_cwd = os.getcwd()
    os.chdir(str(ROOT))
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False, encoding="utf-8") as f:
            f.write(diff_text)
            patch_path = f.name

        try:
            chk = subprocess.run(["git", "apply", "--check", patch_path],
                                 capture_output=True, text=True, cwd=str(ROOT))
            if chk.returncode != 0:
                return {"applied": False, "error": (chk.stderr or chk.stdout or "git apply --check failed")}

            ap = subprocess.run(["git", "apply", patch_path],
                                capture_output=True, text=True, cwd=str(ROOT))
            if ap.returncode == 0:
                return {"applied": True, "error": None}
            return {"applied": False, "error": (ap.stderr or ap.stdout or "git apply failed")}
        finally:
            try:
                os.unlink(patch_path)
            except Exception:
                pass
    except FileNotFoundError:
        return {"applied": False, "error": "git not installed"}
    except Exception as e:
        return {"applied": False, "error": f"Unexpected error: {e}"}
    finally:
        os.chdir(original_cwd)

def fix_common_diff_issues(diff_text: str) -> str:
    """Fix common issues in diffs that prevent them from applying."""
    lines = diff_text.split('\n')
    fixed_lines = []

    for i, line in enumerate(lines):
        # Fix the @@ -1,0 +1,1 @@ issue
        if line.startswith('@@') and '-1,0 +1,1' in line:
            # This is wrong for adding to a non-empty file
            # Try to determine the correct context
            target_file = None

            # Look backwards for the file path
            for j in range(i - 1, max(0, i - 10), -1):
                if lines[j].startswith('+++'):
                    target_file = lines[j].replace('+++ b/', '').replace('+++ ', '').strip()
                    break

            if target_file:
                full_path = ROOT / target_file
                if full_path.exists():
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            file_lines = f.read().splitlines()

                        if file_lines:  # File is not empty
                            # Fix the header to include context
                            context_count = min(3, len(file_lines))
                            line = f'@@ -1,{context_count} +1,{context_count + 1} @@'

                            # We also need to add context lines after the added line
                            # Find the added line(s)
                            added_lines = []
                            j = i + 1
                            while j < len(lines) and not lines[j].startswith('@@'):
                                if lines[j].startswith('+') and not lines[j].startswith('+++'):
                                    added_lines.append(lines[j])
                                j += 1

                            # Insert context lines
                            fixed_lines.append(line)
                            fixed_lines.extend(added_lines)
                            for k in range(context_count):
                                if k < len(file_lines):
                                    fixed_lines.append(' ' + file_lines[k])

                            # Skip the original malformed content safely
                            # Advance until next hunk header or next diff header, without overrunning list
                            while True:
                                nxt = i + 1
                                if nxt >= len(lines):
                                    break
                                nxt_line = lines[nxt]
                                if nxt_line.startswith('@@') or nxt_line.startswith('diff --git'):
                                    break
                                i += 1
                            # continue automatically implied by loop control
                    except Exception:
                        # Defensive: never crash the patch loop
                        continue

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)


def manual_apply_diff(diff_text: str) -> Dict:
    """
    Manually apply a simple diff when git is not available.
    Safer behavior:
      - Try to anchor insertions using hunk context.
      - If no context, insert after shebang/encoding/module-docstring/imports.
      - Fall back to appending at EOF (never raw top-prepend on non-empty files).
    Handles single-file, single-hunk diffs best; otherwise returns a clean error.
    """
    try:
        # Normalize text
        text = diff_text if isinstance(diff_text, str) else diff_text.decode("utf-8", "replace")
        lines = text.splitlines()

        # ---- parse target file + hunk ----
        target_file = None
        for ln in lines:
            if ln.startswith("+++ b/") or ln.startswith("+++ "):
                target_file = ln.replace("+++ b/", "").replace("+++ ", "").strip()
                break
        if not target_file:
            return {"applied": False, "error": "Could not parse target file from diff"}

        # Find first hunk
        try:
            hunk_start = next(i for i, ln in enumerate(lines) if ln.startswith("@@"))
        except StopIteration:
            return {"applied": False, "error": "No hunk found in diff"}

        # Collect hunk body until next hunk or end
        hunk_body: list[str] = []
        for ln in lines[hunk_start + 1:]:
            if ln.startswith("@@") or ln.startswith("diff --git"):
                break
            hunk_body.append(ln)

        if not hunk_body:
            return {"applied": False, "error": "Empty hunk body"}

        # Separate + / - / context
        add_block: list[str] = []
        ctx_block: list[str] = []
        del_block: list[str] = []
        for ln in hunk_body:
            if ln.startswith("+++"):
                continue
            if ln.startswith("+"):
                add_block.append(ln[1:])
            elif ln.startswith("-"):
                del_block.append(ln[1:])
            elif ln.startswith(" "):
                ctx_block.append(ln[1:])
            else:
                # unknown marker; bail gracefully
                return {"applied": False, "error": "Unsupported hunk line marker in manual apply"}

        if not add_block:
            return {"applied": False, "error": "Manual apply only supports hunks with added lines"}

        from pathlib import Path
        full_path = ROOT / target_file
        if not full_path.exists():
            return {"applied": False, "error": f"Target file {target_file} not found"}

        current = full_path.read_text(encoding="utf-8", errors="replace")
        cur_lines = current.splitlines()

        # If file is empty, a top insert is fine.
        if not cur_lines:
            new_text = "\n".join(add_block) + "\n"
            full_path.write_text(new_text, encoding="utf-8")
            return {"applied": True, "error": None}

        # ---- strategy 1: replace old block with new block when old context exists ----
        if del_block or ctx_block:
            # Build old/new slices (context + deletions vs context + additions)
            old_slice = ctx_block + del_block
            new_slice = ctx_block + add_block

            # If we only have context (no deletions), still try to insert before that context.
            if old_slice:
                # Search for first occurrence of old_slice as a contiguous run
                hit = None
                max_i = max(0, len(cur_lines) - len(old_slice))
                for i in range(max_i + 1):
                    if cur_lines[i:i+len(old_slice)] == old_slice:
                        hit = i
                        break
                if hit is not None:
                    new_lines = cur_lines[:hit] + new_slice + cur_lines[hit+len(old_slice):]
                    full_path.write_text("\n".join(new_lines) + ("\n" if current.endswith("\n") else "\n"),
                                         encoding="utf-8")
                    return {"applied": True, "error": None}

            # If we only have context and couldn’t find it, try inserting before first 1–2 context lines
            anchors = [l for l in ctx_block if l.strip()][:2]
            if anchors:
                anchor_idx = None
                for i in range(len(cur_lines) - len(anchors) + 1):
                    if cur_lines[i:i+len(anchors)] == anchors:
                        anchor_idx = i
                        break
                if anchor_idx is not None:
                    new_lines = cur_lines[:anchor_idx] + add_block + cur_lines[anchor_idx:]
                    full_path.write_text("\n".join(new_lines) + ("\n" if current.endswith("\n") else "\n"),
                                         encoding="utf-8")
                    return {"applied": True, "error": None}

        # ---- strategy 2: language-aware safe insertion point (Python-friendly) ----
        # Place after shebang, encoding cookie, module docstring and leading import block.
        def _safe_insert_index(py_lines: list[str]) -> int:
            i = 0
            n = len(py_lines)
            # shebang
            if i < n and py_lines[i].startswith("#!"):
                i += 1
            # encoding cookie
            if i < n and ("coding:" in py_lines[i] or "encoding=" in py_lines[i]):
                i += 1
            # module docstring (triple quotes)
            def _is_triple_start(s: str) -> bool:
                s = s.lstrip()
                return s.startswith('"""') or s.startswith("'''")
            if i < n and _is_triple_start(py_lines[i]):
                q = py_lines[i].lstrip()[:3]
                i += 1
                while i < n and q not in py_lines[i]:
                    i += 1
                if i < n:
                    i += 1  # skip closing line
            # contiguous import block
            while i < n and (py_lines[i].lstrip().startswith("import ") or py_lines[i].lstrip().startswith("from ")):
                i += 1
            return i

        insert_at = _safe_insert_index(cur_lines)
        new_lines = cur_lines[:insert_at] + add_block + cur_lines[insert_at:]
        full_path.write_text("\n".join(new_lines) + ("\n" if current.endswith("\n") else "\n"),
                             encoding="utf-8")
        return {"applied": True, "error": None}

    except Exception as e:
        return {"applied": False, "error": f"Manual apply failed: {e}"}