from __future__ import annotations

import contextlib
import json
import re
import time
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime, timedelta

# Enhanced storage paths
LOG_DIR = Path(".antagent")
LOG_DIR.mkdir(exist_ok=True)
HISTORY = LOG_DIR / "self_improve_history.jsonl"
LESSONS = LOG_DIR / "lessons.json"
PATTERNS = LOG_DIR / "pattern_recognition.json"
SUCCESS_DB = LOG_DIR / "successful_patterns.json"
FAILURE_DB = LOG_DIR / "failure_patterns.json"
CONTEXT_MEMORY = LOG_DIR / "context_memory.json"
_QUEUE_PATH = LOG_DIR / "si_queue.jsonl"

# Default lessons structure
DEFAULT = {
    "counters": {
        "top_insert_detected": 0,
        "empty_diff": 0,
        "apply_success_but_no_change": 0,
        "wrong_scope_edit": 0,
        "bad_hunk_context": 0,
    },
    "anchor_phrases": [],   # stable snippets to anchor on next runs
    "last_10_goals": [],
}

def _extract_unified_diff_from_text(text: str) -> str | None:
    """
    Pull the first UNIX unified diff out of a mixed response.
    Accepts outputs that include analysis or markdown fences.
    """
    if not text:
        return None
    import re
    # strip code fences
    text = re.sub(r"^```(?:diff|patch)?\s*|\s*```$", "", text.strip(), flags=re.M)
    m = re.search(r"(?ms)^diff --git\s+.*$", text)
    if not m:
        return None
    # take from first 'diff --git' to end or next non-diff content if you prefer
    return text[m.start():].strip()

def _repo_root() -> Path:
    # try git first, fall back to parents
    try:
        import subprocess, sys
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=2
        )
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip())
    except Exception:
        pass
    # fallback: walk up until we see a .git folder or run out
    p = Path(__file__).resolve()
    for _ in range(10):
        if (p / ".git").exists():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[1]

def _read_text_safe(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

def _all_candidate_paths(max_bytes: int = 250_000) -> list[str]:
    """
    Repo-wide scan (no hardcoding, no allowlist). Limits to *.py and a size cap.
    Set ANT_READ_CONTEXT_ALL=0 to disable and handle elsewhere if you want.
    """
    if not bool(int(os.getenv("ANT_READ_CONTEXT_ALL", "1"))):
        return []
    root = _repo_root()
    out: list[str] = []
    for p in root.rglob("*.py"):
        try:
            if p.is_file() and p.stat().st_size <= max_bytes:
                out.append(p.relative_to(root).as_posix())
        except Exception:
            continue
    out.sort()
    return out

def _extract_literals_from_objective(goal: str) -> list[str]:
    """Pull textual cues (not filenames) from the objective."""
    if not goal:
        return []
    lits: list[str] = []
    lits += re.findall(r"`([^`]{2,200})`", goal)                    # backticks
    lits += re.findall(r"[\"']([^\"']{2,200})[\"']", goal)          # quotes
    m = re.search(r"(#\s*[^,;]+?:\s*[^\s,;]+)", goal)               # comment-like “# X: Y”
    if m: lits.append(m.group(1))
    seen, out = set(), []
    for s in (x.strip() for x in lits if x.strip()):
        if s not in seen:
            seen.add(s); out.append(s)
    return out[:8]

def _files_containing_any_literals(literals: list[str], max_files: int = 16, max_bytes: int = 250_000) -> list[str]:
    """Return repo-relative files that actually contain any literal."""
    if not literals:
        return []
    root = _repo_root()
    hits: list[str] = []
    for rel in _all_candidate_paths(max_bytes=max_bytes):
        p = root / rel
        txt = _read_text_safe(p)
        if txt and any(lit in txt for lit in literals):
            hits.append(rel)
            if len(hits) >= max_files:
                break
    return hits

def _diff_targets(udiff_text: str) -> list[str]:
    if not udiff_text:
        return []
    pats = re.findall(r"^diff --git\s+(?:a/)?([^\s]+)\s+(?:b/)?\1", udiff_text, flags=re.M)
    seen, out = set(), []
    for p in pats:
        q = p.replace("\\", "/")
        if q not in seen:
            seen.add(q); out.append(q)
    return out

def _diff_touches_any_literal_in_real_file(diff_text: str, literals: list[str]) -> bool:
    """
    True iff the diff removes/changes a line containing any literal AND
    that literal exists in the current working copy of the file.
    """
    if not diff_text or not literals:
        return False
    root = _repo_root()
    for rel in _diff_targets(diff_text):
        file_txt = _read_text_safe(root / rel)
        if not file_txt:
            continue
        # pull this file's chunk
        m = re.search(
            rf"^diff --git\s+(?:a/)?{re.escape(rel)}\s+(?:b/)?{re.escape(rel)}\s*\n(?:(?!^diff --git).)*",
            diff_text, flags=re.M | re.S
        )
        if not m:
            continue
        chunk = m.group(0)
        for lit in literals:
            if lit and lit in file_txt and re.search(rf"^-\s*.*{re.escape(lit)}.*$", chunk, flags=re.M):
                return True
    return False

@dataclass
class LearningContext:
    """Enhanced context for learning from attempts with detailed debugging information"""
    # Basic info (no defaults)
    timestamp: float
    goal: str
    file_path: str
    success: bool
    diff_size: int
    context_lines_used: int
    anchors_used: List[str]
    
    # All fields with defaults
    diff_content: str = ""  # Store the actual diff for analysis
    llm_explanation: str = ""  # The explanation the LLM provided
    engine_used: str = ""  # "llama", "openai", "deepseek", etc.
    llm_confidence: float = 0.0
    retry_count: int = 0
    error_type: Optional[str] = None
    error_detail: Optional[str] = None
    debug_errors: List[str] = field(default_factory=list)  # All debug messages
    validation_errors: List[str] = field(default_factory=list)  # Specific validation failures
    generation_time_ms: float = 0.0
    validation_time_ms: float = 0.0
    apply_time_ms: float = 0.0
    total_time_ms: float = 0.0
    patterns_found: List[str] = field(default_factory=list)
    pattern_type: str = ""
    file_size_bytes: int = 0
    file_line_count: int = 0
    target_line_number: int = 0
    learning_strategy: str = ""  # "conservative", "standard", "confident"
    predicted_difficulty: float = 0.0
    similar_successes_used: int = 0


@dataclass
class PatternMemory:
    """Remembers successful patterns for specific types of changes"""
    # Fields without defaults
    pattern_type: str  # e.g., "comment_replacement", "function_rename", "import_cleanup"
    success_rate: float
    total_attempts: int
    successful_attempts: int
    example_goals: List[str]
    common_anchors: List[str]
    optimal_context_lines: int
    file_patterns: Dict[str, int]  # Which files this pattern works in
    last_updated: float


class EnhancedLearningSystem:
    """
    Advanced learning system that helps the LLM improve over time by:
    1. Learning from successes and failures
    2. Pattern recognition for similar tasks
    3. Context optimization
    4. Predictive difficulty assessment
    5. Strategy adaptation
    """

    def __init__(self):
        self.ensure_dirs()
        self.lessons = self.load_lessons()
        self.patterns = self.load_patterns()
        self.success_db = self.load_success_db()
        self.failure_db = self.load_failure_db()
        self.context_memory = self.load_context_memory()

    def ensure_dirs(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def load_lessons(self) -> Dict:
        if LESSONS.exists():
            try:
                return json.loads(LESSONS.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "total_attempts": 0,
            "successful_changes": 0,
            "failure_reasons": defaultdict(int),
            "anchor_effectiveness": defaultdict(float),
            "file_difficulty": defaultdict(float),
            "pattern_success_rate": defaultdict(float),
            "optimal_settings": {
                "context_lines": 3,
                "max_diff_size": 500,
                "preferred_engine": "llama"
            }
        }

    def load_patterns(self) -> Dict[str, PatternMemory]:
        if PATTERNS.exists():
            try:
                data = json.loads(PATTERNS.read_text(encoding="utf-8"))
                return {k: PatternMemory(**v) for k, v in data.items()}
            except Exception:
                pass
        return {}

    def load_success_db(self) -> List[LearningContext]:
        if SUCCESS_DB.exists():
            try:
                data = json.loads(SUCCESS_DB.read_text(encoding="utf-8"))
                return [LearningContext(**item) for item in data]
            except Exception:
                pass
        return []

    def load_failure_db(self) -> List[LearningContext]:
        if FAILURE_DB.exists():
            try:
                data = json.loads(FAILURE_DB.read_text(encoding="utf-8"))
                return [LearningContext(**item) for item in data]
            except Exception:
                pass
        return []

    def load_context_memory(self) -> Dict:
        if CONTEXT_MEMORY.exists():
            try:
                return json.loads(CONTEXT_MEMORY.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "file_contexts": {},  # Remember important context per file
            "function_signatures": {},  # Remember function patterns
            "import_structures": {},  # Remember import patterns
            "comment_styles": {}  # Remember comment patterns
        }

    def save_all(self):
        """Persist all learning data"""
        LESSONS.write_text(json.dumps(self.lessons, indent=2), encoding="utf-8")

        patterns_data = {k: asdict(v) for k, v in self.patterns.items()}
        PATTERNS.write_text(json.dumps(patterns_data, indent=2), encoding="utf-8")

        success_data = [asdict(ctx) for ctx in self.success_db[-1000:]]  # Keep last 1000
        SUCCESS_DB.write_text(json.dumps(success_data, indent=2), encoding="utf-8")

        failure_data = [asdict(ctx) for ctx in self.failure_db[-1000:]]  # Keep last 1000
        FAILURE_DB.write_text(json.dumps(failure_data, indent=2), encoding="utf-8")

        CONTEXT_MEMORY.write_text(json.dumps(self.context_memory, indent=2), encoding="utf-8")

    def classify_goal(self, goal: str) -> str:
        """Classify the type of change being requested"""
        goal_lower = goal.lower()

        if "comment" in goal_lower or "#" in goal:
            return "comment_modification"
        elif "import" in goal_lower:
            return "import_management"
        elif "rename" in goal_lower or "replace" in goal_lower:
            return "rename_operation"
        elif "add" in goal_lower or "insert" in goal_lower:
            return "code_insertion"
        elif "remove" in goal_lower or "delete" in goal_lower:
            return "code_deletion"
        elif "fix" in goal_lower or "bug" in goal_lower:
            return "bug_fix"
        elif "refactor" in goal_lower:
            return "refactoring"
        elif "docstring" in goal_lower or "documentation" in goal_lower:
            return "documentation"
        else:
            return "general_modification"

    def predict_difficulty(self, goal: str, file_path: str) -> Dict[str, Any]:
        """Predict how difficult this change will be"""
        pattern_type = self.classify_goal(goal)

        # Check historical success rate for this pattern
        pattern_success = 0.5  # default
        if pattern_type in self.patterns:
            pattern = self.patterns[pattern_type]
            pattern_success = pattern.success_rate

        # Check file difficulty
        file_difficulty = self.lessons["file_difficulty"].get(file_path, 0.5)

        # Calculate overall difficulty
        difficulty = 1.0 - (pattern_success * (1.0 - file_difficulty))

        # Recommend strategy based on difficulty
        if difficulty > 0.7:
            strategy = {
                "approach": "conservative",
                "context_lines": 5,
                "use_multiple_anchors": True,
                "verify_twice": True,
                "explanation_detail": "high"
            }
        elif difficulty > 0.4:
            strategy = {
                "approach": "standard",
                "context_lines": 3,
                "use_multiple_anchors": False,
                "verify_twice": False,
                "explanation_detail": "medium"
            }
        else:
            strategy = {
                "approach": "confident",
                "context_lines": 2,
                "use_multiple_anchors": False,
                "verify_twice": False,
                "explanation_detail": "low"
            }

        return {
            "difficulty": difficulty,
            "pattern_type": pattern_type,
            "historical_success": pattern_success,
            "file_complexity": file_difficulty,
            "recommended_strategy": strategy,
            "similar_successes": self.find_similar_successes(goal, limit=3)
        }

    def find_similar_successes(self, goal: str, limit: int = 5) -> List[Dict]:
        """Find similar successful changes to learn from"""
        goal_words = set(goal.lower().split())
        pattern_type = self.classify_goal(goal)

        similar = []
        for ctx in self.success_db:
            if ctx.goal == goal:
                continue

            # Calculate similarity
            ctx_words = set(ctx.goal.lower().split())
            word_overlap = len(goal_words & ctx_words) / max(len(goal_words), 1)

            # Boost if same pattern type
            pattern_match = 1.0 if self.classify_goal(ctx.goal) == pattern_type else 0.5

            similarity = word_overlap * pattern_match

            if similarity > 0.3:
                similar.append({
                    "goal": ctx.goal,
                    "similarity": similarity,
                    "file": ctx.file_path,
                    "anchors": ctx.anchors_used,
                    "context_lines": ctx.context_lines_used
                })

        # Sort by similarity and return top matches
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        return similar[:limit]

    def learn_from_attempt(self, context: LearningContext):
        """Learn from a single attempt with enhanced debugging information"""
        self.lessons["total_attempts"] += 1

        # Store detailed attempt information
        attempt_data = {
            "timestamp": context.timestamp,
            "goal": context.goal,
            "file_path": context.file_path,
            "success": context.success,
            "engine_used": context.engine_used,
            "llm_explanation": context.llm_explanation,
            "error_type": context.error_type,
            "error_detail": context.error_detail,
            "debug_errors": context.debug_errors,
            "validation_errors": context.validation_errors,
            "performance": {
                "generation_time_ms": context.generation_time_ms,
                "validation_time_ms": context.validation_time_ms,
                "apply_time_ms": context.apply_time_ms,
                "total_time_ms": context.total_time_ms
            },
            "context": {
                "lines_used": context.context_lines_used,
                "anchors_used": context.anchors_used,
                "strategy": context.learning_strategy,
                "predicted_difficulty": context.predicted_difficulty
            }
        }

        if context.success:
            self.lessons["successful_changes"] += 1
            self.success_db.append(context)

            # Update pattern memory with enhanced data
            pattern_type = self.classify_goal(context.goal)
            if pattern_type not in self.patterns:
                self.patterns[pattern_type] = PatternMemory(
                    pattern_type=pattern_type,
                    success_rate=0.0,
                    total_attempts=0,
                    successful_attempts=0,
                    example_goals=[],
                    common_anchors=[],
                    optimal_context_lines=3,
                    file_patterns=defaultdict(int),
                    last_updated=time.time()
                )

            pattern = self.patterns[pattern_type]
            pattern.total_attempts += 1
            pattern.successful_attempts += 1
            pattern.success_rate = pattern.successful_attempts / pattern.total_attempts
            pattern.example_goals.append(context.goal)
            pattern.common_anchors.extend(context.anchors_used)
            pattern.file_patterns[context.file_path] += 1
            pattern.last_updated = time.time()

            # Update anchor effectiveness with engine-specific data
            for anchor in context.anchors_used:
                current = self.lessons["anchor_effectiveness"].get(anchor, 0.0)
                # Boost effectiveness if this anchor worked with the current engine
                boost = 0.1 if context.engine_used == "llama" else 0.05
                self.lessons["anchor_effectiveness"][anchor] = current * 0.9 + boost

            # Learn from successful patterns
            if context.llm_explanation:
                # Extract key phrases from successful explanations
                explanation_words = context.llm_explanation.lower().split()
                key_phrases = [w for w in explanation_words if len(w) > 4 and w.isalpha()]
                self.lessons.setdefault("successful_explanations", []).extend(key_phrases[:5])

        else:
            self.failure_db.append(context)
            self.lessons["failure_reasons"][context.error_type or "unknown"] += 1

            # Enhanced failure learning
            if context.debug_errors:
                self.lessons.setdefault("debug_error_patterns", []).extend(context.debug_errors[:3])
            
            if context.validation_errors:
                self.lessons.setdefault("validation_error_patterns", []).extend(context.validation_errors[:3])

            # Update file difficulty based on specific error types
            current = self.lessons["file_difficulty"].get(context.file_path, 0.5)
            if context.error_type == "validation_error":
                self.lessons["file_difficulty"][context.file_path] = min(1.0, current + 0.15)
            elif context.error_type == "apply_failed":
                self.lessons["file_difficulty"][context.file_path] = min(1.0, current + 0.1)
            else:
                self.lessons["file_difficulty"][context.file_path] = min(1.0, current + 0.05)

            # Learn from failure patterns with engine-specific data
            if context.error_type == "no_context":
                self.lessons["optimal_settings"]["context_lines"] = min(
                    10, self.lessons["optimal_settings"]["context_lines"] + 1
                )
            
            # Track engine-specific failure patterns
            engine_key = f"{context.engine_used}_failures"
            self.lessons.setdefault(engine_key, []).append({
                "error_type": context.error_type,
                "error_detail": context.error_detail,
                "timestamp": context.timestamp
            })

        # Store detailed attempt for analysis
        self.lessons.setdefault("detailed_attempts", []).append(attempt_data)
        
        # Keep only last 100 detailed attempts to prevent memory bloat
        if len(self.lessons["detailed_attempts"]) > 100:
            self.lessons["detailed_attempts"] = self.lessons["detailed_attempts"][-100:]

        self.save_all()

    def get_smart_constraints(self, goal: str, base_constraints: Dict) -> Dict:
        """
        Enhance constraints based on learned patterns and past successes
        """
        prediction = self.predict_difficulty(goal, base_constraints.get("paths", [""])[0])
        strategy = prediction["recommended_strategy"]

        enhanced = base_constraints.copy()

        # Adjust context lines based on difficulty
        enhanced["require_context_lines"] = strategy["context_lines"]

        # Add learned anchors
        enhanced.setdefault("must_anchor_any", [])

        # Add anchors from similar successes
        for similar in prediction["similar_successes"]:
            for anchor in similar["anchors"]:
                if anchor not in enhanced["must_anchor_any"]:
                    enhanced["must_anchor_any"].append(anchor)

        # Add highly effective anchors
        for anchor, effectiveness in self.lessons["anchor_effectiveness"].items():
            if effectiveness > 0.7 and anchor not in enhanced["must_anchor_any"]:
                enhanced["must_anchor_any"].append(anchor)
                if len(enhanced["must_anchor_any"]) >= 10:
                    break

        # Add pattern-specific settings
        pattern_type = prediction["pattern_type"]
        if pattern_type in self.patterns:
            pattern = self.patterns[pattern_type]
            if pattern.optimal_context_lines > enhanced["require_context_lines"]:
                enhanced["require_context_lines"] = pattern.optimal_context_lines

        # Store metadata for learning
        enhanced["_learning_metadata"] = {
            "predicted_difficulty": prediction["difficulty"],
            "pattern_type": pattern_type,
            "strategy": strategy["approach"]
        }

        return enhanced

    def generate_enhanced_learning_report(self) -> str:
        """Generate a comprehensive learning report with debugging insights"""
        total = self.lessons["total_attempts"]
        if total == 0:
            return "No learning data yet."

        success_rate = self.lessons["successful_changes"] / total * 100

        report = f"""
Enhanced Learning Report (Total Attempts: {total})
==================================================

Success Rate: {success_rate:.1f}%
Successful Changes: {self.lessons["successful_changes"]}

Engine Performance:
"""
        
        # Analyze engine-specific performance
        engine_stats = {}
        for attempt in self.lessons.get("detailed_attempts", []):
            engine = attempt.get("engine_used", "unknown")
            if engine not in engine_stats:
                engine_stats[engine] = {"total": 0, "success": 0, "avg_time": 0}
            engine_stats[engine]["total"] += 1
            if attempt.get("success"):
                engine_stats[engine]["success"] += 1
            if attempt.get("performance", {}).get("total_time_ms"):
                engine_stats[engine]["avg_time"] += attempt["performance"]["total_time_ms"]
        
        for engine, stats in engine_stats.items():
            if stats["total"] > 0:
                success_pct = stats["success"] / stats["total"] * 100
                avg_time = stats["avg_time"] / stats["total"] if stats["total"] > 0 else 0
                report += f"  - {engine}: {success_pct:.1f}% success ({stats['success']}/{stats['total']}), avg {avg_time:.0f}ms\n"

        report += "\nTop Failure Reasons:\n"
        for reason, count in sorted(self.lessons["failure_reasons"].items(),
                                    key=lambda x: x[1], reverse=True)[:5]:
            report += f"  - {reason}: {count} times\n"

        report += "\nDebug Error Patterns:\n"
        debug_patterns = self.lessons.get("debug_error_patterns", [])
        if debug_patterns:
            from collections import Counter
            pattern_counts = Counter(debug_patterns)
            for pattern, count in pattern_counts.most_common(3):
                report += f"  - '{pattern[:50]}...': {count} occurrences\n"
        else:
            report += "  - No debug error patterns recorded\n"

        report += "\nValidation Error Patterns:\n"
        validation_patterns = self.lessons.get("validation_error_patterns", [])
        if validation_patterns:
            from collections import Counter
            pattern_counts = Counter(validation_patterns)
            for pattern, count in pattern_counts.most_common(3):
                report += f"  - '{pattern[:50]}...': {count} occurrences\n"
        else:
            report += "  - No validation error patterns recorded\n"

        report += "\nMost Effective Anchors:\n"
        for anchor, effectiveness in sorted(self.lessons["anchor_effectiveness"].items(),
                                            key=lambda x: x[1], reverse=True)[:5]:
            report += f"  - '{anchor[:30]}...': {effectiveness:.2f}\n"

        report += "\nPattern Success Rates:\n"
        for pattern_type, pattern in self.patterns.items():
            report += f"  - {pattern_type}: {pattern.success_rate * 100:.1f}% ({pattern.successful_attempts}/{pattern.total_attempts})\n"

        report += "\nMost Difficult Files:\n"
        for file, difficulty in sorted(self.lessons["file_difficulty"].items(),
                                       key=lambda x: x[1], reverse=True)[:5]:
            report += f"  - {file}: difficulty {difficulty:.2f}\n"

        # Performance insights
        report += "\nPerformance Insights:\n"
        recent_attempts = self.lessons.get("detailed_attempts", [])[-10:]
        if recent_attempts:
            avg_generation = sum(a.get("performance", {}).get("generation_time_ms", 0) for a in recent_attempts) / len(recent_attempts)
            avg_validation = sum(a.get("performance", {}).get("validation_time_ms", 0) for a in recent_attempts) / len(recent_attempts)
            avg_apply = sum(a.get("performance", {}).get("apply_time_ms", 0) for a in recent_attempts) / len(recent_attempts)
            report += f"  - Avg generation time: {avg_generation:.0f}ms\n"
            report += f"  - Avg validation time: {avg_validation:.0f}ms\n"
            report += f"  - Avg apply time: {avg_apply:.0f}ms\n"

        # Learning recommendations
        report += "\nLearning Recommendations:\n"
        if success_rate < 50:
            report += "  - Consider increasing context lines for better anchoring\n"
        if self.lessons.get("validation_error_patterns"):
            report += "  - Focus on improving diff validation accuracy\n"
        if len(self.lessons.get("anchor_effectiveness", {})) < 5:
            report += "  - Build more anchor phrases for better targeting\n"

        return report


# Global learning system instance
_learning_system = None


def get_learning_system() -> EnhancedLearningSystem:
    """Get or create the global learning system"""
    global _learning_system
    if _learning_system is None:
        _learning_system = EnhancedLearningSystem()
    return _learning_system


def demonstrate_enhanced_learning():
    """Demonstrate the enhanced learning capabilities"""
    learning = get_learning_system()
    
    print("=== Enhanced Learning System Demo ===")
    print(f"Total attempts recorded: {learning.lessons['total_attempts']}")
    print(f"Successful changes: {learning.lessons['successful_changes']}")
    
    if learning.lessons['total_attempts'] > 0:
        print("\n=== Enhanced Learning Report ===")
        print(learning.generate_enhanced_learning_report())
        
        print("\n=== Recent Detailed Attempts ===")
        recent = learning.lessons.get("detailed_attempts", [])[-3:]
        for i, attempt in enumerate(recent, 1):
            print(f"\nAttempt {i}:")
            print(f"  Goal: {attempt['goal'][:50]}...")
            print(f"  Engine: {attempt['engine_used']}")
            print(f"  Success: {attempt['success']}")
            if attempt.get('llm_explanation'):
                print(f"  Explanation: {attempt['llm_explanation'][:100]}...")
            if attempt.get('debug_errors'):
                print(f"  Debug errors: {len(attempt['debug_errors'])} messages")
            if attempt.get('performance'):
                perf = attempt['performance']
                print(f"  Performance: {perf.get('total_time_ms', 0):.0f}ms total")
    else:
        print("No learning data available yet. Run some self-improvement attempts to see enhanced learning in action!")
    
    return learning


def enhanced_self_improve_with_learning(goal: str, constraints: Dict, max_attempts: int = 3) -> Dict:
    """
    Self-improvement with advanced learning capabilities
    """
    learning = get_learning_system()

    # Get smart constraints based on learning
    enhanced_constraints = learning.get_smart_constraints(goal, constraints)

    # Predict difficulty and get strategy
    prediction = learning.predict_difficulty(goal, enhanced_constraints.get("paths", [""])[0])

    print(f"\n[LEARNING] Analyzing goal: {goal[:50]}...")
    print(f"[LEARNING] Predicted difficulty: {prediction['difficulty']:.2f}")
    print(f"[LEARNING] Pattern type: {prediction['pattern_type']}")
    print(f"[LEARNING] Strategy: {prediction['recommended_strategy']['approach']}")

    if prediction['similar_successes']:
        print(f"[LEARNING] Found {len(prediction['similar_successes'])} similar successful changes")

    attempts = []

    for attempt_num in range(max_attempts):
        # Adjust strategy based on previous attempts
        if attempt_num > 0:
            # Increase context on retry
            enhanced_constraints["require_context_lines"] = min(
                10, enhanced_constraints["require_context_lines"] + 2
            )
            print(
                f"[LEARNING] Retry {attempt_num}: Increased context to {enhanced_constraints['require_context_lines']}")

        # Create learning context
        learning_ctx = LearningContext(
            timestamp=time.time(),
            goal=goal,
            file_path=enhanced_constraints.get("paths", [""])[0],
            success=False,
            diff_size=0,
            context_lines_used=enhanced_constraints["require_context_lines"],
            anchors_used=enhanced_constraints.get("must_anchor_any", []),
            retry_count=attempt_num
        )

        try:
            # Call the actual improvement function
            from .manager import propose_patch_with_explanation, apply_patch

            summary, diff, explanation = propose_patch_with_explanation(goal, enhanced_constraints)

            if not diff:
                learning_ctx.error_type = "no_diff_generated"
                learning_ctx.error_detail = "Model produced no diff"
                learning.learn_from_attempt(learning_ctx)

                attempts.append({
                    "attempt": attempt_num + 1,
                    "status": "no_diff",
                    "explanation": explanation
                })
                continue

            learning_ctx.diff_size = len(diff)

            # Try to apply
            result = apply_patch(diff)

            if result["applied"]:
                learning_ctx.success = True
                learning.learn_from_attempt(learning_ctx)

                print(f"[LEARNING] Success! Recording successful pattern.")

                return {
                    "success": True,
                    "attempts": attempts + [{
                        "attempt": attempt_num + 1,
                        "status": "applied",
                        "explanation": explanation,
                        "diff": diff
                    }],
                    "final_diff": diff,
                    "explanation": explanation,
                    "learning_report": learning.generate_learning_report()
                }
            else:
                learning_ctx.error_type = "apply_failed"
                learning_ctx.error_detail = result.get("error", "unknown")
                learning.learn_from_attempt(learning_ctx)

                attempts.append({
                    "attempt": attempt_num + 1,
                    "status": "apply_failed",
                    "error": result.get("error"),
                    "explanation": explanation
                })

        except Exception as e:
            learning_ctx.error_type = "exception"
            learning_ctx.error_detail = str(e)
            learning.learn_from_attempt(learning_ctx)

            attempts.append({
                "attempt": attempt_num + 1,
                "status": "exception",
                "error": str(e)
            })

    # All attempts failed - learn from this
    print(f"[LEARNING] All attempts failed. Recording failure patterns.")

    return {
        "success": False,
        "attempts": attempts,
        "final_diff": None,
        "explanation": "Failed after all attempts",
        "learning_report": learning.generate_learning_report()
    }


def _diff_targets(udiff_text: str) -> list[str]:
    """Extract file paths from unified diff headers."""
    if not udiff_text:
        return []

    seen, out = set(), []

    # More flexible pattern to handle both "diff --git a/path b/path" and "diff --git path path"
    # First try the standard format with a/ and b/ prefixes
    pats = re.findall(r"^diff --git\s+a/([^\s]+)\s+b/([^\s]+)", udiff_text, flags=re.M)

    if pats:
        # Take the b/ path (second group) from each match
        for a_path, b_path in pats:
            if a_path == b_path and b_path not in seen:
                seen.add(b_path)
                out.append(b_path.replace("\\", "/"))
    else:
        # Try without a/ b/ prefixes
        pats = re.findall(r"^diff --git\s+([^\s]+)\s+([^\s]+)", udiff_text, flags=re.M)
        for a_path, b_path in pats:
            # Usually both paths are the same for modifications
            if b_path not in seen:
                seen.add(b_path)
                out.append(b_path.replace("\\", "/"))

    print(f"[DEBUG] Extracted paths from diff: {out}")
    return out

def _project_root() -> Path:
    # This file is at .../AntAgent/autodev/manager_learning.py
    # So parents[1] gives us .../AntAgent
    return Path(__file__).resolve().parents[1]

def _allowlist_path() -> Path:
    return _project_root() / "autodev" / "allowlist.txt"

def _allowed_paths() -> list[str]:
    p = _allowlist_path()
    if not p.exists():
        print(f"[DEBUG] Allowlist not found at {p}")
        return []
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    allowed = [ln for ln in lines if ln and not ln.startswith("#")]
    print(f"[DEBUG] Allowed paths: {allowed}")
    return allowed

def _normalize_targets_to_allowlist(targets: list[str], allowed: list[str]) -> list[str]:
    if not targets:
        return []
    allowed_norm = [p.replace("\\", "/") for p in (allowed or [])]
    by_base: dict[str, list[str]] = {}
    for p in allowed_norm:
        by_base.setdefault(p.rsplit("/", 1)[-1], []).append(p)

    resolved: list[str] = []
    for t in targets:
        t_norm = t.replace("\\", "/")
        if t_norm in allowed_norm:
            resolved.append(t_norm)
            continue
        base = t_norm.rsplit("/", 1)[-1]
        cand = by_base.get(base, [])
        if len(cand) == 1:
            resolved.append(cand[0])
    return resolved
@dataclass
class ObjectiveItem:
    # Fields without defaults
    ts: float
    objective: str
    # Fields with defaults
    source: str = "user"   # user | heuristic | retry
    priority: int = 0      # higher first
    rounds: int = 1

def enqueue_objective(obj: ObjectiveItem) -> None:
    with _QUEUE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(obj), ensure_ascii=False) + "\n")

def dequeue_objective() -> Optional[ObjectiveItem]:
    if not _QUEUE_PATH.exists():
        return None
    lines = _QUEUE_PATH.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None
    # pick highest priority, then oldest
    items = [json.loads(x) for x in lines]
    items.sort(key=lambda d: (-int(d.get("priority", 0)), float(d.get("ts", 0.0))))
    picked = items.pop(0)
    _QUEUE_PATH.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items) + ("\n" if items else ""), encoding="utf-8")
    return ObjectiveItem(**picked)

def queue_len() -> int:
    if not _QUEUE_PATH.exists():
        return 0
    return sum(1 for _ in _QUEUE_PATH.open("r", encoding="utf-8"))

def load_lessons() -> dict:
    if _LESSONS_PATH.exists():
        try:
            return json.loads(_LESSONS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_lessons(d: dict) -> None:
    _LESSONS_PATH.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

def update_lessons(key: str, payload: dict) -> None:
    d = load_lessons()
    bucket = d.setdefault(key, [])
    bucket.append({"ts": time.time(), **(payload or {})})
    # optional: maintain derived anchor set
    if key == "anchor_phrase":
        anchors = set(d.get("anchor_phrases", []))
        val = payload.get("value")
        if isinstance(val, str):
            anchors.add(val)
        d["anchor_phrases"] = sorted(anchors)
    save_lessons(d)

def get_lessons() -> dict:
    return load_lessons()

def _iter_repo_py_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        # cheap allowlist: only under AntAgent/
        try:
            if "AntAgent" in p.as_posix():
                yield p
        except Exception:
            continue

def generate_objectives(max_items: int = 5) -> list[ObjectiveItem]:
    """
    Heuristically propose small, safe improvements:
      - TODO / FIXME lines
      - Docstring smoke markers to bump versions
      - Style nits (duplicate imports)
    """
    root = Path(".").resolve()
    out: list[ObjectiveItem] = []
    for fp in _iter_repo_py_files(root):
        try:
            txt = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # 1) Replace the Random animal line (kept as a canonical smoke objective)
        if fp.as_posix().endswith("AntAgent/autodev/manager.py") and re.search(r"^#\s*Random\s+animal:\s*\w+", txt, flags=re.M):
            out.append(ObjectiveItem(ts=time.time(), objective="In manager.py, replace # Random animal: <old> with another random animal comment.", source="heuristic", priority=5, rounds=1))

        # 2) TODO/FIXME -> turn into precise micro-edits if trivial
        for m in re.finditer(r"^(?P<indent>\s*)#\s*(TODO|FIXME)\s*:\s*(?P<body>.+)$", txt, flags=re.M):
            body = m.group("body").strip()
            if 8 <= len(body) <= 120:
                out.append(ObjectiveItem(
                    ts=time.time(),
                    objective=f"Address a trivial TODO in {fp.as_posix()}: {body}",
                    source="heuristic",
                    priority=1,
                    rounds=1
                ))

        # 3) Duplicate import cleanup on small modules
        if re.search(r"(?m)^(from\s+\S+\s+import\s+\S+|import\s+\S+).*\n.*\1", txt):
            out.append(ObjectiveItem(
                ts=time.time(),
                objective=f"Deduplicate adjacent imports in {fp.as_posix()} without changing behavior.",
                source="heuristic",
                priority=1,
                rounds=1
            ))

        if len(out) >= max_items:
            break
    return out

def self_improve_once(prompt: Optional[str] = None, *, rounds: int = 1) -> Dict:
    """
    If `prompt` provided, enqueue and consume it; otherwise dequeue or generate an objective.
    Executes auto_self_improve() for one objective and returns its result object.
    """
    if prompt and str(prompt).strip():
        enqueue_objective(ObjectiveItem(ts=time.time(), objective=str(prompt).strip(), source="user", priority=10, rounds=rounds))

    item = dequeue_objective()
    if item is None:
        # generate fresh objectives and enqueue one
        for obj in generate_objectives(max_items=3):
            enqueue_objective(obj)
        item = dequeue_objective()

    if item is None:
        # nothing to do
        return {
            "branch": None,
            "base": _git_current_branch(),
            "objective": "(none)",
            "rounds": 0,
            "results": []
        }

    return auto_self_improve(item.objective, rounds=item.rounds)

def _ensure_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def _load_lessons() -> Dict[str, Any]:
    _ensure_dir()
    if LESSONS.exists():
        try:
            return json.loads(LESSONS.read_text(encoding="utf-8"))
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULT))

def _save_lessons(d: Dict[str, Any]) -> None:
    _ensure_dir()
    LESSONS.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

def record_history(entry: Dict[str, Any]) -> None:
    _ensure_dir()
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def update_lessons(event: str, payload: Dict[str, Any]) -> None:
    """
    Update counters and anchor phrases based on events.
    events: top_insert, empty_diff, no_effect, wrong_scope, bad_hunk
    payload may include: goal, must_anchor_any, offending_lines, context_line, file_path
    """
    d = _load_lessons()
    c = d["counters"]

    if event == "top_insert":
        c["top_insert_detected"] += 1
        # top insert => add structural anchors that usually exist near top:
        _merge_anchors(d, ['"""', "import ", "from "])
    elif event == "empty_diff":
        c["empty_diff"] += 1
    elif event == "no_effect":
        c["apply_success_but_no_change"] += 1
    elif event == "wrong_scope":
        c["wrong_scope_edit"] += 1
        # add function name or quoted phrases from payload as stabilizers
        _merge_anchors(d, _extract_likely_anchors(payload.get("goal", "")))
    elif event == "bad_hunk":
        c["bad_hunk_context"] += 1

    # keep last 10 goals
    g = payload.get("goal", "")
    if g:
        d["last_10_goals"] = (d.get("last_10_goals") or [])[-9:] + [g]

    _save_lessons(d)

def get_lessons() -> Dict[str, Any]:
    return _load_lessons()

QUOTED = re.compile(r"[\"']([^\"']{3,200})[\"']")
DEFN   = re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")

def _extract_likely_anchors(goal: str) -> List[str]:
    anchors: List[str] = []
    anchors += [m.group(1) for m in QUOTED.finditer(goal)]
    anchors += [m.group(1) for m in DEFN.finditer(goal)]
    uniq = []
    seen = set()
    for a in anchors:
        a = a.strip()
        if not a or len(a) > 300:
            continue
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq[:8]

def _merge_anchors(lessons: Dict[str, Any], new_items: List[str]) -> None:
    merged = list(dict.fromkeys((lessons.get("anchor_phrases") or []) + (new_items or [])))
    lessons["anchor_phrases"] = merged[-24:]  # cap
import subprocess, time, re, os, json, tempfile
from pathlib import Path
from typing import Dict, Tuple, List

# Adjust if your repo layout is different
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALLOWED_DIRS = ("AntAgent/autodev", "AntAgent/app.py")  # limit self-edits
_MAX_CHANGED_BYTES = 40_000  # hard cap for a single diff
_MIN_CONTEXT_LINES = 2       # require robust hunks

_DIFF_HEADER_RE = re.compile(r"(?m)^diff --git\s+a/([^\s\n]+)\s+b/([^\s\n]+)$")
_HUNK_RE        = re.compile(r"(?m)^@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@")

def _run(cmd: List[str], cwd: Path | None = None, timeout: int = 120) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=str(cwd or _REPO_ROOT),
                       text=True, capture_output=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr

def _validate_unified_diff(diff_text: str) -> None:
    if not isinstance(diff_text, str) or not diff_text.strip():
        raise ValueError("Empty diff")
    if not _DIFF_HEADER_RE.search(diff_text):
        raise ValueError("Invalid diff: missing 'diff --git' header")
    if not _HUNK_RE.search(diff_text):
        raise ValueError("Invalid diff: missing '@@' hunks")
    if len(diff_text.encode("utf-8")) > _MAX_CHANGED_BYTES:
        raise ValueError(f"Diff too large (> {_MAX_CHANGED_BYTES} bytes)")
    # enforce min context per hunk
    for h in re.findall(r"(?ms)^@@[^\n]*\n(.*?)(?=^@@|\Z)", diff_text):
        ctx = sum(1 for line in h.splitlines() if line.startswith(" "))
        if ctx < _MIN_CONTEXT_LINES:
            raise ValueError("Hunk lacks sufficient unchanged context lines")

def _all_targets_allowed(diff_text: str) -> list[str]:
    # Reuse the same _diff_targets the code uses elsewhere (import or inline)
    targets = _diff_targets(diff_text)  # if not available here, copy that helper too
    allowed = _allowed_paths() or []    # if not available here, read allowlist.txt like app.py does
    targets = _normalize_targets_to_allowlist(targets, allowed)
    if not targets:
        raise ValueError("No valid targets in diff")
    allowed_norm = {p.replace("\\", "/") for p in allowed}
    if not set(targets).issubset(allowed_norm):
        bad = [t for t in targets if t not in allowed_norm]
        raise ValueError(f"Path not allowed for self-edit: {', '.join(bad)}")
    return targets

def _git_current_branch() -> str:
    rc, out, _ = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if rc != 0:
        raise RuntimeError("Not a git repository or HEAD unresolved")
    return out.strip()

def _git_checkout_new_branch(name: str) -> None:
    rc, _, err = _run(["git", "checkout", "-b", name])
    if rc != 0 and "already exists" not in err.lower():
        raise RuntimeError(f"git checkout -b failed: {err}")

def _git_restore_paths(paths: List[str]) -> None:
    """
    Revert only specific paths back to HEAD, unstaging them as well.
    No repo-wide clean.
    """
    if not paths:
        return
    _run(["git", "restore", "--staged", "--worktree", "--source=HEAD", "--"] + paths)



def _git_commit_paths(paths: List[str], message: str) -> None:
    """
    Stage and commit only the given paths.
    """
    if not paths:
        return
    _run(["git", "add", "--"] + paths)
    # Ensure identity exists (no-op if already set)
    _run(["git", "config", "user.email"], timeout=5)
    _run(["git", "config", "user.name"], timeout=5)
    _run(["git", "commit", "-m", message])

def _git_is_clean() -> bool:
    rc, out, _ = _run(["git", "status", "--porcelain"])
    return rc == 0 and out.strip() == ""

def _require_clean_tree() -> None:
    if not _git_is_clean():
        raise RuntimeError("Working tree is dirty. Commit/stash your unrelated changes before self-improve.")

class _StashGuard:
    def __init__(self):
        self._did = False
    def __enter__(self):
        rc, out, err = _run(["git", "status", "--porcelain"])
        if rc == 0 and out.strip():
            # working tree is dirty -> stash everything including untracked
            rc2, out2, err2 = _run(["git", "stash", "-u", "-k"])
            if rc2 == 0:
                self._did = True
        return self
    def __exit__(self, exc_type, exc, tb):
        if self._did:
            _run(["git", "stash", "pop"])

def _smoke_checks() -> Tuple[bool, str]:
    """
    Strict, quick checks with no fallbacks:
    - Byte-compile the package
    - Import the API module
    - (Optional) run pytest if present
    """
    ok_steps: List[str] = []
    fail_steps: List[str] = []

    rc, out, err = _run(["python", "-m", "compileall", "-q", str(_REPO_ROOT)])
    (ok_steps if rc == 0 else fail_steps).append(f"compileall rc={rc} {err or out}".strip())

    rc, out, err = _run(["python", "-c", "import AntAgent.app; print('import_ok')"])
    (ok_steps if rc == 0 else fail_steps).append(f"import AntAgent.app rc={rc} {err or out}".strip())

    if (_REPO_ROOT / "pytest.ini").exists() or (_REPO_ROOT / "tests").exists():
        rc, out, err = _run(["python", "-m", "pytest", "-q"])
        (ok_steps if rc == 0 else fail_steps).append(f"pytest rc={rc} {err or out}".strip())

    msg = " | ".join(ok_steps + fail_steps)
    return (len(fail_steps) == 0), msg
def _goal_adjacent_anchors(file_text: str, goal: str, radius: int = 2) -> list[str]:
    """
    From the real file, extract up to `radius` lines around any literal snippet suggested by the goal.
    This forces the model to use actual nearby lines as context, so strict git apply can place hunks.
    """
    if not goal or not file_text:
        return []
    anchors: list[str] = []
    lines = file_text.splitlines()

    # Pull quoted substrings and common comment patterns from the goal
    lits = [m.group(1) for m in re.finditer(r"[\"']([^\"']{3,200})[\"']", goal)]
    m = re.search(r"(#\s*Random animal:\s*\w+)", goal)
    if m:
        lits.append(m.group(1))

    for lit in lits:
        lit = lit.strip()
        if not lit:
            continue
        for i, ln in enumerate(lines):
            if lit in ln:
                s = max(0, i - radius)
                e = min(len(lines), i + radius + 1)
                for ctx in lines[s:e]:
                    c = ctx.rstrip("\n")
                    if c and c not in anchors:
                        anchors.append(c)
                break
    return anchors[:10]


def auto_self_improve(objective: str, *, rounds: int = 1) -> Dict:
    """
    Attempt up to `rounds` self-edits with advanced learning capabilities.
    Auto-stashes unrelated changes before running, then restores them afterwards.
    Only commits/reverts files touched by the diff.
    Learns and improves from each attempt using pattern recognition and historical data.
    """
    from .manager import propose_patch_with_explanation, apply_patch
    from .manager_learning import get_lessons, update_lessons
    import os

    literals = _extract_literals_from_objective(objective)

    # build dynamic context: only files that contain those cues; if none found, fall back to scanning
    context_paths = _files_containing_any_literals(literals) or _all_candidate_paths()

    # Initialize learning system
    learning = get_learning_system()

    git_isolation = bool(int(os.getenv("ANT_SI_GIT_ISOLATION", "0")))
    revert_on_fail = bool(int(os.getenv("ANT_SI_REVERT_ON_FAIL", "0")))

    base_branch = _git_current_branch()
    work_branch = f"si/{int(time.time())}" if git_isolation else base_branch

    results: List[Dict] = []
    DIFF_RETURN_LIMIT = 20000  # chars to return to API; raw diff still fully applied

    # Analyze the objective before starting
    initial_analysis = learning.predict_difficulty(objective, "")
    print(f"\n[LEARNING] Starting self-improvement: {objective[:60]}...")
    print(f"[LEARNING] Pattern type: {initial_analysis['pattern_type']}")
    print(f"[LEARNING] Strategy: {initial_analysis['recommended_strategy']['approach']}")

    # Check for similar successes
    similar = learning.find_similar_successes(objective, limit=3)
    if similar:
        print(f"[LEARNING] Found {len(similar)} similar successful changes to learn from")

    guard = _StashGuard() if git_isolation else contextlib.ExitStack()

    with guard:  # auto-stash dirty tree, pop on exit
        if git_isolation:
            _git_checkout_new_branch(work_branch)

        for i in range(1, rounds + 1):
            print(f"\n[ATTEMPT {i}/{rounds}] Starting...")

            # --- Intelligent target path resolution ---
            target_paths = []

            # First, check if we can learn from similar successes
            if similar and i == 1:  # Use similar success info on first attempt
                for s in similar:
                    if s['file'] and s['file'] in _allowed_paths():
                        print(f"[LEARNING] Using file from similar success: {s['file']}")
                        target_paths = [s['file']]
                        break

            # If not learned, use smart file discovery
            if not target_paths:
                # Extract potential file references from objective
                import re
                file_mentions = re.findall(r'(\w+\.py)', objective.lower())

                if file_mentions:
                    # Look for mentioned files in allowlist
                    for mention in file_mentions:
                        for allowed in _allowed_paths():
                            if mention in allowed.lower():
                                target_paths.append(allowed)
                                print(f"[LEARNING] Found mentioned file: {allowed}")

                # Fallback to directory expansion if needed
                if not target_paths:
                    if "manager.py" in objective.lower():
                        manager_path = "AntAgent/autodev/manager.py"
                        if manager_path in _allowed_paths():
                            target_paths = [manager_path]
                    else:
                        # Expand directories to actual files
                        for dir_path in _ALLOWED_DIRS:
                            if Path(_project_root() / dir_path).is_dir():
                                for allowed in _allowed_paths():
                                    if allowed.startswith(dir_path + "/"):
                                        target_paths.append(allowed)
                            elif dir_path in _allowed_paths():
                                target_paths.append(dir_path)


            # --- Build smart constraints with learning ---

            if not target_paths:
                target_paths = context_paths

            base_constraints = {
                "paths": context_paths,                     # let LLM inspect the right files
                "require_context_lines": max(5, _MIN_CONTEXT_LINES),
                "allow_top_insert": False,
                "allow_lenient": False,
                # no hardcoded paths, no must_edit_one_of
            }

            # Get enhanced constraints from learning system
            constraints = learning.get_smart_constraints(objective, base_constraints)

            # Adjust based on retry count
            if i > 1:
                # Increase context and strictness on retry
                constraints["require_context_lines"] = min(
                    10, constraints["require_context_lines"] + (i - 1) * 2
                )
                print(f"[LEARNING] Retry {i}: Increased context to {constraints['require_context_lines']} lines")

                # Add more anchors from failure analysis
                if results and results[-1].get("message"):
                    # Extract potential anchors from error messages
                    error = results[-1]["message"]
                    if "not found" in error.lower():
                        print("[LEARNING] Previous anchor not found, searching for alternatives...")
                        # This triggers the file scanner in propose_patch_with_explanation

            # Add learned anchors
            lessons = get_lessons()
            anchors_learned = lessons.get("anchor_phrases") or []
            constraints.setdefault("must_anchor_any", [])

            # Prioritize highly effective anchors
            anchor_effectiveness = learning.lessons.get("anchor_effectiveness", {})
            sorted_anchors = sorted(anchor_effectiveness.items(), key=lambda x: x[1], reverse=True)

            for anchor, effectiveness in sorted_anchors[:5]:
                if anchor not in constraints["must_anchor_any"]:
                    constraints["must_anchor_any"].append(anchor)
                    print(f"[LEARNING] Added effective anchor (score {effectiveness:.2f}): {anchor[:30]}...")

            # Add anchors from similar successes
            for s in similar[:2]:
                for anchor in s.get('anchors', [])[:3]:
                    if anchor not in constraints["must_anchor_any"]:
                        constraints["must_anchor_any"].append(anchor)

            # Dynamically harvest anchors from the current file
            try:
                paths_for_scan = constraints.get("paths") or []
                if paths_for_scan:
                    primary = Path(paths_for_scan[0])
                    if primary.exists() and primary.is_file():
                        txt = primary.read_text(encoding="utf-8", errors="replace")
                        for a in _goal_adjacent_anchors(txt, objective, radius=2):
                            if a and a not in constraints["must_anchor_any"]:
                                constraints["must_anchor_any"].append(a)
            except Exception:
                pass

            # Create learning context for this attempt
            learning_ctx = LearningContext(
                timestamp=time.time(),
                goal=objective,
                file_path=target_paths[0] if target_paths else "",
                success=False,
                diff_size=0,
                context_lines_used=constraints["require_context_lines"],
                anchors_used=constraints.get("must_anchor_any", [])[:10],  # Store top 10
                retry_count=i - 1,
                llm_confidence=0.0
            )

            # --- Generate the proposed diff with enhanced prompting ---
            try:
                # Add learning hints to the goal
                enhanced_goal = f"[SELF-IMPROVE] {objective}. Keep patch minimal and targeted. Do not change tests."

                if similar:
                    hint = f" (Hint: Similar to previous change: {similar[0]['goal'][:50]})"
                    enhanced_goal += hint

                res = propose_patch_with_explanation(enhanced_goal, constraints)
            except Exception as e:
                learning_ctx.error_type = "generation_failed"
                learning_ctx.error_detail = str(e)
                learning.learn_from_attempt(learning_ctx)

                outcome = {
                    "round": i,
                    "summary": None,
                    "explanation": f"Generation failed: {e}",
                    "unified_diff": None,
                    "diff_truncated": False,
                    "applied": False,
                    "kept": False,
                    "message": f"Generation error: {e}",
                    "paths": [],
                    "checks_log": None,
                }
                results.append(outcome)
                continue

            # Extract the three parts: summary, diff, explanation
            summary: str = ""
            explanation: str = ""
            diff: str = ""
            if isinstance(res, tuple):
                if len(res) == 1:
                    diff = res[0] or ""
                elif len(res) == 2:
                    summary = res[0] or ""
                    diff = res[1] or ""
                elif len(res) >= 3:
                    summary = res[0] or ""
                    diff = res[1] or ""
                    explanation = res[2] or ""
            else:
                diff = res or ""
            diff = (diff or "").replace("\r\n", "\n").strip()
            if not diff:
                # Try to recover a diff from the explanation
                print(f"[DEBUG] No diff from primary source, trying to extract from explanation...")
                diff = _extract_unified_diff_from_text(explanation or "") or ""
                if diff:
                    print(f"[DEBUG] Recovered diff from explanation: {len(diff)} bytes")

            if not diff:
                print(f"[DEBUG] No diff produced. Summary: {summary[:100] if summary else 'none'}")
                print(f"[DEBUG] Explanation: {explanation[:200] if explanation else 'none'}")
                outcome["message"] = "no diff produced by model"
                results.append(outcome)
                continue

            # Update learning context with diff info
            learning_ctx.diff_size = len(diff)
            if explanation:
                # Simple confidence heuristic based on explanation clarity
                learning_ctx.llm_confidence = 0.7 if "line" in explanation.lower() else 0.3

            outcome = {
                "round": i,
                "summary": (summary[:240] + "…") if summary and len(summary) > 241 else summary or None,
                "explanation": explanation or None,
                "unified_diff": None,
                "diff_truncated": False,
                "applied": False,
                "kept": False,
                "message": "",
                "paths": [],
                "checks_log": None,
            }

            if not diff:
                learning_ctx.error_type = "empty_diff"
                learning_ctx.error_detail = "No diff generated"
                learning.learn_from_attempt(learning_ctx)

                outcome["message"] = "No diff produced"
                results.append(outcome)
                print("[LEARNING] No diff generated, will adjust strategy...")
                continue

            try:
                print(f"[DEBUG] Validating diff (first 500 chars):\n{diff[:500]}")
                _validate_unified_diff(diff)
                paths = _all_targets_allowed(diff)
                print(f"[DEBUG] Extracted paths from diff: {paths}")
                outcome["paths"] = paths
                learning_ctx.file_path = paths[0] if paths else ""

                if len(diff) > DIFF_RETURN_LIMIT:
                    outcome["unified_diff"] = diff[:DIFF_RETURN_LIMIT] + "\n# …(truncated)…"
                    outcome["diff_truncated"] = True
                else:
                    outcome["unified_diff"] = diff

                res_apply = apply_patch(diff)
                if not res_apply.get("applied"):
                    print(f"[DEBUG] Apply failed with error: {res_apply.get('message', 'unknown')}")
                    print(f"[DEBUG] Apply error details: {res_apply.get('error', 'no error details')}")
                    outcome["message"] = f"apply failed: {res_apply.get('message', res_apply.get('error', 'unknown'))}"


                    # Learn from apply failure
                    learning_ctx.error_type = "apply_failed"
                    learning_ctx.error_detail = res_apply.get('message', 'unknown')
                    learning.learn_from_attempt(learning_ctx)

                    # Enhanced failure learning
                    if "hunk" in outcome["message"].lower():
                        update_lessons("bad_hunk", {
                            "goal": objective,
                            "file_path": paths[0] if paths else None,
                            "context_lines": constraints["require_context_lines"]
                        })
                    elif "not found" in outcome["message"].lower():
                        update_lessons("anchor_not_found", {
                            "goal": objective,
                            "anchors": constraints.get("must_anchor_any", [])
                        })

                    results.append(outcome)
                    print(f"[LEARNING] Apply failed: {outcome['message'][:60]}")
                    continue

                ok, check_msg = _smoke_checks()
                outcome["checks_log"] = check_msg

                if ok:
                    _git_commit_paths(paths, f"self-improve: {objective} (round {i})")
                    outcome["applied"] = True
                    outcome["kept"] = True
                    outcome["message"] = "checks passed"

                    # Record success!
                    learning_ctx.success = True
                    learning.learn_from_attempt(learning_ctx)

                    print(f"[LEARNING] SUCCESS! Recording successful pattern for future use.")
                    print(
                        f"[LEARNING] Updated success rate: {learning.lessons['successful_changes']}/{learning.lessons['total_attempts']}")

                else:
                    outcome["applied"] = True
                    outcome["kept"] = False
                    outcome["message"] = "checks failed"
                    _git_restore_paths(paths)

                    # Learn from check failure
                    learning_ctx.error_type = "checks_failed"
                    learning_ctx.error_detail = check_msg
                    learning.learn_from_attempt(learning_ctx)

                    # ensure tree is clean for next round
                    rc, out, _ = _run(["git", "status", "--porcelain"])
                    if rc != 0:
                        raise RuntimeError("git status failed")

            except Exception as e:
                outcome["message"] = f"validation error: {e}"

                learning_ctx.error_type = "validation_error"
                learning_ctx.error_detail = str(e)
                learning.learn_from_attempt(learning_ctx)

            results.append(outcome)

            # Stop if successful
            if outcome.get("kept"):
                break

    # Generate learning summary
    learning_summary = {
        "pattern_type": initial_analysis['pattern_type'],
        "attempts_made": len(results),
        "success": any(r.get("kept") for r in results),
        "similar_changes_found": len(similar),
        "total_historical_attempts": learning.lessons['total_attempts'],
        "overall_success_rate": (
            learning.lessons['successful_changes'] / learning.lessons['total_attempts'] * 100
            if learning.lessons['total_attempts'] > 0 else 0
        )
    }

    return {
        "branch": work_branch,
        "base": base_branch,
        "objective": objective,
        "rounds": rounds,
        "results": results,
        "learning_summary": learning_summary
    }