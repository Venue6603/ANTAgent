from typing import Dict, Any
from .sandbox import run_checks

def quality_report() -> Dict[str, Any]:
    return {"tools": run_checks()}
