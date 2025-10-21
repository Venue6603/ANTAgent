import re
from fastapi import HTTPException

BLOCK = re.compile(r"\b(pipett|incubat|centrifug|aliquot|vortex|rpm|°C|degrees|[0-9]+\s?(min|hour|h|sec|s)\b)", re.I)

def enforce_non_operational(text: str) -> None:
    if not text:
        return
    if BLOCK.search(text):
        raise HTTPException(status_code=400, detail="Planning-grade only. No operational lab steps.")
