import os
from typing import List
from fastapi import UploadFile

DEST = os.path.join(os.path.dirname(__file__), "..", "data", "docs")
os.makedirs(DEST, exist_ok=True)

def ingest_pdfs(files: List[UploadFile]) -> List[str]:
    paths: List[str] = []
    for uf in files:
        fn = os.path.basename(uf.filename or "doc.pdf")
        path = os.path.abspath(os.path.join(DEST, fn))
        with open(path, "wb") as out:
            out.write(uf.file.read())
        paths.append(path)
    return paths
