from typing import List
from fastapi import UploadFile
from AntAgent.models import SequenceCheckReport, SequenceRecord


def check_sequences(files: List[UploadFile]) -> SequenceCheckReport:
    recs: List[SequenceRecord] = []
    for uf in files:
        data = uf.file.read()
        length = len(data) if data else None
        recs.append(
            SequenceRecord(
                filename=uf.filename or "sequence",
                length=length,
                features=[],
                checks=["loaded-ok" if length else "empty-file"],
            )
        )
    return SequenceCheckReport(records=recs)
