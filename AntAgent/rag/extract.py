from AntAgent.models import EvidenceTable, EvidenceItem


def extract_facts(spec: dict) -> EvidenceTable:
    # Minimal placeholder: return empty evidence rows
    fields = spec.get("fields") or []
    _ = fields  # unused in stub
    return EvidenceTable(items=[EvidenceItem(source="placeholder", fields={})])
