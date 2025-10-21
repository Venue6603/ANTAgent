from AntAgent.models import Project, ComplianceChecklist, ComplianceItem

def build_compliance(project: Project) -> ComplianceChecklist:
    items = [
        ComplianceItem(topic="Facility classification", requirement="Confirm BSL/plant work area classification", status="pending"),
        ComplianceItem(topic="Permits", requirement="Assess local/state permits for ornamental plant handling (non-GMO release not planned)", status="pending"),
        ComplianceItem(topic="Biosecurity", requirement="Inventory control & access logs documented", status="pending"),
    ]
    return ComplianceChecklist(items=items)
