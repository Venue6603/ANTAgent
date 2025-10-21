from __future__ import annotations

from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field


# ---------- Core project + planning ----------

class Project(BaseModel):
    name: str = Field(..., description="Project name")
    objective: str = Field(..., description="High-level objective (planning-only; no operational steps)")
    constraints: Dict[str, object] = Field(default_factory=dict)
    budget: Optional[float] = Field(None, description="USD budget cap")
    deadline: Optional[str] = Field(None, description="ISO date string")


class Task(BaseModel):
    name: str
    depends_on: List[str] = Field(default_factory=list)
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    est_duration_days: float = 1.0
    role: Literal["Research", "Design", "Ops", "Regulatory", "PM"] = "Research"
    risk_level: Literal["low", "medium", "high"] = "medium"
    qc_gate: str = "review"


class PlanDraft(BaseModel):
    project: Project
    hypotheses: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    tasks: List[Task] = Field(default_factory=list)


# ---------- Literature / evidence ----------

class EvidenceItem(BaseModel):
    source: str
    fields: Dict[str, str] = Field(default_factory=dict)


class EvidenceTable(BaseModel):
    items: List[EvidenceItem] = Field(default_factory=list)


# ---------- Sequence validation (read-only) ----------

class SequenceRecord(BaseModel):
    filename: str
    length: Optional[int] = None
    features: List[str] = Field(default_factory=list)
    checks: List[str] = Field(default_factory=list)


class SequenceCheckReport(BaseModel):
    records: List[SequenceRecord] = Field(default_factory=list)


# ---------- Costing / BOM ----------

class BOMSpec(BaseModel):
    targets: List[str] = Field(default_factory=list)
    vendor_csv_paths: List[str] = Field(default_factory=list)


class BOMItem(BaseModel):
    name: str
    vendor: Optional[str] = None
    catalog_no: Optional[str] = None
    unit_cost: Optional[float] = None
    qty: int = 1


class BOMTable(BaseModel):
    items: List[BOMItem] = Field(default_factory=list)
    est_total_cost: Optional[float] = None


# ---------- Compliance ----------

class ComplianceItem(BaseModel):
    topic: str
    requirement: str
    status: Literal["n/a", "pending", "ok", "risk"] = "pending"
    evidence: Optional[str] = None


class ComplianceChecklist(BaseModel):
    items: List[ComplianceItem] = Field(default_factory=list)


# ---------- Timeline / critical path ----------

class TimelineSpec(BaseModel):
    tasks: List[Task] = Field(default_factory=list)


class Timeline(BaseModel):
    tasks: List[Task] = Field(default_factory=list)
    critical_path: List[str] = Field(default_factory=list)


# ---------- Reporting ----------

class ReportRequest(BaseModel):
    project_name: str
    objective: str
    plan: PlanDraft
    bom: BOMTable
    timeline: Timeline
    compliance: ComplianceChecklist


class ReportBundle(BaseModel):
    path_markdown: str


__all__ = [
    "Project",
    "Task",
    "PlanDraft",
    "EvidenceItem",
    "EvidenceTable",
    "SequenceRecord",
    "SequenceCheckReport",
    "BOMSpec",
    "BOMItem",
    "BOMTable",
    "ComplianceItem",
    "ComplianceChecklist",
    "TimelineSpec",
    "Timeline",
    "ReportRequest",
    "ReportBundle",
]
