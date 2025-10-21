from typing import List
from AntAgent.models import Project, PlanDraft, Task

def plan_from_goal(project: Project) -> PlanDraft:
    tasks: List[Task] = [
        Task(
            name="Define readouts & success metrics",
            outputs=["success_criteria.md"],
            est_duration_days=1.0,
            role="Design",
            qc_gate="criteria-reviewed",
        ),
        Task(
            name="Literature triage",
            depends_on=["Define readouts & success metrics"],
            inputs=["PDFs"],
            outputs=["evidence.json"],
            est_duration_days=2.0,
            role="Research",
            qc_gate="evidence-reviewed",
        ),
        Task(
            name="Design specification (planning-grade)",
            depends_on=["Literature triage"],
            outputs=["design_spec.md"],
            est_duration_days=2.0,
            role="Design",
            qc_gate="spec-approved",
        ),
    ]
    return PlanDraft(
        project=project,
        hypotheses=["Transcriptional modulation can shift visible petal traits via promoter-targeted gRNAs."],
        success_criteria=["Non-operational design dossier completed", "Risks & QC gates enumerated"],
        risks=["Supply-chain delays", "Regulatory ambiguity", "Insufficient evidence quality"],
        tasks=tasks,
    )
