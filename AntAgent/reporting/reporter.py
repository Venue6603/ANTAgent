import os
from jinja2 import Environment, FileSystemLoader, select_autoescape
from AntAgent.models import ReportRequest, ReportBundle


TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "deliverables"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def package_report(req: ReportRequest) -> ReportBundle:
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=select_autoescape())
    tpl = env.get_template("executive_summary.md.j2")
    md = tpl.render(
        project_name=req.project_name,
        objective=req.objective,
        plan=req.plan,
        bom=req.bom,
        timeline=req.timeline,
        compliance=req.compliance,
    )
    out_md = os.path.join(OUTPUT_DIR, f"{req.project_name.replace(' ', '_')}_ExecutiveSummary.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    return ReportBundle(path_markdown=out_md)
