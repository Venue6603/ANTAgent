# Lab Planning & Protocol Assistant (Local, Self-Improving)

## Setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download a local instruct model (e.g., Qwen2.5-14B-Instruct-Q4_K_M.gguf)
export LABAGENT_MODEL_PATH="/path/to/Qwen2.5-14B-Instruct-Q4_K_M.gguf"

## Run
uvicorn app:app --reload --port 8008

---

### Planning Endpoints
POST /plan/draft  
POST /literature/ingest  
POST /literature/extract  
POST /sequence/check  
POST /bom/build  
POST /compliance/check  
POST /timeline/build  
POST /report/package  

### Self-Improvement Endpoints
POST /code/list  
POST /code/read  
POST /code/propose_patch  
POST /code/run_checks  
POST /code/apply_patch  

### Self-Improvement Flow
1. `/code/list` → choose files  
2. `/code/read` → get code content  
3. `/code/propose_patch` → LLM proposes unified diff  
4. `/code/run_checks` → run formatting/lint/test/type/security checks  
5. `/code/apply_patch` → apply diff & re-run checks  

All writes are limited to `autodev/allowlist.txt`.  
