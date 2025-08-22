# FinOps Planner (MVP)

Planner‑centric, multi‑agent FinOps copilot: describe a project → get the lowest‑cost, constraint‑aware architecture with BoM, savings, and risks.

## Quickstart
```bash
pip install -r requirements.txt
python app.py

Example prompt

Train a ViT monthly with 2 GPUs in us-central1; 2TB data (40% hot), 500GB egress, latency 200ms.

# 4) Stage & commit
```bash
git add .
git commit -m "feat: MVP planner with cost engine, optimizer, estimate guard, viz, and GCP blueprint"