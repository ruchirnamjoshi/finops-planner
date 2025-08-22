# planner/planner.py
"""
LLM-powered FinOps planner with lazy initialization to prevent mutex blocking.
"""
from __future__ import annotations
import os, json, uuid
from typing import Dict, Any, List, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"Warning: Failed to load .env file: {e}")

# Lazy imports to prevent blocking during module load
_openai_client = None

def get_openai_client():
    """Lazy initialization of OpenAI client to prevent import-time blocking"""
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
            
            # Check if API key is available
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")
            
            _openai_client = OpenAI(api_key=api_key)
            print("✅ OpenAI client initialized successfully with API key")
        except Exception as e:
            print(f"Warning: OpenAI client initialization failed: {e}")
            _openai_client = None
    return _openai_client

# These defaults can be overridden by settings.yaml → llm.model or env var
OPENAI_MODEL = os.getenv("PLANNER_MODEL", "gpt-4o-mini")

# ---------- LLM SPEC GENERATION ----------
SCHEMA_KEYS = {
    "name": str,
    "workload": {
        "train_gpus": int,
        "train_steps": int,
        "inference_qps": float,
        "latency_ms": int,
        "batch": bool
    },
    "data": {
        "size_gb": float,
        "growth_gb_per_month": float,
        "hot_fraction": float,
        "retention_days": int,
        "egress_gb_per_month": float
    },
    "constraints": {
        "clouds": list,
        "regions": list,
        "compliance": list,
        "region_lock": (str, type(None)),
        "managed_ok": bool,
        "serverless_ok": bool,
        "max_monthly_cost": (float, type(None))
    }
}

JSON_TEMPLATE = {
    "name": "string",
    "workload": {
        "train_gpus": 0,
        "train_steps": 0,
        "inference_qps": 0.0,
        "latency_ms": 0,
        "batch": True
    },
    "data": {
        "size_gb": 0.0,
        "growth_gb_per_month": 0.0,
        "hot_fraction": 0.3,
        "retention_days": 90,
        "egress_gb_per_month": 0.0
    },
    "constraints": {
        "clouds": [],
        "regions": [],
        "compliance": [],
        "region_lock": None,
        "managed_ok": True,
        "serverless_ok": True,
        "max_monthly_cost": None
    }
}

SYSTEM_PROMPT = f"""
You are a FinOps planner AI. Convert the user's brief into a STRICT JSON object
that matches EXACTLY this template (keys and nesting must match):

{json.dumps(JSON_TEMPLATE, indent=2)}

Rules:
- Only return JSON. No prose.
- Keys must be EXACTLY as in the template.
- Fill realistic defaults if missing; never omit required keys.
- clouds allowed: ["aws","gcp","azure"]. If user indicates one, put it in constraints.clouds.
- regions should be valid for chosen clouds. If user indicates one, add to constraints.regions.
- If training is implied, set workload.train_steps to 10000 when unknown.
- If not an online service, workload.latency_ms = 0.
- data.hot_fraction in [0,1], default 0.3. data.retention_days default 90.
- egress_gb_per_month: estimate conservatively from the brief; else 0.
- managed_ok/serverless_ok default True unless specifically disallowed.
- max_monthly_cost is null unless a budget is given.
"""

def _llm_make_spec(brief: str, run_name: str) -> Dict[str, Any]:
    """Generate project spec using LLM with error handling"""
    try:
        client = get_openai_client()
        if client is None:
            raise RuntimeError("OpenAI client not available")
            
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            max_tokens=900,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Project name: {run_name}\n\nBrief:\n{brief.strip()}"}
            ],
        )
        raw = resp.choices[0].message.content.strip()
        
        # Extract JSON
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(raw[start:end+1])
            raise RuntimeError(f"Planner LLM did not return JSON:\n{raw}")
    except Exception as e:
        print(f"LLM spec generation failed: {e}")
        # Return a basic fallback spec
        return {
            "name": run_name,
            "workload": {"train_gpus": 0, "train_steps": 0, "inference_qps": 0.0, "latency_ms": 0, "batch": True},
            "data": {"size_gb": 100.0, "growth_gb_per_month": 10.0, "hot_fraction": 0.3, "retention_days": 90, "egress_gb_per_month": 50.0},
            "constraints": {"clouds": ["aws"], "regions": ["us-east-1"], "compliance": [], "region_lock": None, "managed_ok": True, "serverless_ok": True, "max_monthly_cost": None}
        }

# ---------- Orchestration ----------
class PlannerService:
    """
    LLM-powered FinOps planning service with lazy initialization to prevent mutex blocking.
    """
    def __init__(self, settings_path: Optional[str] = None):
        # Store settings path but don't load during init to prevent blocking
        self.settings_path = settings_path
        
        # Basic settings - no file I/O during init
        self.horizon_days = 30
        self.forecast_model = "auto"
        self.forecast_horizon = 30
        self.forecast_seasonal = True
        self.run_id = None
        
        # Lazy initialization - agents will be created when needed
        self._blueprint_agent = None
        self._cost_agent = None
        self._optimizer_agent = None
        self._risk_agent = None
        
        # Lazy data loading
        self._blueprints = None
        self._sku = None
        self._history_csv_path = None

    def _initialize_agents(self):
        """Lazy initialization of agents to prevent import-time blocking"""
        try:
            if self._blueprint_agent is None:
                from .blueprint_bot import IntelligentBlueprintAgent
                self._blueprint_agent = IntelligentBlueprintAgent()
            
            if self._cost_agent is None:
                from .cost_engine import IntelligentCostEngineAgent
                self._cost_agent = IntelligentCostEngineAgent()
            
            if self._optimizer_agent is None:
                from .optimizer_bot import IntelligentCostOptimizerAgent
                self._optimizer_agent = IntelligentCostOptimizerAgent()
            
            if self._risk_agent is None:
                from .risk_bot import IntelligentRiskAssessmentAgent
                self._risk_agent = IntelligentRiskAssessmentAgent()
        except Exception as e:
            print(f"Warning: Agent initialization failed: {e}")

    def _load_data(self):
        """Lazy loading of data to prevent import-time blocking"""
        try:
            if self._blueprints is None:
                if self._blueprint_agent:
                    self._blueprints = self._blueprint_agent.load_blueprints()
            
            if self._sku is None:
                import pandas as pd
                self._sku = pd.read_csv("data/price_snapshot.csv")
            
            if self._history_csv_path is None:
                self._history_csv_path = "data/history.csv"
        except Exception as e:
            print(f"Warning: Data loading failed: {e}")

    def plan(self, brief: str) -> Dict[str, Any]:
        """LLM-powered planning with lazy initialization to prevent blocking"""
        self.run_id = str(uuid.uuid4())[:8]
        
        try:
            # Generate spec using LLM
            spec_dict = _llm_make_spec(brief, f"run-{self.run_id}")
            
            # Initialize agents and data lazily
            self._initialize_agents()
            self._load_data()
            
            # Generate blueprints using LLM agent
            if self._blueprint_agent and self._blueprints:
                try:
                    # Convert dict to ProjectSpec object for blueprint agent
                    from .schemas import ProjectSpec, Workload, DataSpec, Constraints
                    
                    # Create ProjectSpec from the dictionary
                    spec_obj = ProjectSpec(
                        name=spec_dict.get('name', 'Project'),
                        workload=Workload(**spec_dict.get('workload', {})),
                        data=DataSpec(**spec_dict.get('data', {})),
                        constraints=Constraints(**spec_dict.get('constraints', {}))
                    )
                    
                    bps = self._blueprint_agent.propose_blueprints(spec_obj, self._blueprints)
                except Exception as e:
                    print(f"Blueprint generation failed: {e}")
                    bps = []
            else:
                bps = []
            
            if not bps:
                return {
                    "spec": spec_dict,
                    "candidates": [],
                    "optimized": [],
                    "risks": {},
                    "winner": None,
                    "error": "No suitable blueprints found"
                }
            
            # Generate cost estimates using LLM agent
            estimates = []
            if self._cost_agent and self._sku is not None:
                for bp in bps:
                    try:
                        est = self._cost_agent.price_blueprint(spec_obj, bp, self._sku)
                        estimates.append(est)
                    except Exception as e:
                        # Handle both dict and object access
                        bp_id = bp.get('id', 'unknown') if hasattr(bp, 'get') else getattr(bp, 'id', 'unknown')
                        print(f"Cost estimation failed for {bp_id}: {e}")
                        estimates.append(None)
            
            # Generate optimizations using LLM agent
            optimized = []
            if self._optimizer_agent:
                for bp, est in zip(bps, estimates):
                    if est:
                        try:
                            opt = self._optimizer_agent.optimize(bp, est, spec_obj)
                            optimized.append(opt)
                        except Exception as e:
                            # Handle both dict and object access
                            bp_id = bp.get('id', 'unknown') if hasattr(bp, 'get') else getattr(bp, 'id', 'unknown')
                            print(f"Optimization failed for {bp_id}: {e}")
                            optimized.append(None)
            
            # Generate risk assessments using LLM agent
            risks = {}
            if self._risk_agent:
                for bp, est in zip(bps, estimates):
                    if est:
                        try:
                            risk = self._risk_agent.assess_plan(spec_obj, bp, est)
                            # Handle both dict and object access
                            bp_id = bp.get('id', 'unknown') if hasattr(bp, 'get') else getattr(bp, 'id', 'unknown')
                            risks[bp_id] = risk
                        except Exception as e:
                            # Handle both dict and object access
                            bp_id = bp.get('id', 'unknown') if hasattr(bp, 'get') else getattr(bp, 'id', 'unknown')
                            print(f"Risk assessment failed for {bp_id}: {e}")
                            risks[bp_id] = []
            
            # Select winner
            winner = None
            if optimized:
                valid_optimized = [opt for opt in optimized if opt is not None]
                if valid_optimized:
                    winner = min(valid_optimized, key=lambda x: getattr(x, 'monthly_cost', float('inf')))
            
            return {
                "spec": spec_dict,
                "candidates": list(zip(bps, estimates)),
                "optimized": optimized,
                "risks": risks,
                "winner": winner,
                "winner_history": None,  # Will be generated by viz agent
                "winner_forecast": None,  # Will be generated by viz agent
                "message": f"LLM-powered planning completed for {spec_dict.get('name', 'Project')}",
                "spec_obj": spec_obj  # Pass the spec object for visualization
            }
            
        except Exception as e:
            print(f"Planning failed: {e}")
            return {
                "spec": {"name": "Error", "workload": {}, "data": {}, "constraints": {}},
                "candidates": [],
                "optimized": [],
                "risks": {},
                "winner": None,
                "error": f"Planning failed: {e}"
            }