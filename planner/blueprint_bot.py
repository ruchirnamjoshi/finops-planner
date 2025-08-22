from __future__ import annotations
import yaml, glob, os, json
from typing import List, Dict, Any, Optional
from openai import OpenAI
import pandas as pd
from .schemas import ProjectSpec, Blueprint
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"Warning: Failed to load .env file: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelligentBlueprintAgent:
    """
    LLM-powered agent that intelligently generates, validates, and optimizes cloud architecture blueprints.
    Capabilities:
    - Dynamic blueprint generation based on project requirements
    - Intelligent blueprint selection and ranking
    - Architecture validation and best practices enforcement
    - Multi-cloud optimization recommendations
    """
    
    def __init__(self, openai_client: Optional[OpenAI] = None):
        self._openai_client = openai_client
        self.blueprint_cache = {}
        self.architecture_patterns = self._load_architecture_patterns()
        
    @property
    def client(self):
        """Lazy initialization of OpenAI client to prevent import hangs"""
        if self._openai_client is None:
            # Check if API key is available
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")
            
            self._openai_client = OpenAI(api_key=api_key)
            print("✅ Blueprint OpenAI client initialized successfully with API key")
        return self._openai_client
    
    def _load_architecture_patterns(self) -> Dict[str, Any]:
        """Load common architecture patterns and best practices."""
        return {
            "ml_training": {
                "description": "Machine Learning Training Infrastructure",
                "components": ["compute", "storage", "network", "monitoring"],
                "best_practices": [
                    "Use spot instances for cost optimization",
                    "Implement data locality for training data",
                    "Use managed storage services for scalability",
                    "Implement checkpointing for fault tolerance"
                ]
            },
            "inference_serving": {
                "description": "ML Model Inference and Serving",
                "components": ["compute", "load_balancer", "storage", "monitoring"],
                "best_practices": [
                    "Use auto-scaling for variable load",
                    "Implement health checks and circuit breakers",
                    "Use CDN for global distribution",
                    "Monitor latency and throughput"
                ]
            },
            "data_warehouse": {
                "description": "Data Warehouse and Analytics",
                "components": ["compute", "storage", "database", "etl"],
                "best_practices": [
                    "Use columnar storage for analytics",
                    "Implement data partitioning strategies",
                    "Use managed database services",
                    "Implement data lifecycle management"
                ]
            }
        }
    
    def _generate_dynamic_blueprint(self, spec: ProjectSpec, cloud: str, region: str) -> Blueprint:
        """Generate a new blueprint dynamically using LLM based on project requirements."""
        
        system_prompt = f"""You are an expert cloud architect specializing in {cloud.upper()}. 
        Generate a detailed cloud infrastructure blueprint for the following project requirements.
        
        Project: {spec.name}
        Workload: {spec.workload.model_dump()}
        Data: {spec.data.model_dump()}
        Constraints: {spec.constraints.model_dump()}
        
        Return a JSON blueprint with the following structure:
        {{
            "id": "unique_blueprint_id",
            "cloud": "{cloud}",
            "region": "{region}",
            "services": [
                {{
                    "service": "service_name",
                    "sku": "specific_sku",
                    "qty_expr": "quantity_expression",
                    "unit": "billing_unit"
                }}
            ],
            "assumptions": {{
                "architecture_type": "ml_training|inference_serving|data_warehouse",
                "scalability_strategy": "auto_scaling|manual|reserved",
                "cost_optimization": ["spot_instances", "reserved_instances", "storage_tiering"],
                "security_features": ["vpc", "encryption", "iam"],
                "monitoring": ["cloudwatch", "prometheus", "custom"],
                "notes": "Architecture rationale and considerations"
            }}
        }}
        
        Ensure the blueprint follows {cloud.upper()} best practices and is cost-optimized."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=1000,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate a {cloud} blueprint for {spec.name}"}
                ]
            )
            
            blueprint_data = json.loads(response.choices[0].message.content)
            return Blueprint(**blueprint_data)
            
        except Exception as e:
            logger.error(f"Failed to generate dynamic blueprint: {e}")
            return self._create_fallback_blueprint(spec, cloud, region)
    
    def _create_fallback_blueprint(self, spec: ProjectSpec, cloud: str, region: str) -> Blueprint:
        """Create a fallback blueprint when LLM generation fails."""
        return Blueprint(
            id=f"{cloud}_{region}_fallback",
            cloud=cloud,
            region=region,
            services=[
                {
                    "service": "compute",
                    "sku": f"{cloud}_default_compute",
                    "qty_expr": "1",
                    "unit": "instance_month"
                }
            ],
            assumptions={
                "architecture_type": "fallback",
                "notes": "Fallback blueprint created due to generation failure"
            }
        )
    
    def _validate_blueprint(self, bp: Blueprint, spec: ProjectSpec) -> Dict[str, Any]:
        """Validate blueprint against best practices and project requirements."""
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Check required services
        required_services = {"compute", "storage", "network"}
        blueprint_services = {svc["service"] for svc in bp.services}
        missing_services = required_services - blueprint_services
        
        if missing_services:
            validation_results["warnings"].append(f"Missing services: {missing_services}")
        
        # Check cost optimization opportunities
        if not any("spot" in svc.get("sku", "").lower() for svc in bp.services):
            validation_results["recommendations"].append("Consider using spot instances for cost optimization")
        
        # Check data locality
        if spec.data.size_gb > 1000 and "storage" in blueprint_services:
            validation_results["recommendations"].append("Large dataset detected - ensure compute and storage are co-located")
        
        return validation_results
    
    def _rank_blueprints(self, blueprints: List[Blueprint], spec: ProjectSpec) -> List[Blueprint]:
        """Intelligently rank blueprints based on multiple criteria."""
        scored_blueprints = []
        
        for bp in blueprints:
            score = 0
            
            # Workload-specific scoring
            if spec.workload.train_gpus > 0:  # ML Training workload
                if "gpu" in bp.id.lower() or "training" in bp.id.lower():
                    score += 50  # Perfect match for ML training
                elif "data" in bp.id.lower() or "warehouse" in bp.id.lower():
                    score += 20  # Good for data processing
                else:
                    score += 10  # Basic compute
            
            elif spec.workload.inference_qps > 500:  # High-traffic workload
                if "web" in bp.id.lower() or "app" in bp.id.lower():
                    score += 50  # Perfect match for web apps
                elif "gpu" in bp.id.lower():
                    score += 30  # GPU can handle high traffic
                else:
                    score += 15  # Basic compute
            
            elif spec.data.size_gb > 5000:  # Data-heavy workload
                if "data" in bp.id.lower() or "warehouse" in bp.id.lower():
                    score += 50  # Perfect match for data workloads
                elif "gpu" in bp.id.lower():
                    score += 25  # GPU can process large datasets
                else:
                    score += 10  # Basic storage
            
            else:  # General workload
                score += 25  # Base score for general workloads
            
            # Cost optimization score
            if any("spot" in svc.get("sku", "").lower() for svc in bp.services):
                score += 20
            
            # Service coverage score
            service_coverage = len(bp.services)
            score += min(service_coverage * 5, 25)
            
            # Region preference score
            if spec.constraints.region_lock == bp.region:
                score += 15
            
            # Cloud preference score
            if spec.constraints.clouds and bp.cloud in spec.constraints.clouds:
                score += 10
            
            scored_blueprints.append((score, bp))
        
        # Sort by score (highest first) and return blueprints
        # Use stable sort with blueprint ID as tiebreaker to avoid comparison errors
        return [bp for score, bp in sorted(scored_blueprints, key=lambda x: (x[0], x[1].id), reverse=True)]
    
    def load_blueprints(self, paths=None) -> List[Blueprint]:
        """Load blueprints from YAML files with enhanced error handling."""
        bps: List[Blueprint] = []
        
        # Handle both string glob pattern and list of specific paths
        if isinstance(paths, str):
            # Single glob pattern
            file_paths = glob.glob(paths)
        elif isinstance(paths, list):
            # List of specific file paths
            file_paths = paths
        else:
            # Default fallback
            file_paths = glob.glob("blueprints/*.yaml")
        
        for f in file_paths:
            try:
                with open(f, "r") as fh:
                    data = yaml.safe_load(fh)
                if not isinstance(data, dict):
                    logger.warning(f"Skipping blueprint (empty/invalid): {os.path.basename(f)}")
                    continue
                bps.append(Blueprint(**data))
            except yaml.YAMLError as e:
                logger.error(f"YAML parse error in {f}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error loading {f}: {e}")
        
        if not bps:
            logger.warning("No valid blueprints found")
        
        return bps
    
    def propose_blueprints(self, spec: ProjectSpec, blueprints: List[Blueprint]) -> List[Blueprint]:
        """Intelligently propose and rank blueprints based on project requirements."""
        if not blueprints:
            return []
        
        # Filter by basic constraints
        c = spec.constraints
        candidates = [b for b in blueprints
                      if (not c.clouds or b.cloud in c.clouds)
                      and (not c.regions or b.region in c.regions)]
        
        if not candidates:
            # Generate dynamic blueprints if no static ones match
            logger.info("No matching blueprints found, generating dynamic ones...")
            for cloud in (c.clouds or ["aws", "azure", "gcp"]):
                for region in (c.regions or ["us-east-1", "eastus", "us-central1"]):
                    dynamic_bp = self._generate_dynamic_blueprint(spec, cloud, region)
                    candidates.append(dynamic_bp)
        
        # Validate and rank blueprints
        validated_candidates = []
        for bp in candidates:
            validation = self._validate_blueprint(bp, spec)
            if validation["is_valid"]:
                validated_candidates.append(bp)
                if validation["warnings"] or validation["recommendations"]:
                    logger.info(f"Blueprint {bp.id}: {validation['warnings']} {validation['recommendations']}")
        
        # Rank and return top candidates
        ranked_candidates = self._rank_blueprints(validated_candidates, spec)
        return ranked_candidates[:5]  # Return top 5 instead of 3
    
    def get_blueprint_recommendations(self, spec: ProjectSpec, blueprints: List[Blueprint]) -> Dict[str, Any]:
        """Get comprehensive blueprint recommendations with analysis."""
        proposed = self.propose_blueprints(spec, blueprints)
        
        recommendations = {
            "top_blueprints": proposed,
            "analysis": {
                "total_candidates": len(proposed),
                "cloud_distribution": {},
                "cost_optimization_opportunities": [],
                "risk_factors": []
            }
        }
        
        # Analyze cloud distribution
        for bp in proposed:
            cloud = bp.cloud
            recommendations["analysis"]["cloud_distribution"][cloud] = \
                recommendations["analysis"]["cloud_distribution"].get(cloud, 0) + 1
        
        # Identify cost optimization opportunities
        for bp in proposed:
            if any("spot" in svc.get("sku", "").lower() for svc in bp.services):
                recommendations["analysis"]["cost_optimization_opportunities"].append(
                    f"{bp.id}: Spot instances available"
                )
        
        return recommendations

# Backward compatibility functions
def load_blueprints(path: str = "blueprints/*.yaml") -> List[Blueprint]:
    """Legacy function for backward compatibility."""
    agent = IntelligentBlueprintAgent()
    return agent.load_blueprints(path)

def propose_blueprints(spec: ProjectSpec, blueprints: List[Blueprint]) -> List[Blueprint]:
    """Legacy function for backward compatibility."""
    agent = IntelligentBlueprintAgent()
    return agent.propose_blueprints(spec, blueprints)