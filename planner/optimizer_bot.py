from __future__ import annotations
import copy
import json
from typing import List, Dict, Any, Optional
import pandas as pd
from .schemas import Estimate, OptimizationResult, OptimizationAction, Blueprint, ProjectSpec
import logging

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

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

# LangChain output schemas
class OptimizationRecommendationsSchema(BaseModel):
    """Schema for optimization recommendations."""
    recommendations: List[Dict[str, Any]] = Field(description="List of optimization recommendations")

class IntelligentCostOptimizerAgent:
    """
    LLM-powered agent that intelligently analyzes costs and identifies optimization opportunities.
    Capabilities:
    - Advanced cost analysis and pattern recognition
    - Multi-strategy optimization recommendations
    - Reserved instance and savings plan analysis
    - Storage lifecycle optimization
    - Network cost optimization
    - Performance-cost trade-off analysis
    """
    
    def __init__(self, openai_client: Optional[ChatOpenAI] = None):
        self._langchain_client = openai_client
        self.optimization_strategies = self._load_optimization_strategies()
        self.cost_patterns = self._load_cost_patterns()
        
    @property
    def client(self):
        """Lazy initialization of LangChain client to prevent import hangs"""
        if self._langchain_client is None:
            # Check if API key is available
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")
            
            self._langchain_client = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,
                openai_api_key=api_key
            )
            print("✅ Optimizer LangChain client initialized successfully with API key")
        return self._langchain_client
    
    def _load_optimization_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load comprehensive optimization strategies for different service types."""
        return {
            "compute": {
                "spot_instances": {
                    "description": "Use spot/preemptible instances for interruptible workloads",
                    "savings_potential": 0.6,
                    "risk_level": "medium",
                    "best_for": ["training", "batch_processing", "development"],
                    "implementation": "Replace on-demand instances with spot instances where possible"
                },
                "reserved_instances": {
                    "description": "Purchase reserved instances for predictable workloads",
                    "savings_potential": 0.4,
                    "risk_level": "low",
                    "best_for": ["production", "steady_state", "long_running"],
                    "implementation": "Analyze usage patterns and purchase 1-3 year reservations"
                },
                "auto_scaling": {
                    "description": "Implement auto-scaling to match demand",
                    "savings_potential": 0.3,
                    "risk_level": "low",
                    "best_for": ["variable_load", "web_applications", "microservices"],
                    "implementation": "Set up auto-scaling groups with appropriate policies"
                }
            },
            "storage": {
                "lifecycle_policies": {
                    "description": "Implement storage lifecycle policies for cost optimization",
                    "savings_potential": 0.5,
                    "risk_level": "low",
                    "best_for": ["data_archival", "backup", "long_term_storage"],
                    "implementation": "Move infrequently accessed data to cheaper storage tiers"
                },
                "compression": {
                    "description": "Enable data compression to reduce storage costs",
                    "savings_potential": 0.2,
                    "risk_level": "low",
                    "best_for": ["databases", "log_files", "unstructured_data"],
                    "implementation": "Enable compression at the application or storage level"
                }
            },
            "network": {
                "data_transfer": {
                    "description": "Optimize data transfer costs through co-location",
                    "savings_potential": 0.7,
                    "risk_level": "low",
                    "best_for": ["cross_region", "high_egress", "data_pipeline"],
                    "implementation": "Co-locate compute and storage in the same region"
                },
                "cdn_optimization": {
                    "description": "Use CDN for static content delivery",
                    "savings_potential": 0.4,
                    "risk_level": "low",
                    "best_for": ["web_applications", "static_content", "global_users"],
                    "implementation": "Configure CDN for static assets and enable caching"
                }
            }
        }
    
    def _load_cost_patterns(self) -> Dict[str, Any]:
        """Load common cost patterns and their optimization strategies."""
        return {
            "high_compute_ratio": {
                "threshold": 0.7,
                "description": "Compute costs dominate the budget",
                "optimizations": ["spot_instances", "reserved_instances", "auto_scaling"]
            },
            "high_storage_ratio": {
                "threshold": 0.5,
                "description": "Storage costs are significant",
                "optimizations": ["lifecycle_policies", "compression", "tiering"]
            },
            "high_network_ratio": {
                "threshold": 0.4,
                "description": "Network costs are high",
                "optimizations": ["data_transfer", "cdn_optimization", "co_location"]
            }
        }
    
    def _analyze_cost_patterns(self, estimate: Estimate) -> Dict[str, Any]:
        """Analyze cost patterns to identify optimization opportunities."""
        total_cost = estimate.monthly_cost
        if total_cost == 0:
            return {}
        
        # Calculate cost ratios by service type
        compute_cost = sum(li.cost for li in estimate.bom if li.service in ["compute", "ec2", "vm"])
        storage_cost = sum(li.cost for li in estimate.bom if li.service in ["storage", "s3", "blob"])
        network_cost = sum(li.cost for li in estimate.bom if li.service in ["network", "egress"])
        
        patterns = {
            "compute_ratio": compute_cost / total_cost,
            "storage_ratio": storage_cost / total_cost,
            "network_ratio": network_cost / total_cost,
            "high_compute": compute_cost / total_cost > 0.7,
            "high_storage": storage_cost / total_cost > 0.5,
            "high_network": network_cost / total_cost > 0.4
        }
        
        return patterns
    
    def _get_llm_optimization_recommendations(self, estimate: Estimate, spec: ProjectSpec, 
                                            cost_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get LLM-powered optimization recommendations using LangChain."""
        
        try:
            # Create LangChain prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an expert FinOps consultant specializing in cloud cost optimization.
                Analyze the cost estimate and provide specific optimization recommendations.
                
                IMPORTANT: You must respond with ONLY valid JSON in exactly the specified format.
                Do not include any text before or after the JSON. Return ONLY the JSON object."""),
                ("human", """Analyze this cost estimate and provide optimization recommendations:

Cost Breakdown: {cost_breakdown}
Project Requirements: {project_req}
Cost Patterns: {patterns}

Provide optimization recommendations in this JSON format:
{{
    "recommendations": [
        {{
            "strategy": "strategy_name",
            "description": "detailed_description",
            "savings_potential": "estimated_savings_percentage",
            "implementation_effort": "low|medium|high",
            "risk_level": "low|medium|high",
            "time_to_implement": "estimated_time_in_weeks",
            "prerequisites": ["list", "of", "prerequisites"],
            "step_by_step": ["step1", "step2", "step3"]
        }}
    ]
}}

Focus on actionable, specific recommendations with realistic savings estimates.""")
            ])
            
            # Create output parser
            parser = JsonOutputParser(pydantic_object=OptimizationRecommendationsSchema)
            
            # Create the chain
            chain = prompt_template | self.client | parser
            
            # Prepare inputs
            inputs = {
                "cost_breakdown": json.dumps([li.model_dump() for li in estimate.bom], indent=2),
                "project_req": spec.model_dump(),
                "patterns": json.dumps(cost_patterns, indent=2)
            }
            
            # Invoke the chain
            result = chain.invoke(inputs)
            logger.info("✅ LangChain optimization recommendations generated successfully")
            return result.get("recommendations", [])
            
        except Exception as e:
            logger.error(f"LangChain optimization recommendations failed: {e}")
            raise RuntimeError(f"LLM optimization recommendations failed: {e}")
    
    def _apply_spot_discount(self, est: Estimate, discount: float = 0.5) -> float:
        """Apply spot instance discounts to compute resources."""
        saved = 0.0
        for li in est.bom:
            if any(k in (li.sku.lower() + " " + li.service.lower()) for k in ["gpu", "compute", "ec2", "vm"]):
                before = li.cost
                li.unit_price *= (1 - discount)
                li.cost = li.qty * li.unit_price
                saved += max(0.0, before - li.cost)
        est.monthly_cost = round(sum(li.cost for li in est.bom), 2)
        return saved

    def _apply_reserved_instance_discount(self, est: Estimate, discount: float = 0.4) -> float:
        """Apply reserved instance discounts to predictable workloads."""
        saved = 0.0
        for li in est.bom:
            if any(k in (li.sku.lower() + " " + li.service.lower()) for k in ["compute", "ec2", "vm"]):
                before = li.cost
                li.unit_price *= (1 - discount)
                li.cost = li.qty * li.unit_price
                saved += max(0.0, before - li.cost)
        est.monthly_cost = round(sum(li.cost for li in est.bom), 2)
        return saved
    
    def _apply_storage_optimization(self, est: Estimate) -> float:
        """Apply storage lifecycle optimizations."""
        saved = 0.0
        for li in est.bom:
            if li.service in ["storage", "s3", "blob"]:
                # Apply tiering optimization (move to cheaper tier)
                before = li.cost
                li.unit_price *= 0.7  # Assume 30% savings from tiering
                li.cost = li.qty * li.unit_price
                saved += max(0.0, before - li.cost)
        est.monthly_cost = round(sum(li.cost for li in est.bom), 2)
        return saved
    
    def optimize(self, bp: Blueprint, estimate: Estimate, spec: Optional[ProjectSpec] = None) -> OptimizationResult:
        """Intelligently optimize costs using multiple strategies."""
        est2 = copy.deepcopy(estimate)
        actions = []
        total_savings = 0.0
        
        # Analyze cost patterns
        cost_patterns = self._analyze_cost_patterns(est2)
        
        # Get LLM recommendations
        llm_recommendations = self._get_llm_optimization_recommendations(est2, spec, cost_patterns) if spec else []
        
        # Apply spot instance optimization
        spot_savings = self._apply_spot_discount(est2, 0.5)
        if spot_savings > 0:
            actions.append(OptimizationAction(
                type="spot_instances",
                rationale="Use spot/preemptible capacity for training nodes where interruption is tolerable.",
                delta_cost=-round(spot_savings, 2),
            ))
            total_savings += spot_savings
        
        # Apply reserved instance optimization for predictable workloads
        if spec and spec.workload.batch:  # Batch workloads are predictable
            reserved_savings = self._apply_reserved_instance_discount(est2, 0.4)
            if reserved_savings > 0:
                actions.append(OptimizationAction(
                    type="reserved_instances",
                    rationale="Use reserved instances for predictable batch workloads to reduce costs.",
                    delta_cost=-round(reserved_savings, 2),
                ))
                total_savings += reserved_savings
        
        # Apply storage optimization
        storage_savings = self._apply_storage_optimization(est2)
        if storage_savings > 0:
            actions.append(OptimizationAction(
                type="storage_lifecycle",
                rationale="Implement storage lifecycle policies to move data to cheaper tiers.",
                delta_cost=-round(storage_savings, 2),
            ))
            total_savings += storage_savings
        
        # Add LLM recommendations as additional actions
        for rec in llm_recommendations:
            if rec.get("strategy") not in [a.type for a in actions]:
                actions.append(OptimizationAction(
                    type=rec["strategy"],
                    rationale=rec["description"],
                    delta_cost=0,  # LLM doesn't provide specific cost savings
                ))
        
        return OptimizationResult(
            blueprint_id=bp.id, 
            actions=actions, 
            estimate=est2,
            metadata={
                "total_savings": round(total_savings, 2),
                "cost_patterns": cost_patterns,
                "llm_recommendations": llm_recommendations,
                "optimization_score": self._calculate_optimization_score(actions, total_savings, estimate.monthly_cost)
            }
        )
    
    def _calculate_optimization_score(self, actions: List[OptimizationAction], total_savings: float, original_cost: float) -> float:
        """Calculate an optimization score based on actions and savings."""
        if original_cost == 0:
            return 0.0
        
        # Base score from savings percentage
        savings_ratio = total_savings / original_cost
        base_score = min(savings_ratio * 100, 50)  # Cap at 50 points
        
        # Bonus points for number of strategies
        strategy_bonus = min(len(actions) * 5, 20)  # Cap at 20 points
        
        # Bonus for high-impact strategies
        high_impact_bonus = 0
        for action in actions:
            if action.type in ["spot_instances", "reserved_instances"]:
                high_impact_bonus += 10
        
        return min(base_score + strategy_bonus + high_impact_bonus, 100)
    
    def get_optimization_report(self, bp: Blueprint, estimate: Estimate, spec: Optional[ProjectSpec] = None) -> Dict[str, Any]:
        """Generate a comprehensive optimization report."""
        optimization_result = self.optimize(bp, estimate, spec)
        
        return {
            "blueprint_id": bp.id,
            "original_cost": estimate.monthly_cost,
            "optimized_cost": optimization_result.estimate.monthly_cost,
            "total_savings": optimization_result.metadata["total_savings"],
            "savings_percentage": round(
                (optimization_result.metadata["total_savings"] / estimate.monthly_cost) * 100, 2
            ) if estimate.monthly_cost > 0 else 0,
            "optimization_score": optimization_result.metadata["optimization_score"],
            "actions": [action.model_dump() for action in optimization_result.actions],
            "cost_patterns": optimization_result.metadata["cost_patterns"],
            "llm_recommendations": optimization_result.metadata["llm_recommendations"],
            "next_steps": self._generate_next_steps(optimization_result.actions)
        }
    
    def _generate_next_steps(self, actions: List[OptimizationAction]) -> List[str]:
        """Generate actionable next steps for implementation."""
        next_steps = []
        
        for action in actions:
            if action.type == "spot_instances":
                next_steps.append("Configure spot instance pools and implement fault tolerance")
            elif action.type == "reserved_instances":
                next_steps.append("Analyze usage patterns and purchase reserved instances")
            elif action.type == "storage_lifecycle":
                next_steps.append("Implement storage lifecycle policies and monitor effectiveness")
            else:
                next_steps.append(f"Research and implement {action.type} strategy")
        
        return next_steps


# Backward compatibility function
def optimize(bp: Blueprint, estimate: Estimate) -> OptimizationResult:
    """Legacy function for backward compatibility."""
    agent = IntelligentCostOptimizerAgent()
    return agent.optimize(bp, estimate)