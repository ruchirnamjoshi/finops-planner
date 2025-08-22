from __future__ import annotations
import copy
import json, logging
from typing import List, Dict, Any, Optional
import pandas as pd
from .schemas import Estimate, OptimizationResult, OptimizationAction, Blueprint, ProjectSpec
import logging

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CostPatternsSchema(BaseModel):
    """Schema for cost pattern analysis."""
    compute_ratio: float = Field(description="Compute cost as percentage of total")
    storage_ratio: float = Field(description="Storage cost as percentage of total")
    network_ratio: float = Field(description="Network cost as percentage of total")
    dominant_cost_driver: str = Field(description="Dominant cost driver (compute|storage|network)")
    cost_efficiency_score: float = Field(description="Cost efficiency score (0-100)")
    optimization_priority: str = Field(description="Optimization priority focus")
    high_compute: bool = Field(description="Whether compute costs are high")
    high_storage: bool = Field(description="Whether storage costs are high")
    high_network: bool = Field(description="Whether network costs are high")
    efficiency_analysis: str = Field(description="Brief explanation of efficiency score")

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
    
    def _load_optimization_strategies(self) -> Dict[str, Any]:
        """Load optimization strategies - now LLM-driven only."""
        # No hardcoded strategies - LLM will generate all recommendations
        return {}
    
    def _load_cost_patterns(self) -> Dict[str, Any]:
        """Load cost patterns - now LLM-driven only."""
        # No hardcoded patterns - LLM will analyze and identify patterns
        return {}
    
    def _analyze_cost_patterns(self, estimate: Estimate) -> Dict[str, Any]:
        """Analyze cost patterns using LLM for intelligent insights."""
        try:
            logger.info("🎯 Generating LLM-powered cost pattern analysis")
            
            # Calculate actual cost ratios from BOM
            total_cost = estimate.monthly_cost
            compute_cost = sum(item.cost for item in estimate.bom if 'compute' in item.service.lower() or 'ec2' in item.service.lower() or 'gpu' in item.service.lower())
            storage_cost = sum(item.cost for item in estimate.bom if 'storage' in item.service.lower() or 's3' in item.service.lower() or 'rds' in item.service.lower())
            network_cost = sum(item.cost for item in estimate.bom if 'network' in item.service.lower() or 'cloudfront' in item.service.lower() or 'transfer' in item.service.lower())
            
            # Calculate actual percentages
            compute_ratio = (compute_cost / total_cost * 100) if total_cost > 0 else 0
            storage_ratio = (storage_cost / total_cost * 100) if total_cost > 0 else 0
            network_ratio = (network_cost / total_cost * 100) if total_cost > 0 else 0
            
            # Determine workload type for context
            workload_type = "general"
            if hasattr(estimate, 'metadata') and estimate.metadata:
                if 'gpu' in str(estimate.metadata).lower() or 'training' in str(estimate.metadata).lower():
                    workload_type = "ml_training"
                elif 'web' in str(estimate.metadata).lower() or 'app' in str(estimate.metadata).lower():
                    workload_type = "web_application"
                elif 'data' in str(estimate.metadata).lower() or 'warehouse' in str(estimate.metadata).lower():
                    workload_type = "data_warehouse"
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an expert FinOps analyst specializing in cloud cost pattern analysis.
                
                Your task: Analyze cost patterns and provide intelligent insights about cost efficiency.
                
                CRITICAL: Generate realistic efficiency scores based on the actual cost distribution.
                DO NOT default to 0/100 - analyze the patterns intelligently.
                
                IMPORTANT: Respond with ONLY valid JSON in the exact specified format."""),
                
                ("human", """Analyze this cost breakdown and identify cost patterns:

Cost Breakdown: {cost_breakdown}

Actual Cost Ratios:
- Compute: {compute_ratio:.1f}%
- Storage: {storage_ratio:.1f}%
- Network: {network_ratio:.1f}%
- Total Cost: ${total_cost:,.2f}

Workload Type: {workload_type}

Provide analysis in this JSON format:
{{
    "compute_ratio": {compute_ratio:.1f},
    "storage_ratio": {storage_ratio:.1f},
    "network_ratio": {network_ratio:.1f},
    "dominant_cost_driver": "compute|storage|network",
    "cost_efficiency_score": 0-100,
    "optimization_priority": "compute|storage|network|overall",
    "high_compute": true/false,
    "high_storage": true/false,
    "high_network": true/false,
    "efficiency_analysis": "brief explanation of the score"
}}

EFFICIENCY SCORING GUIDELINES:
- 80-100: Excellent (well-optimized, balanced costs)
- 60-79: Good (reasonable distribution, some optimization possible)
- 40-59: Fair (imbalanced costs, optimization needed)
- 0-39: Poor (significant cost issues, major optimization required)

Consider the workload type when scoring:
- ML Training: Higher compute costs are expected
- Web Apps: Balanced compute/storage is ideal
- Data Warehouse: Higher storage costs are expected
- General: Balanced distribution is best""")
            ])
            
            # Create output parser
            parser = JsonOutputParser(pydantic_object=CostPatternsSchema)
            
            # Create the chain
            chain = prompt_template | self.client | parser
            
            # Prepare inputs
            inputs = {
                "cost_breakdown": [item.model_dump() for item in estimate.bom],
                "compute_ratio": compute_ratio,
                "storage_ratio": storage_ratio,
                "network_ratio": network_ratio,
                "total_cost": total_cost,
                "workload_type": workload_type
            }
            
            # Invoke the chain
            result = chain.invoke(inputs)
            logger.info("✅ LLM cost pattern analysis generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"LLM cost pattern analysis failed: {e}")
            # No fallbacks - raise error
            raise RuntimeError(f"LLM cost pattern analysis failed - no fallback available: {e}")
    
    def _get_llm_optimization_recommendations(self, estimate: Estimate, spec: ProjectSpec, cost_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get LLM-powered optimization recommendations."""
        try:
            logger.info("🎯 Generating LLM-powered optimization recommendations")
            
            # Determine workload type for context-specific recommendations
            workload_type = "general"
            if spec.workload.train_gpus > 0:
                workload_type = "ml_training"
            elif spec.workload.inference_qps > 500:
                workload_type = "high_traffic"
            elif spec.data.size_gb > 5000:
                workload_type = "data_intensive"
            
            # Get blueprint services for context
            blueprint_services = []
            if hasattr(estimate, 'bom') and estimate.bom:
                blueprint_services = [item.service for item in estimate.bom]
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an expert FinOps consultant specializing in cloud cost optimization.
                
                Your task: Generate specific, actionable optimization recommendations.
                
                CRITICAL: Make recommendations SPECIFIC to the workload type and services used.
                DO NOT provide generic advice - make it relevant to the actual architecture.
                
                IMPORTANT: Respond with ONLY valid JSON in the exact specified format."""),
                
                ("human", """Generate optimization recommendations for this cloud infrastructure:

Project Details:
- Workload Type: {workload_type}
- Total Cost: ${total_cost:,.2f}
- Services Used: {services}

Cost Patterns:
- Compute Ratio: {compute_ratio:.1f}%
- Storage Ratio: {storage_ratio:.1f}%
- Network Ratio: {network_ratio:.1f}%
- Dominant Driver: {dominant_driver}
- Efficiency Score: {efficiency_score}/100

WORKLOAD-SPECIFIC OPTIMIZATION FOCUS:
- ML Training: GPU optimization, spot instances, training efficiency
- High Traffic: Auto-scaling, CDN, database optimization
- Data Intensive: Storage tiering, query optimization, ETL efficiency
- General: Resource sizing, utilization, cost management

Provide 3-5 specific recommendations in this JSON format:
[
    {{
        "strategy": "strategy_name",
        "description": "detailed description of the optimization",
        "savings_potential": "estimated savings (e.g., 20-30%)",
        "implementation_effort": "effort level (low|medium|high)",
        "workload_relevance": "why this applies to this workload type"
    }}
]

Make each recommendation specific to the actual services and workload type."""),
            ])
            
            # Create output parser
            parser = JsonOutputParser()
            
            # Create the chain
            chain = prompt_template | self.client | parser
            
            # Prepare inputs
            inputs = {
                "workload_type": workload_type,
                "total_cost": estimate.monthly_cost,
                "services": ", ".join(blueprint_services),
                "compute_ratio": cost_patterns.get("compute_ratio", 0),
                "storage_ratio": cost_patterns.get("storage_ratio", 0),
                "network_ratio": cost_patterns.get("network_ratio", 0),
                "dominant_driver": cost_patterns.get("dominant_cost_driver", "unknown"),
                "efficiency_score": cost_patterns.get("cost_efficiency_score", 0)
            }
            
            # Invoke the chain
            result = chain.invoke(inputs)
            logger.info("✅ LangChain optimization recommendations generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"LLM optimization recommendations failed: {e}")
            # No fallbacks - raise error
            raise RuntimeError(f"LLM optimization recommendations failed - no fallback available: {e}")
    
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
                # Apply storage tiering optimization
                if li.service == "storage" and "standard" in li.sku.lower():
                    # Let LLM determine the savings percentage
                    # No hardcoded assumptions
                    li.unit_price *= 0.8  # Basic assumption, but LLM should specify
                before = li.cost
                # Let LLM determine the exact savings percentage
                # No hardcoded assumptions
                li.unit_price *= 0.8  # Basic assumption, but LLM should specify
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