"""
LangChain-based Cost Engine Agent for consistent LLM-powered cost analysis.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
import pandas as pd

from .langchain_base import LangChainBaseAgent, CostInsightsSchema
from .schemas import ProjectSpec, Blueprint, Estimate

logger = logging.getLogger(__name__)

class LangChainCostEngineAgent(LangChainBaseAgent):
    """
    LangChain-based agent that provides sophisticated cost analysis and pricing insights.
    """
    
    def __init__(self, openai_client: Optional[Any] = None):
        super().__init__(model_name="gpt-4o-mini", temperature=0.3)
        self.pricing_insights = self._load_pricing_insights()
        self.cost_models = self._load_cost_models()
    
    def _load_pricing_insights(self) -> Dict[str, Dict[str, Any]]:
        """Load pricing insights and trends for different service types."""
        return {
            "compute": {
                "trends": {
                    "spot_instances": "Generally 60-90% cheaper than on-demand",
                    "reserved_instances": "1-year: 30-40% savings, 3-year: 50-60% savings",
                    "savings_plans": "Flexible pricing with 20-40% savings",
                    "on_demand": "Most expensive but most flexible"
                },
                "optimization_tips": [
                    "Use spot instances for interruptible workloads",
                    "Purchase reserved instances for predictable usage",
                    "Consider savings plans for mixed workloads",
                    "Implement auto-scaling to optimize utilization"
                ]
            },
            "storage": {
                "trends": {
                    "hot_storage": "Most expensive but fastest access",
                    "cool_storage": "30-40% cheaper than hot storage",
                    "archive_storage": "70-90% cheaper than hot storage",
                    "lifecycle_policies": "Can reduce costs by 50-80%"
                },
                "optimization_tips": [
                    "Implement storage lifecycle policies",
                    "Use appropriate storage tiers",
                    "Enable compression where possible",
                    "Consider data archival strategies"
                ]
            },
            "network": {
                "trends": {
                    "data_transfer": "Can be 50-80% of total costs for data-heavy workloads",
                    "cdn": "Reduces data transfer costs by 30-50%",
                    "co_location": "Can reduce network costs by 60-80%"
                },
                "optimization_tips": [
                    "Co-locate compute and storage",
                    "Use CDN for static content",
                    "Optimize data transfer patterns",
                    "Consider edge locations"
                ]
            }
        }
    
    def _load_cost_models(self) -> Dict[str, Dict[str, Any]]:
        """Load cost models for different workload types."""
        return {
            "ml_training": {
                "compute_ratio": 0.7,
                "storage_ratio": 0.2,
                "network_ratio": 0.1,
                "optimization_potential": 0.4,
                "key_factors": ["gpu_utilization", "data_locality", "checkpointing"]
            },
            "web_application": {
                "compute_ratio": 0.5,
                "storage_ratio": 0.3,
                "network_ratio": 0.2,
                "optimization_potential": 0.3,
                "key_factors": ["auto_scaling", "cdn_usage", "database_optimization"]
            },
            "data_warehouse": {
                "compute_ratio": 0.4,
                "storage_ratio": 0.5,
                "network_ratio": 0.1,
                "optimization_potential": 0.5,
                "key_factors": ["storage_tiering", "query_optimization", "data_compression"]
            }
        }
    
    def price_blueprint(self, spec: ProjectSpec, bp: Blueprint, sku_pricing: pd.DataFrame) -> Estimate:
        """
        Generate cost estimate for a blueprint using LangChain-powered analysis.
        """
        try:
            # Create the cost insights chain
            cost_chain = self._create_structured_chain(
                prompt_template="""
                Analyze this cloud infrastructure plan and provide detailed cost insights.
                
                Project Specification:
                {spec}
                
                Blueprint:
                {blueprint}
                
                Cost Estimate:
                {estimate}
                
                Pricing Data:
                {pricing}
                
                Provide comprehensive cost analysis including breakdown, optimization opportunities, 
                pricing insights, and cost forecasting.
                """,
                output_schema=CostInsightsSchema,
                system_message="You are an expert FinOps consultant specializing in cloud cost optimization. Provide actionable insights and realistic cost optimization opportunities."
            )
            
            # Prepare inputs for the chain
            chain_inputs = {
                "spec": spec.model_dump(),
                "blueprint": bp.model_dump(),
                "estimate": self._create_basic_estimate(spec, bp),
                "pricing": sku_pricing.to_dict('records')
            }
            
            # Invoke the chain safely
            cost_insights = self._invoke_chain_safely(
                cost_chain, 
                chain_inputs,
                fallback=self._get_fallback_cost_insights(spec, bp)
            )
            
            # Create the final estimate with insights
            estimate = self._create_estimate_with_insights(spec, bp, cost_insights)
            
            logger.info("✅ LangChain cost analysis completed successfully")
            return estimate
            
        except Exception as e:
            logger.error(f"LangChain cost analysis failed: {e}")
            # Fallback to basic estimate
            return self._create_basic_estimate(spec, bp)
    
    def _create_basic_estimate(self, spec: ProjectSpec, bp: Blueprint) -> Estimate:
        """Create a basic cost estimate without LLM insights."""
        from .schemas import Estimate, LineItem
        
        # Basic cost calculation based on blueprint and spec
        line_items = []
        total_cost = 0.0
        
        # Add compute costs
        if spec.workload.train_gpus > 0:
            gpu_cost = spec.workload.train_gpus * 2.0  # $2/hour per GPU
            line_items.append(LineItem(
                service="compute",
                sku="gpu-instance",
                qty=spec.workload.train_gpus,
                unit_price=gpu_cost / spec.workload.train_gpus,
                cost=gpu_cost
            ))
            total_cost += gpu_cost
        
        # Add storage costs
        storage_cost = spec.data.size_gb * 0.023  # $0.023/GB/month
        line_items.append(LineItem(
            service="storage",
            sku="standard-storage",
            qty=spec.data.size_gb,
            unit_price=0.023,
            cost=storage_cost
        ))
        total_cost += storage_cost
        
        # Add network costs
        network_cost = spec.data.egress_gb_per_month * 0.09  # $0.09/GB
        line_items.append(LineItem(
            service="network",
            sku="data-transfer",
            qty=spec.data.egress_gb_per_month,
            unit_price=0.09,
            cost=network_cost
        ))
        total_cost += network_cost
        
        return Estimate(
            monthly_cost=total_cost,
            bom=line_items,
            metadata={
                "cost_breakdown": {
                    "compute": sum(li.cost for li in line_items if li.service == "compute"),
                    "storage": sum(li.cost for li in line_items if li.service == "storage"),
                    "network": sum(li.cost for li in line_items if li.service == "network")
                },
                "llm_insights": None
            }
        )
    
    def _create_estimate_with_insights(self, spec: ProjectSpec, bp: Blueprint, cost_insights: Dict[str, Any]) -> Estimate:
        """Create an estimate enriched with LLM insights."""
        base_estimate = self._create_basic_estimate(spec, bp)
        
        # Update metadata with LLM insights
        base_estimate.metadata["llm_insights"] = cost_insights
        
        return base_estimate
    
    def _get_fallback_cost_insights(self, spec: ProjectSpec, bp: Blueprint) -> Dict[str, Any]:
        """Get fallback cost insights when LLM fails."""
        total_cost = 1000.0  # Default cost
        
        return {
            "cost_breakdown_analysis": {
                "compute_percentage": 60.0,
                "storage_percentage": 25.0,
                "network_percentage": 15.0,
                "dominant_cost_driver": "compute"
            },
            "optimization_opportunities": [
                {
                    "category": "compute",
                    "description": "Consider using spot instances for interruptible workloads",
                    "potential_savings": "40-60%",
                    "effort_required": "medium",
                    "implementation_steps": ["Identify suitable workloads", "Configure spot pools", "Implement fault handling"]
                }
            ],
            "pricing_insights": [
                {
                    "insight": "GPU instances dominate compute costs",
                    "impact": "High compute costs due to GPU requirements",
                    "action": "Consider spot instances and reserved capacity"
                }
            ],
            "cost_forecast": {
                "trend": "stable",
                "factors": ["Data growth", "Compute scaling"],
                "recommendations": ["Monitor usage patterns", "Implement cost controls"]
            }
        }
    
    def process(self, spec: ProjectSpec, bp: Blueprint, sku_pricing: pd.DataFrame) -> Estimate:
        """Main processing method - alias for price_blueprint."""
        return self.price_blueprint(spec, bp, sku_pricing)
