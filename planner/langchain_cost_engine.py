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
        """Load pricing insights - now LLM-driven only."""
        # No hardcoded insights - LLM will analyze and provide specific insights
        return {}
    
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
            
            # Invoke the chain safely - NO FALLBACKS
            cost_insights = self._invoke_chain_safely(
                cost_chain, 
                chain_inputs
            )
            
            # Create the final estimate with insights
            estimate = self._create_estimate_with_insights(spec, bp, cost_insights)
            
            logger.info("✅ LangChain cost analysis completed successfully")
            return estimate
            
        except Exception as e:
            logger.error(f"LangChain cost analysis failed: {e}")
            # No fallbacks - raise error
            raise RuntimeError(f"LLM cost analysis failed - no fallback available: {e}")
    
    def _create_basic_estimate(self, spec: ProjectSpec, bp: Blueprint) -> Estimate:
        """This method is deprecated - LLM insights must drive all calculations."""
        raise NotImplementedError("Hardcoded estimates not allowed - LLM insights required")
    
    def _create_estimate_with_insights(self, spec: ProjectSpec, bp: Blueprint, cost_insights: Dict[str, Any]) -> Estimate:
        """Create an estimate enriched with LLM insights."""
        # This method should be implemented based on LLM insights only
        # No hardcoded values - all must come from LLM
        return {}
    
    def process(self, spec: ProjectSpec, bp: Blueprint, sku_pricing: pd.DataFrame) -> Estimate:
        """Main processing method - alias for price_blueprint."""
        return self.price_blueprint(spec, bp, sku_pricing)
