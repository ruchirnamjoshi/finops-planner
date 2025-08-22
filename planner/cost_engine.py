#!/usr/bin/env python3
"""
100% LLM-Powered Cost Engine Agent
No external data, no hardcoded values, pure AI-driven cost estimation
"""

import logging
from typing import Dict, Any, List, Optional
from .schemas import ProjectSpec, Blueprint, Estimate, LineItem
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMCostEstimateSchema(BaseModel):
    """Schema for LLM-generated cost estimates."""
    total_monthly_cost: float = Field(description="Total monthly cost in USD")
    resource_breakdown: List[Dict[str, Any]] = Field(description="Detailed resource breakdown")
    cost_optimization_insights: List[str] = Field(description="Cost optimization insights")
    pricing_strategy: str = Field(description="Recommended pricing strategy")
    cost_forecast: Dict[str, Any] = Field(description="Cost forecasting and trends")
    confidence_score: float = Field(description="Confidence in the estimate (0-1)")

class IntelligentCostEngineAgent:
    """
    100% LLM-Powered Cost Engine - No External Data Dependencies
    
    The LLM generates:
    - Resource sizing based on workload requirements
    - Realistic pricing based on current market knowledge
    - Cost optimizations and strategies
    - Dynamic cost estimates that vary with each project
    """
    
    def __init__(self, openai_client: Optional[ChatOpenAI] = None):
        self._langchain_client = openai_client
        
    @property
    def client(self):
        """Lazy initialization of LangChain client."""
        if self._langchain_client is None:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set.")
            
            self._langchain_client = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,
                openai_api_key=api_key
            )
            logger.info("✅ Cost Engine LangChain client initialized successfully")
        return self._langchain_client
    
    def price_blueprint(self, spec: ProjectSpec, bp: Blueprint, sku_pricing=None) -> Estimate:
        """
        Generate 100% LLM-powered cost estimate.
        No external data, no hardcoded values, pure AI-driven estimation.
        """
        try:
            logger.info(f"🎯 Generating LLM-powered cost estimate for {bp.id}")
            
            # Get comprehensive LLM cost analysis
            llm_estimate = self._get_llm_cost_estimate(spec, bp)
            
            # Convert LLM output to Estimate object
            estimate = self._create_estimate_from_llm_output(llm_estimate, bp)
            
            logger.info(f"✅ LLM cost estimate generated: ${estimate.monthly_cost:,.2f}")
            return estimate
            
        except Exception as e:
            logger.error(f"LLM cost estimation failed: {e}")
            raise RuntimeError(f"LLM cost estimation failed - no fallback available: {e}")
    
    def _get_llm_cost_estimate(self, spec: ProjectSpec, bp: Blueprint) -> Dict[str, Any]:
        """Get comprehensive cost estimate from LLM without any external data."""
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert FinOps consultant and cloud architect with deep knowledge of:
            - Current cloud pricing across AWS, Azure, GCP (as of 2024)
            - Resource sizing for different workload types
            - Cost optimization strategies and best practices
            - Market trends and pricing patterns
            
            Your task: Generate a REALISTIC, detailed cost estimate for a cloud infrastructure project.
            Use your knowledge to provide accurate pricing and sizing - no external data needed.
            
            CRITICAL REQUIREMENTS:
            1. NEVER return $0.00 costs - all cloud resources have costs
            2. Use REALISTIC current market pricing (e.g., EC2 g5.2xlarge ~$0.50/hour = ~$360/month)
            3. Size resources based on actual workload requirements
            4. Make estimates vary SIGNIFICANTLY between different project types
            5. Include ALL necessary resources (compute, storage, network, monitoring)
            6. Provide detailed rationale for each resource choice
            
            IMPORTANT: Respond with ONLY valid JSON in the exact specified format."""),
            
            ("human", """Generate a comprehensive cost estimate for this project:

Project Requirements:
- Name: {project_name}
- Workload Type: {workload_type}
- GPU Count: {gpu_count}
- Inference QPS: {inference_qps}
- Data Size: {data_size_gb} GB
- Growth Rate: {growth_rate} GB/month
- Latency Requirement: {latency_ms}ms
- Batch Processing: {batch_processing}

Blueprint: {blueprint_details}

Generate a cost estimate in this JSON format:
{{
    "total_monthly_cost": 0.0,
    "resource_breakdown": [
        {{
            "service": "service_name",
            "sku": "specific_sku",
            "quantity": 1,
            "unit": "instance_month",
            "unit_price": 0.0,
            "monthly_cost": 0.0,
            "rationale": "why this resource and quantity"
        }}
    ],
    "cost_optimization_insights": [
        "insight1",
        "insight2"
    ],
    "pricing_strategy": "strategy_description",
    "cost_forecast": {{
        "trend": "increasing|decreasing|stable",
        "factors": ["factor1", "factor2"],
        "3_month_forecast": 0.0,
        "6_month_forecast": 0.0
    }},
    "confidence_score": 0.95
}}

PRICING GUIDELINES (2024 rates):
- EC2 g5.2xlarge (GPU): ~$360/month
- EC2 c6i.2xlarge (CPU): ~$288/month  
- EC2 m5.2xlarge (General): ~$288/month
- S3 Standard Storage: ~$0.023/GB/month
- RDS db.t3.medium: ~$30/month
- Redshift dc2.large: ~$180/month
- Load Balancer: ~$18/month
- CloudWatch: ~$15/month
- Data Transfer: ~$0.09/GB out

SIZING GUIDELINES:
- ML Training: 1 GPU instance per GPU requirement
- High Traffic: Scale compute based on QPS (1 instance per 500 QPS)
- Data Heavy: Scale storage based on data size + growth
- Always include monitoring, load balancing, and networking costs

Remember: NO $0.00 costs, be realistic, vary significantly between project types!""")
        ])
        
        # Create output parser
        parser = JsonOutputParser(pydantic_object=LLMCostEstimateSchema)
        
        # Create the chain
        chain = prompt_template | self.client | parser
        
        # Prepare inputs
        inputs = {
            "project_name": spec.name,
            "workload_type": self._classify_workload(spec),
            "gpu_count": spec.workload.train_gpus,
            "inference_qps": spec.workload.inference_qps,
            "data_size_gb": spec.data.size_gb,
            "growth_rate": spec.data.growth_gb_per_month,
            "latency_ms": spec.workload.latency_ms,
            "batch_processing": spec.workload.batch,
            "blueprint_details": f"Cloud: {bp.cloud}, Region: {bp.region}, Services: {[s['service'] for s in bp.services]}"
        }
        
        # Invoke the chain
        result = chain.invoke(inputs)
        logger.info("✅ LLM cost estimate generated successfully")
        return result
    
    def _classify_workload(self, spec: ProjectSpec) -> str:
        """Classify workload type for LLM context."""
        if spec.workload.train_gpus > 0:
            return f"ML Training ({spec.workload.train_gpus} GPUs)"
        elif spec.workload.inference_qps > 500:
            return "High-Traffic Inference"
        elif spec.data.size_gb > 5000:
            return "Data-Intensive Analytics"
        else:
            return "General Compute"
    
    def _create_estimate_from_llm_output(self, llm_output: Dict[str, Any], bp: Blueprint) -> Estimate:
        """Convert LLM output to Estimate object with validation."""
        
        # Get the total cost from LLM output
        total_cost = float(llm_output.get("total_monthly_cost", 0))
        
        # Create BOM from LLM resource breakdown
        bom = []
        for resource in llm_output.get("resource_breakdown", []):
            # Get individual resource costs from LLM
            resource_cost = float(resource.get("monthly_cost", 0))
            
            bom.append(LineItem(
                service=resource.get("service", "unknown"),
                sku=resource.get("sku", "unknown"),
                qty=float(resource.get("quantity", 1)),
                unit=resource.get("unit", "month"),
                unit_price=float(resource.get("unit_price", 0)),
                cost=resource_cost
            ))
        
        # Create estimate with LLM metadata
        estimate = Estimate(
            blueprint_id=bp.id,
            monthly_cost=round(total_cost, 2),
            bom=bom
        )
        
        # Store LLM insights in metadata
        estimate.metadata = {
            "llm_cost_analysis": "100%_llm_powered",
            "cost_optimization_insights": llm_output.get("cost_optimization_insights", []),
            "pricing_strategy": llm_output.get("pricing_strategy", ""),
            "cost_forecast": llm_output.get("cost_forecast", {}),
            "confidence_score": llm_output.get("confidence_score", 0.0),
            "resource_rationale": [r.get("rationale", "") for r in llm_output.get("resource_breakdown", [])]
        }
        
        return estimate
    
    def get_cost_analysis_report(self, spec: ProjectSpec, bp: Blueprint) -> Dict[str, Any]:
        """Get comprehensive cost analysis report from LLM."""
        try:
            llm_estimate = self._get_llm_cost_estimate(spec, bp)
            
            return {
                "total_cost": llm_estimate.get("total_monthly_cost", 0),
                "resource_breakdown": llm_estimate.get("resource_breakdown", []),
                "optimization_insights": llm_estimate.get("cost_optimization_insights", []),
                "pricing_strategy": llm_estimate.get("pricing_strategy", ""),
                "cost_forecast": llm_estimate.get("cost_forecast", {}),
                "confidence": llm_estimate.get("confidence_score", 0),
                "analysis_method": "100%_llm_powered"
            }
            
        except Exception as e:
            logger.error(f"LLM cost analysis failed: {e}")
            raise RuntimeError(f"LLM cost analysis failed - no fallback available: {e}")

# Backward compatibility
def price_blueprint(spec: ProjectSpec, bp: Blueprint, sku_pricing=None) -> Estimate:
    """Legacy function for backward compatibility."""
    agent = IntelligentCostEngineAgent()
    return agent.price_blueprint(spec, bp, sku_pricing)