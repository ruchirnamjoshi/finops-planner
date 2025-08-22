#!/usr/bin/env python3
"""
Dedicated Insights Agent for FinOps Planner
Generates comprehensive AI-powered insights from cost analysis
"""

import logging
from typing import Dict, Any, List, Optional
from .schemas import ProjectSpec, Estimate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveInsightsSchema(BaseModel):
    """Schema for comprehensive insights."""
    cost_analysis: List[str] = Field(description="Detailed cost analysis insights")
    resource_optimization: List[str] = Field(description="Resource optimization recommendations")
    cost_savings: List[str] = Field(description="Specific cost savings opportunities")
    risk_mitigation: List[str] = Field(description="Risk mitigation strategies")
    performance_insights: List[str] = Field(description="Performance and efficiency insights")
    strategic_recommendations: List[str] = Field(description="Strategic long-term recommendations")

class IntelligentInsightsAgent:
    """
    Dedicated agent for generating comprehensive AI-powered insights.
    Analyzes cost patterns, resource utilization, and provides actionable recommendations.
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
            logger.info("✅ Insights Agent LangChain client initialized successfully")
        return self._langchain_client
    
    def generate_comprehensive_insights(self, spec: ProjectSpec, estimate: Estimate, 
                                      cost_patterns: Dict[str, Any], 
                                      optimization_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive insights using LLM analysis.
        """
        try:
            logger.info("🎯 Generating comprehensive insights using LLM")
            
            # Determine workload type for more specific insights
            workload_type = self._classify_workload(spec)
            blueprint_context = self._get_blueprint_context(estimate)
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an expert FinOps consultant with deep expertise in:
                - Cloud cost optimization and management
                - Resource utilization analysis
                - Performance optimization strategies
                - Risk assessment and mitigation
                - Strategic cloud planning
                
                Your task: Analyze the project data and provide comprehensive, actionable insights.
                
                CRITICAL: Generate insights that are SPECIFIC to the workload type and blueprint.
                DO NOT provide generic advice - make it relevant to the specific architecture.
                
                IMPORTANT: Respond with ONLY valid JSON in the exact specified format."""),
                
                ("human", """Analyze this FinOps project and provide comprehensive insights:
       
       Project Specification:
       - Name: {project_name}
       - Workload Type: {workload_type}
       - GPU Count: {gpu_count}
       - Inference QPS: {inference_qps}
       - Data Size: {data_size_gb} GB
       - Growth Rate: {growth_rate} GB/month
       - Latency Requirement: {latency_ms}ms
       
       Cost Analysis:
       - Total Monthly Cost: ${total_cost}
       - Cost Breakdown: {cost_breakdown}
       
       Cost Patterns:
       - Compute Ratio: {compute_ratio}%
       - Storage Ratio: {storage_ratio}%
       - Network Ratio: {network_ratio}%
       - Dominant Driver: {dominant_driver}
       - Efficiency Score: {efficiency_score}/100
       
       Blueprint Context: {blueprint_context}
       
       Optimization Recommendations: {optimization_recs}
       
       WORKLOAD-SPECIFIC REQUIREMENTS:
       - If ML Training: Focus on GPU optimization, spot instances, training efficiency
       - If High-Traffic Web: Focus on auto-scaling, CDN, database optimization
       - If Data Warehouse: Focus on storage tiering, query optimization, ETL efficiency
       - If General Compute: Focus on resource sizing, utilization, cost management
       
       Provide insights in this JSON format:
       {{
           "cost_analysis": [
               "insight1",
               "insight2"
           ],
           "resource_optimization": [
               "recommendation1",
               "recommendation2"
           ],
           "cost_savings": [
               "opportunity1",
               "opportunity2"
           ],
           "risk_mitigation": [
               "strategy1",
               "strategy2"
           ],
           "performance_insights": [
               "insight1",
               "insight2"
           ],
           "strategic_recommendations": [
               "recommendation1",
               "recommendation2"
           ]
       }}
       
       Focus on actionable, specific insights that provide real value for cost optimization.
       Make insights relevant to the specific workload type and architecture."""),
            ])
            
            # Create output parser
            parser = JsonOutputParser(pydantic_object=ComprehensiveInsightsSchema)
            
            # Create the chain
            chain = prompt_template | self.client | parser
            
            # Prepare inputs
            inputs = {
                "project_name": spec.name,
                "workload_type": workload_type,
                "gpu_count": spec.workload.train_gpus,
                "inference_qps": spec.workload.inference_qps,
                "data_size_gb": spec.data.size_gb,
                "growth_rate": spec.data.growth_gb_per_month,
                "latency_ms": spec.workload.latency_ms,
                "total_cost": estimate.monthly_cost,
                "cost_breakdown": [li.model_dump() for li in estimate.bom],
                "compute_ratio": cost_patterns.get("compute_ratio", 0),
                "storage_ratio": cost_patterns.get("storage_ratio", 0),
                "network_ratio": cost_patterns.get("network_ratio", 0),
                "dominant_driver": cost_patterns.get("dominant_cost_driver", "unknown"),
                "efficiency_score": cost_patterns.get("cost_efficiency_score", 0),
                "blueprint_context": blueprint_context,
                "optimization_recs": optimization_recommendations
            }
            
            # Invoke the chain
            result = chain.invoke(inputs)
            logger.info("✅ Comprehensive insights generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"LLM insights generation failed: {e}")
            raise RuntimeError(f"LLM insights generation failed - no fallback available: {e}")
    
    def _classify_workload(self, spec: ProjectSpec) -> str:
        """Classify workload type for insights context."""
        if spec.workload.train_gpus > 0:
            return f"ML Training ({spec.workload.train_gpus} GPUs)"
        elif spec.workload.inference_qps > 500:
            return "High-Traffic Inference"
        elif spec.data.size_gb > 5000:
            return "Data-Intensive Analytics"
        else:
            return "General Compute"
    
    def _get_blueprint_context(self, estimate: Estimate) -> str:
        """Extract blueprint context from the estimate."""
        try:
            if hasattr(estimate, 'bom') and estimate.bom:
                services = [item.service for item in estimate.bom]
                if 'ec2' in services and 'gpu' in str(estimate.metadata).lower():
                    return "GPU Training Infrastructure (EC2 + GPU instances)"
                elif 'rds' in services and 'cloudfront' in services:
                    return "Web Application (EC2 + RDS + CDN)"
                elif 'redshift' in services or 'emr' in services:
                    return "Data Warehouse (Redshift/EMR + S3)"
                else:
                    return f"General Infrastructure ({', '.join(services[:3])})"
            return "Standard Cloud Infrastructure"
        except Exception:
            return "Cloud Infrastructure"
    
    def generate_cost_summary_insights(self, estimate: Estimate, cost_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate quick cost summary insights for display.
        """
        try:
            # Calculate quick insights from cost patterns
            insights = {
                "cost_distribution": {
                    "compute": f"{cost_patterns.get('compute_ratio', 0):.1f}%",
                    "storage": f"{cost_patterns.get('storage_ratio', 0):.1f}%",
                    "network": f"{cost_patterns.get('network_ratio', 0):.1f}%"
                },
                "key_findings": [],
                "immediate_actions": []
            }
            
            # Generate key findings
            if cost_patterns.get("high_compute", False):
                insights["key_findings"].append("🔥 Compute costs dominate the budget")
                insights["immediate_actions"].append("Consider spot instances for interruptible workloads")
            
            if cost_patterns.get("high_storage", False):
                insights["key_findings"].append("💾 Storage costs are significant")
                insights["immediate_actions"].append("Implement storage lifecycle policies")
            
            if cost_patterns.get("high_network", False):
                insights["key_findings"].append("🌐 Network costs are high")
                insights["immediate_actions"].append("Optimize data transfer patterns")
            
            # Add efficiency insights
            efficiency = cost_patterns.get("cost_efficiency_score", 0)
            if efficiency < 50:
                insights["key_findings"].append("❌ Cost efficiency is poor")
                insights["immediate_actions"].append("Review resource sizing and utilization")
            elif efficiency < 80:
                insights["key_findings"].append("⚠️ Cost efficiency has room for improvement")
                insights["immediate_actions"].append("Implement recommended optimizations")
            else:
                insights["key_findings"].append("✅ Cost efficiency is good")
                insights["immediate_actions"].append("Monitor for optimization opportunities")
            
            return insights
            
        except Exception as e:
            logger.error(f"Cost summary insights generation failed: {e}")
            # No fallbacks - raise error
            raise RuntimeError(f"Cost summary insights generation failed - no fallback available: {e}")

# Backward compatibility
def generate_comprehensive_insights(spec: ProjectSpec, estimate: Estimate, 
                                  cost_patterns: Dict[str, Any], 
                                  optimization_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Legacy function for backward compatibility."""
    agent = IntelligentInsightsAgent()
    return agent.generate_comprehensive_insights(spec, estimate, cost_patterns, optimization_recommendations)
