try:
    from types import SimpleNamespace
except ImportError:
    # Fallback for older Python versions
    class SimpleNamespace:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json
from .schemas import ProjectSpec, Blueprint, Estimate, LineItem
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
class CostInsightsSchema(BaseModel):
    """Schema for cost analysis insights."""
    cost_breakdown_analysis: Dict[str, Any] = Field(description="Cost breakdown analysis")
    optimization_opportunities: List[Dict[str, Any]] = Field(description="List of optimization opportunities")
    pricing_insights: List[Dict[str, Any]] = Field(description="Pricing insights and observations")
    cost_forecast: Dict[str, Any] = Field(description="Cost forecasting and trends")

class IntelligentCostEngineAgent:
    """
    LLM-powered agent that provides sophisticated cost analysis and pricing insights.
    Capabilities:
    - Advanced cost modeling and analysis
    - Pricing trend analysis and forecasting
    - Cost optimization recommendations
    - Multi-cloud cost comparison
    - Resource utilization analysis
    - Cost anomaly detection
    """
    
    def __init__(self, openai_client: Optional[ChatOpenAI] = None):
        self._langchain_client = openai_client
        self.pricing_insights = self._load_pricing_insights()
        self.cost_models = self._load_cost_models()
        
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
            print("✅ Cost Engine LangChain client initialized successfully with API key")
        return self._langchain_client
    
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
            "inference_serving": {
                "compute_ratio": 0.6,
                "storage_ratio": 0.1,
                "network_ratio": 0.3,
                "optimization_potential": 0.3,
                "key_factors": ["auto_scaling", "load_balancing", "caching"]
            },
            "data_warehouse": {
                "compute_ratio": 0.4,
                "storage_ratio": 0.5,
                "network_ratio": 0.1,
                "optimization_potential": 0.5,
                "key_factors": ["storage_tiering", "compression", "partitioning"]
            }
        }
    
    def _safe_eval(self, expr, ctx):
        """Safely evaluate expressions with context."""
        if isinstance(expr, (int, float)):
            return float(expr)
        try:
            return float(eval(expr, {"__builtins__": {}}, ctx))
        except Exception as e:
            logger.warning(f"Failed to evaluate expression '{expr}': {e}")
            return 0.0
    
    def _analyze_pricing_anomalies(self, sku_pricing: pd.DataFrame, bp: Blueprint) -> List[Dict[str, Any]]:
        """Analyze pricing for potential anomalies or optimization opportunities."""
        anomalies = []
        
        # Get pricing for this blueprint's region and cloud
        region_pricing = sku_pricing[
            (sku_pricing["cloud"] == bp.cloud) & 
            (sku_pricing["region"] == bp.region)
        ]
        
        if region_pricing.empty:
            return anomalies
        
        # Check for unusually high prices
        for _, row in region_pricing.iterrows():
            sku = row["sku"]
            unit_price = row["unit_price"]
            
            # Compare with other regions in the same cloud
            cloud_pricing = sku_pricing[sku_pricing["cloud"] == bp.cloud]
            if len(cloud_pricing) > 1:
                avg_price = cloud_pricing["unit_price"].mean()
                price_ratio = unit_price / avg_price
                
                if price_ratio > 1.5:  # 50% more expensive than average
                    anomalies.append({
                        "type": "high_price",
                        "sku": sku,
                        "current_price": unit_price,
                        "average_price": avg_price,
                        "price_ratio": price_ratio,
                        "recommendation": f"Consider using {sku} in other regions or alternative services"
                    })
        
        return anomalies
    
    def _get_llm_cost_insights(self, spec: ProjectSpec, bp: Blueprint, estimate: Estimate,
                               sku_pricing: pd.DataFrame) -> Dict[str, Any]:
        """Get LLM-powered cost insights and recommendations using LangChain."""
        
        try:
            # Create LangChain prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an expert FinOps consultant specializing in cloud cost optimization.
                Analyze the cost estimate and provide detailed insights and recommendations.
                
                IMPORTANT: You must respond with ONLY valid JSON in exactly the specified format.
                Do not include any text before or after the JSON. Return ONLY the JSON object."""),
                ("human", """Analyze this cost estimate and provide optimization insights:

Project Specification: {spec}
Blueprint: {blueprint}
Cost Estimate: {estimate}
Pricing Data: {pricing}

Provide cost analysis in this JSON format:
{{
    "cost_breakdown_analysis": {{
        "compute_percentage": "percentage_of_total",
        "storage_percentage": "percentage_of_total",
        "network_percentage": "percentage_of_total",
        "dominant_cost_driver": "main_cost_component"
    }},
    "optimization_opportunities": [
        {{
            "category": "compute|storage|network|overall",
            "description": "detailed_description",
            "potential_savings": "estimated_savings_percentage",
            "effort_required": "low|medium|high",
            "implementation_steps": ["step1", "step2", "step3"]
        }}
    ],
    "pricing_insights": [
        {{
            "insight": "pricing_observation",
            "impact": "cost_impact_description",
            "action": "recommended_action"
        }}
    ],
    "cost_forecast": {{
        "trend": "increasing|decreasing|stable",
        "factors": ["factor1", "factor2"],
        "recommendations": ["recommendation1", "recommendation2"]
    }}
}}

Focus on actionable insights and realistic cost optimization opportunities.""")
            ])
            
            # Create output parser
            parser = JsonOutputParser(pydantic_object=CostInsightsSchema)
            
            # Create the chain
            chain = prompt_template | self.client | parser
            
            # Prepare inputs
            inputs = {
                "spec": spec.model_dump(),
                "blueprint": bp.model_dump(),
                "estimate": estimate.model_dump(),
                "pricing": sku_pricing.to_dict('records')
            }
            
            # Invoke the chain
            result = chain.invoke(inputs)
            logger.info("✅ LangChain cost insights generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"LangChain cost insights failed: {e}")
            raise RuntimeError(f"LLM cost insights generation failed: {e}")
    
    def _calculate_cost_efficiency_score(self, estimate: Estimate, spec: ProjectSpec) -> float:
        """Calculate a cost efficiency score based on multiple factors."""
        if estimate.monthly_cost == 0:
            return 0.0
        
        score = 100.0
        
        # Deduct points for high egress costs
        egress_cost = sum(li.cost for li in estimate.bom if "egress" in li.sku.lower())
        egress_ratio = egress_cost / estimate.monthly_cost
        if egress_ratio > 0.4:
            score -= 20
        
        # Deduct points for over-provisioning
        if spec.workload.train_gpus > 0:
            # Assume reasonable GPU utilization
            expected_gpu_cost = spec.workload.train_gpus * 2.0  # Rough estimate
            actual_gpu_cost = sum(li.cost for li in estimate.bom if "gpu" in li.sku.lower())
            if actual_gpu_cost > expected_gpu_cost * 1.5:
                score -= 15
        
        # Bonus points for cost optimization features
        if any("spot" in li.sku.lower() for li in estimate.bom):
            score += 10
        
        return max(0.0, min(100.0, score))
    
    def price_blueprint(self, spec: ProjectSpec, bp: Blueprint, sku_pricing: pd.DataFrame) -> Estimate:
        """Generate cost estimate with LLM-enhanced analysis and dynamic pricing."""
        # Ensure numeric pricing
        if sku_pricing["unit_price"].dtype.kind not in "fi":
            sku_pricing = sku_pricing.copy()
            sku_pricing["unit_price"] = pd.to_numeric(sku_pricing["unit_price"], errors="coerce").fillna(0.0)

        # Create context for expression evaluation
        ctx = {
            "train_gpus": spec.workload.train_gpus,
            "data": SimpleNamespace(
                size_gb=spec.data.size_gb,
                hot_fraction=spec.data.hot_fraction,
                egress_gb_per_month=spec.data.egress_gb_per_month,
            ),
        }

        # Generate base cost estimate using static formulas
        bom: List[LineItem] = []
        total = 0.0
        
        for svc in bp.services:
            qty = self._safe_eval(svc["qty_expr"], ctx)

            # Find matching pricing
            row = sku_pricing[
                (sku_pricing["cloud"] == bp.cloud) &
                (sku_pricing["region"] == bp.region) &
                (sku_pricing["sku"] == svc["sku"])
            ]

            unit_price = float(row.iloc[0]["unit_price"]) if not row.empty else 0.0
            cost = qty * unit_price

            bom.append(LineItem(
                service=svc["service"],
                sku=svc["sku"],
                qty=float(qty),
                unit=svc["unit"],
                unit_price=unit_price,
                cost=float(cost),
            ))
            total += cost

        # Get LLM-powered cost analysis and dynamic adjustments
        llm_insights = None
        cost_analysis_method = "static_pricing"
        llm_cost_optimization = ""
        llm_cost_forecast = ""
        
        try:
            # Create a temporary estimate for LLM analysis
            temp_estimate = Estimate(
                blueprint_id=bp.id,
                monthly_cost=total,
                bom=bom
            )
            llm_insights = self._get_llm_cost_insights(spec, bp, temp_estimate, sku_pricing)
            
            # Apply LLM-suggested cost adjustments
            if llm_insights and "cost_forecast" in llm_insights:
                forecast = llm_insights["cost_forecast"]
                
                # Apply dynamic cost adjustments based on LLM insights
                if forecast.get("trend") == "decreasing":
                    # LLM suggests cost reduction opportunities
                    cost_reduction = 0.05  # 5% base reduction
                    
                    # Look for specific optimization opportunities
                    if "optimization_opportunities" in llm_insights:
                        for opp in llm_insights["optimization_opportunities"]:
                            if opp.get("potential_savings"):
                                try:
                                    savings_pct = float(str(opp["potential_savings"]).replace("%", "")) / 100
                                    cost_reduction = max(cost_reduction, savings_pct * 0.5)  # Apply 50% of suggested savings
                                except:
                                    pass
                    
                    total = total * (1 - cost_reduction)
                    llm_cost_optimization = f"Applied {cost_reduction*100:.1f}% cost reduction based on LLM insights"
                
                elif forecast.get("trend") == "increasing":
                    # LLM suggests cost growth factors
                    growth_factor = 1.05  # 5% base growth
                    
                    # Look for specific growth factors
                    if "factors" in forecast:
                        for factor in forecast["factors"]:
                            if "data growth" in factor.lower():
                                growth_factor = 1.10  # 10% growth for data-heavy workloads
                            elif "traffic" in factor.lower():
                                growth_factor = 1.15  # 15% growth for high-traffic workloads
                    
                    total = total * growth_factor
                    llm_cost_forecast = f"Applied {growth_factor*100-100:.1f}% growth factor based on LLM insights"
                
                            # Apply workload-specific cost adjustments based on LLM insights
            if "cost_breakdown_analysis" in llm_insights:
                breakdown = llm_insights["cost_breakdown_analysis"]
                
                # Apply dynamic resource allocation based on LLM insights
                if spec.workload.train_gpus > 0:  # ML Training workload
                    # ML workloads need more compute and storage
                    ml_compute_factor = 1.5 + (spec.workload.train_gpus * 0.3)  # Scale with GPU count
                    ml_storage_factor = 1.2 + (spec.data.size_gb / 1000)  # Scale with data size
                    
                    total = total * ml_compute_factor
                    llm_cost_optimization += f" + ML workload scaling (GPUs: {spec.workload.train_gpus}, Data: {spec.data.size_gb}GB)"
                    
                    # Apply GPU-specific optimizations
                    if "gpu" in breakdown.get("dominant_cost_driver", "").lower():
                        gpu_optimization = 0.15  # 15% reduction for ML workloads
                        total = total * (1 - gpu_optimization)
                        llm_cost_optimization += f" + {gpu_optimization*100:.1f}% ML workload optimization"
                
                elif spec.workload.inference_qps > 500:  # High-traffic workload
                    # High-traffic workloads need more compute and network
                    traffic_factor = 1.0 + (spec.workload.inference_qps / 1000)  # Scale with QPS
                    network_factor = 1.0 + (spec.data.egress_gb_per_month / 100)  # Scale with egress
                    
                    total = total * traffic_factor * network_factor
                    llm_cost_forecast += f" + High-traffic scaling (QPS: {spec.workload.inference_qps}, Egress: {spec.data.egress_gb_per_month}GB)"
                    
                    # Apply traffic-based adjustments
                    if "compute" in breakdown.get("dominant_cost_driver", "").lower():
                        traffic_adjustment = 1.20  # 20% increase for high-traffic
                        total = total * traffic_adjustment
                        llm_cost_forecast += f" + {traffic_adjustment*100-100:.1f}% high-traffic adjustment"
                
                elif spec.data.size_gb > 5000:  # Data-heavy workload
                    # Data-heavy workloads need more storage and compute for processing
                    data_factor = 1.0 + (spec.data.size_gb / 10000)  # Scale with data size
                    growth_factor = 1.0 + (spec.data.growth_gb_per_month / 100)  # Scale with growth rate
                    
                    total = total * data_factor * growth_factor
                    llm_cost_forecast += f" + Data-heavy scaling (Size: {spec.data.size_gb}GB, Growth: {spec.data.growth_gb_per_month}GB/month)"
                    
                    # Apply data lifecycle optimizations
                    if "storage" in breakdown.get("dominant_cost_driver", "").lower():
                        storage_optimization = 0.25  # 25% reduction for data-heavy workloads
                        total = total * (1 - storage_optimization)
                        llm_cost_optimization += f" + {storage_optimization*100:.1f}% data lifecycle optimization"
                
                # Apply general LLM-suggested optimizations
                if "optimization_opportunities" in llm_insights:
                    for opp in llm_insights["optimization_opportunities"]:
                        if opp.get("category") == "storage" and opp.get("potential_savings"):
                            try:
                                savings_pct = float(str(opp["potential_savings"]).replace("%", "")) / 100
                                storage_savings = min(savings_pct * 0.7, 0.3)  # Apply up to 30% storage savings
                                total = total * (1 - storage_savings)
                                llm_cost_optimization += f" + {storage_savings*100:.1f}% storage optimization"
                            except:
                                pass
            
            cost_analysis_method = "llm_enhanced"
            
        except Exception as e:
            logger.error(f"LLM insights failed: {e}")
            llm_insights = None

        # Create final estimate with metadata
        estimate = Estimate(
            blueprint_id=bp.id, 
            monthly_cost=round(total, 2), 
            bom=bom
        )
        
        # Add comprehensive cost analysis metadata
        estimate.metadata.update({
            "cost_efficiency_score": self._calculate_cost_efficiency_score(estimate, spec),
            "pricing_anomalies": self._analyze_pricing_anomalies(sku_pricing, bp),
            "llm_insights": llm_insights,
            "cost_analysis_method": cost_analysis_method,
            "llm_cost_optimization": llm_cost_optimization,
            "llm_cost_forecast": llm_cost_forecast,
            "cost_breakdown": {
                "compute": sum(li.cost for li in bom if li.service in ["compute", "ec2", "vm"]),
                "storage": sum(li.cost for li in bom if li.service in ["storage", "s3", "blob"]),
                "network": sum(li.cost for li in bom if li.service not in ["compute", "ec2", "vm", "storage", "s3", "blob"])
            }
        })
        
        return estimate
    
    def get_cost_analysis_report(self, spec: ProjectSpec, bp: Blueprint, estimate: Estimate,
                                sku_pricing: pd.DataFrame) -> Dict[str, Any]:
        """Generate a comprehensive cost analysis report."""
        return {
            "blueprint_id": bp.id,
            "total_monthly_cost": estimate.monthly_cost,
            "cost_efficiency_score": estimate.metadata["cost_efficiency_score"],
            "cost_breakdown": estimate.metadata["cost_breakdown"],
            "pricing_anomalies": estimate.metadata["pricing_anomalies"],
            "llm_insights": estimate.metadata["llm_insights"],
            "optimization_recommendations": self._generate_optimization_recommendations(estimate),
            "cost_trends": self._analyze_cost_trends(estimate, spec),
            "resource_utilization": self._analyze_resource_utilization(estimate, spec)
        }
    
    def _generate_optimization_recommendations(self, estimate: Estimate) -> List[Dict[str, Any]]:
        """Generate specific optimization recommendations based on cost analysis."""
        recommendations = []
        
        breakdown = estimate.metadata["cost_breakdown"]
        total = estimate.monthly_cost
        
        # High compute costs
        if breakdown["compute"] / total > 0.7:
            recommendations.append({
                "category": "compute",
                "priority": "high",
                "description": "Compute costs dominate the budget",
                "actions": [
                    "Consider spot instances for interruptible workloads",
                    "Purchase reserved instances for predictable usage",
                    "Implement auto-scaling to optimize utilization"
                ]
            })
        
        # High storage costs
        if breakdown["storage"] / total > 0.5:
            recommendations.append({
                "category": "storage",
                "priority": "medium",
                "description": "Storage costs are significant",
                "actions": [
                    "Implement storage lifecycle policies",
                    "Use appropriate storage tiers",
                    "Enable compression where possible"
                ]
            })
        
        # High network costs
        if breakdown["network"] / total > 0.4:
            recommendations.append({
                "category": "network",
                "priority": "high",
                "description": "Network costs are high",
                "actions": [
                    "Co-locate compute and storage",
                    "Use CDN for static content",
                    "Optimize data transfer patterns"
                ]
            })
        
        return recommendations
    
    def _analyze_cost_trends(self, estimate: Estimate, spec: ProjectSpec) -> Dict[str, Any]:
        """Analyze potential cost trends based on project characteristics."""
        trends = {
            "direction": "stable",
            "factors": [],
            "projections": {}
        }
        
        # Data growth impact
        if spec.data.growth_gb_per_month > 0:
            trends["factors"].append(f"Data growing at {spec.data.growth_gb_per_month} GB/month")
            trends["projections"]["storage_growth"] = f"Storage costs may increase by {spec.data.growth_gb_per_month * 0.02:.2f}/month"
        
        # Egress growth impact
        if spec.data.egress_gb_per_month > 0:
            trends["factors"].append(f"Monthly egress of {spec.data.egress_gb_per_month} GB")
            trends["projections"]["egress_growth"] = f"Network costs may increase with data growth"
        
        # Workload scaling impact
        if spec.workload.inference_qps > 1000:
            trends["factors"].append("High inference QPS requirement")
            trends["projections"]["compute_scaling"] = "Compute costs may increase with traffic growth"
        
        return trends
    
    def _analyze_resource_utilization(self, estimate: Estimate, spec: ProjectSpec) -> Dict[str, Any]:
        """Analyze resource utilization patterns and efficiency."""
        utilization = {
            "gpu_efficiency": "unknown",
            "storage_efficiency": "unknown",
            "network_efficiency": "unknown",
            "recommendations": []
        }
        
        # GPU utilization analysis
        if spec.workload.train_gpus > 0:
            gpu_cost = sum(li.cost for li in estimate.bom if "gpu" in li.sku.lower())
            if gpu_cost > 0:
                utilization["gpu_efficiency"] = "high" if gpu_cost < spec.workload.train_gpus * 3.0 else "medium"
                if utilization["gpu_efficiency"] == "medium":
                    utilization["recommendations"].append("Consider optimizing GPU instance selection")
        
        # Storage efficiency analysis
        if spec.data.size_gb > 1000:
            storage_cost = estimate.metadata["cost_breakdown"]["storage"]
            if storage_cost > spec.data.size_gb * 0.01:  # More than $0.01 per GB
                utilization["storage_efficiency"] = "low"
                utilization["recommendations"].append("Consider implementing storage lifecycle policies")
            else:
                utilization["storage_efficiency"] = "high"
        
        return utilization


# Backward compatibility function
def price_blueprint(spec: ProjectSpec, bp: Blueprint, sku_pricing: pd.DataFrame) -> Estimate:
    """Legacy function for backward compatibility."""
    agent = IntelligentCostEngineAgent()
    return agent.price_blueprint(spec, bp, sku_pricing)