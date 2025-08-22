"""
Intelligent Strategy Comparison Agent for FinOps Planner

This agent provides comprehensive comparison and recommendation analysis
for multiple architecture blueprints, helping users make informed decisions.
"""

import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

# Local imports
from .schemas import ProjectSpec, Estimate, Blueprint
from .langchain_base import LangChainBaseAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyComparisonSchema(BaseModel):
    """Schema for strategy comparison results."""
    winner_blueprint_id: str = Field(description="ID of the recommended blueprint")
    winner_reason: str = Field(description="Detailed explanation of why this blueprint was chosen")
    comparison_matrix: Dict[str, Dict[str, Any]] = Field(description="Detailed comparison of all blueprints")
    ranking: List[str] = Field(description="Ranked list of blueprint IDs from best to worst")
    cost_analysis: Dict[str, Any] = Field(description="Cost comparison and analysis")
    performance_analysis: Dict[str, Any] = Field(description="Performance comparison and analysis")
    risk_analysis: Dict[str, Any] = Field(description="Risk comparison and analysis")
    strategic_recommendations: List[str] = Field(description="Strategic recommendations for the chosen blueprint")
    implementation_roadmap: Dict[str, Any] = Field(description="Implementation roadmap and next steps")

class IntelligentStrategyComparisonAgent(LangChainBaseAgent):
    """
    LLM-powered agent for comprehensive strategy comparison and recommendation.
    
    This agent analyzes multiple blueprints considering:
    - Cost efficiency and optimization potential
    - Performance characteristics and scalability
    - Risk factors and mitigation strategies
    - Strategic alignment with business goals
    - Implementation complexity and timeline
    """
    
    def __init__(self):
        """Initialize the Strategy Comparison Agent."""
        super().__init__(model_name="gpt-4o-mini", temperature=0.1)
        self.agent_name = "Strategy Comparison Agent"
    
    def compare_strategies(
        self, 
        spec: ProjectSpec, 
        candidates: List[Tuple[Blueprint, Estimate]], 
        optimizations: List[Any], 
        risks: Dict[str, Any],
        cost_patterns: Dict[str, Any] = None
    ) -> StrategyComparisonSchema:
        """
        Comprehensive strategy comparison and recommendation.
        
        Args:
            spec: Project specification
            candidates: List of (blueprint, estimate) tuples
            optimizations: List of optimization results
            risks: Dictionary of risk assessments by blueprint ID
            cost_patterns: Optional cost pattern analysis
            
        Returns:
            StrategyComparisonSchema with comprehensive comparison results
        """
        try:
            logger.info(f"🎯 Starting comprehensive strategy comparison for {len(candidates)} blueprints")
            
            # Prepare data for LLM analysis
            comparison_data = self._prepare_comparison_data(
                spec, candidates, optimizations, risks, cost_patterns
            )
            
            # Generate comprehensive comparison using LLM
            comparison_result = self._generate_llm_comparison(comparison_data)
            
            logger.info("✅ Strategy comparison completed successfully")
            return comparison_result
            
        except Exception as e:
            logger.error(f"❌ Strategy comparison failed: {e}")
            raise RuntimeError(f"Strategy comparison failed: {e}")
    
    def _prepare_comparison_data(
        self, 
        spec: ProjectSpec, 
        candidates: List[Tuple[Blueprint, Estimate]], 
        optimizations: List[Any], 
        risks: Dict[str, Any],
        cost_patterns: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Prepare structured data for LLM comparison analysis."""
        
        comparison_data = {
            "project_spec": {
                "name": spec.name,
                "workload_type": self._classify_workload(spec),
                "data_size_gb": spec.data.size_gb,
                "performance_requirements": {
                    "inference_qps": spec.workload.inference_qps,
                    "latency_ms": spec.workload.latency_ms,
                    "train_gpus": spec.workload.train_gpus
                },
                "constraints": {
                    "clouds": spec.constraints.clouds,
                    "regions": spec.constraints.regions
                }
            },
            "blueprints": [],
            "cost_patterns": cost_patterns or {}
        }
        
        # Process each blueprint with its data
        for i, (blueprint, estimate) in enumerate(candidates):
            if estimate is None:
                continue
                
            # Get optimization data for this blueprint
            optimization = optimizations[i] if i < len(optimizations) else None
            risk_assessment = risks.get(blueprint.id, [])
            
            blueprint_data = {
                "id": blueprint.id,
                "name": blueprint.id.upper(),
                "cloud": blueprint.cloud,
                "region": blueprint.region,
                "services": blueprint.services,
                "monthly_cost": estimate.monthly_cost,
                "cost_breakdown": self._extract_cost_breakdown(estimate),
                "optimization": self._extract_optimization_data(optimization),
                "risks": self._extract_risk_data(risk_assessment),
                "metadata": estimate.metadata
            }
            
            comparison_data["blueprints"].append(blueprint_data)
        
        return comparison_data
    
    def _classify_workload(self, spec: ProjectSpec) -> str:
        """Classify the workload type for context."""
        if spec.workload.train_gpus > 0:
            return "ML Training"
        elif spec.workload.inference_qps > 500:
            return "High-Traffic Inference"
        elif spec.data.size_gb > 5000:
            return "Data-Intensive"
        else:
            return "General Compute"
    
    def _extract_cost_breakdown(self, estimate: Estimate) -> Dict[str, Any]:
        """Extract cost breakdown information from estimate."""
        if not hasattr(estimate, 'bom') or not estimate.bom:
            return {}
        
        total_cost = sum(item.cost for item in estimate.bom)
        breakdown = {}
        
        for item in estimate.bom:
            service = item.service
            cost = item.cost
            percentage = (cost / total_cost * 100) if total_cost > 0 else 0
            breakdown[service] = {
                "cost": cost,
                "percentage": percentage
            }
        
        return breakdown
    
    def _extract_optimization_data(self, optimization: Any) -> Dict[str, Any]:
        """Extract optimization data from optimization result."""
        if not optimization:
            return {}
        
        try:
            if hasattr(optimization, 'metadata'):
                return {
                    "optimization_score": getattr(optimization, 'optimization_score', 0),
                    "savings_potential": getattr(optimization, 'savings_potential', 0),
                    "recommendations": optimization.metadata.get('optimization_recommendations', [])
                }
            elif hasattr(optimization, 'get'):
                return {
                    "optimization_score": optimization.get('optimization_score', 0),
                    "savings_potential": optimization.get('savings_potential', 0),
                    "recommendations": optimization.get('recommendations', [])
                }
        except Exception as e:
            logger.warning(f"Failed to extract optimization data: {e}")
        
        return {}
    
    def _extract_risk_data(self, risks: List[Any]) -> Dict[str, Any]:
        """Extract risk assessment data."""
        if not risks:
            return {}
        
        try:
            risk_summary = {
                "total_risks": len(risks),
                "critical_risks": 0,
                "high_risks": 0,
                "medium_risks": 0,
                "low_risks": 0,
                "risk_categories": set()
            }
            
            for risk in risks:
                if hasattr(risk, 'severity'):
                    severity = risk.severity
                elif hasattr(risk, 'get'):
                    severity = risk.get('severity', 'medium')
                else:
                    severity = 'medium'
                
                if severity == 'critical':
                    risk_summary["critical_risks"] += 1
                elif severity == 'high':
                    risk_summary["high_risks"] += 1
                elif severity == 'medium':
                    risk_summary["medium_risks"] += 1
                else:
                    risk_summary["low_risks"] += 1
                
                if hasattr(risk, 'category'):
                    risk_summary["risk_categories"].add(risk.category)
                elif hasattr(risk, 'get'):
                    risk_summary["risk_categories"].add(risk.get('category', 'unknown'))
            
            risk_summary["risk_categories"] = list(risk_summary["risk_categories"])
            return risk_summary
            
        except Exception as e:
            logger.warning(f"Failed to extract risk data: {e}")
            return {}
    
    def _generate_llm_comparison(self, comparison_data: Dict[str, Any]) -> StrategyComparisonSchema:
        """Generate comprehensive comparison using LLM."""
        
        # Create the comparison prompt
        system_prompt = """You are an expert FinOps consultant and cloud architect with deep expertise in evaluating cloud infrastructure strategies.

Your task is to comprehensively analyze multiple architecture blueprints and provide an intelligent recommendation for the best strategy.

CRITICAL REQUIREMENTS:
1. Consider cost, performance, risk, scalability, and strategic alignment
2. Provide detailed reasoning for your recommendation
3. Consider the specific workload type and business requirements
4. Evaluate implementation complexity and timeline
5. Provide actionable strategic recommendations

You must analyze:
- Cost efficiency and optimization potential
- Performance characteristics and scalability
- Risk factors and mitigation strategies
- Strategic alignment with business goals
- Implementation complexity and timeline"""

        human_prompt = """Analyze these {num_blueprints} architecture blueprints for the project "{project_name}" and provide a comprehensive comparison.

PROJECT CONTEXT:
- Workload Type: {workload_type}
- Data Size: {data_size_gb} GB
- Performance Requirements: {performance_reqs}
- Cloud Constraints: {cloud_constraints}

BLUEPRINT DETAILS:
{blueprint_details}

COST PATTERNS (if available):
{cost_patterns}

Provide a comprehensive analysis in this exact JSON format:
{{
    "winner_blueprint_id": "id_of_best_blueprint",
    "winner_reason": "Detailed explanation of why this blueprint was chosen, considering all factors",
    "comparison_matrix": {{
        "blueprint_id": {{
            "cost_score": 0-100,
            "performance_score": 0-100,
            "risk_score": 0-100,
            "scalability_score": 0-100,
            "implementation_score": 0-100,
            "overall_score": 0-100,
            "strengths": ["list of key strengths"],
            "weaknesses": ["list of key weaknesses"],
            "recommendations": ["specific improvement recommendations"]
        }}
    }},
    "ranking": ["blueprint_id_1", "blueprint_id_2", "blueprint_id_3"],
    "cost_analysis": {{
        "cost_efficiency": "analysis of cost efficiency across blueprints",
        "optimization_potential": "analysis of cost optimization opportunities",
        "long_term_cost_trends": "prediction of cost trends over time"
    }},
    "performance_analysis": {{
        "scalability": "analysis of scalability characteristics",
        "latency_analysis": "analysis of latency and performance",
        "resource_utilization": "analysis of resource efficiency"
    }},
    "risk_analysis": {{
        "overall_risk_assessment": "comprehensive risk assessment",
        "risk_mitigation_strategies": ["specific risk mitigation approaches"],
        "compliance_considerations": "compliance and governance factors"
    }},
    "strategic_recommendations": [
        "Specific strategic recommendation 1",
        "Specific strategic recommendation 2",
        "Specific strategic recommendation 3"
    ],
    "implementation_roadmap": {{
        "phase_1": "immediate actions (0-30 days)",
        "phase_2": "short-term improvements (1-3 months)",
        "phase_3": "long-term optimization (3-12 months)",
        "key_milestones": ["milestone 1", "milestone 2", "milestone 3"],
        "resource_requirements": "human and technical resources needed"
    }}
}}

Focus on providing actionable insights and specific recommendations. Consider the business context and long-term strategic value, not just immediate costs."""

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        # Create the parser
        parser = JsonOutputParser(pydantic_object=StrategyComparisonSchema)
        
        # Create the chain
        chain = prompt | self.llm | parser
        
        # Prepare the inputs
        inputs = {
            "num_blueprints": len(comparison_data["blueprints"]),
            "project_name": comparison_data["project_spec"]["name"],
            "workload_type": comparison_data["project_spec"]["workload_type"],
            "data_size_gb": comparison_data["project_spec"]["data_size_gb"],
            "performance_reqs": f"QPS: {comparison_data['project_spec']['performance_requirements']['inference_qps']}, Latency: {comparison_data['project_spec']['performance_requirements']['latency_ms']}ms, GPUs: {comparison_data['project_spec']['performance_requirements']['train_gpus']}",
            "cloud_constraints": f"Clouds: {', '.join(comparison_data['project_spec']['constraints']['clouds'])}, Regions: {', '.join(comparison_data['project_spec']['constraints']['regions'])}",
            "blueprint_details": self._format_blueprint_details(comparison_data["blueprints"]),
            "cost_patterns": self._format_cost_patterns(comparison_data.get("cost_patterns", {}))
        }
        
        # Generate the comparison
        try:
            result = chain.invoke(inputs)
            logger.info("✅ LLM strategy comparison generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ LLM strategy comparison failed: {e}")
            raise RuntimeError(f"LLM strategy comparison failed: {e}")
    
    def _format_blueprint_details(self, blueprints: List[Dict[str, Any]]) -> str:
        """Format blueprint details for the LLM prompt."""
        details = []
        
        for bp in blueprints:
            bp_detail = f"""
BLUEPRINT: {bp['name']} ({bp['id']})
- Cloud: {bp['cloud']} | Region: {bp['region']}
- Monthly Cost: ${bp['monthly_cost']:,.2f}
- Services: {', '.join([s.get('service', 'unknown') for s in bp['services']])}
- Cost Breakdown: {', '.join([f"{service}: ${data['cost']:,.2f} ({data['percentage']:.1f}%)" for service, data in bp['cost_breakdown'].items()])}
- Optimization Score: {bp['optimization'].get('optimization_score', 'N/A')}
- Savings Potential: ${bp['optimization'].get('savings_potential', 0):,.2f}
- Risk Assessment: {bp['risks'].get('total_risks', 0)} total risks ({bp['risks'].get('critical_risks', 0)} critical, {bp['risks'].get('high_risks', 0)} high)
"""
            details.append(bp_detail)
        
        return "\n".join(details)
    
    def _format_cost_patterns(self, cost_patterns: Dict[str, Any]) -> str:
        """Format cost patterns for the LLM prompt."""
        if not cost_patterns:
            return "No cost pattern analysis available"
        
        return f"""
- Compute Ratio: {cost_patterns.get('compute_ratio', 'N/A')}%
- Storage Ratio: {cost_patterns.get('storage_ratio', 'N/A')}%
- Network Ratio: {cost_patterns.get('network_ratio', 'N/A')}%
- Dominant Cost Driver: {cost_patterns.get('dominant_cost_driver', 'N/A')}
- Cost Efficiency Score: {cost_patterns.get('cost_efficiency_score', 'N/A')}/100
- Optimization Priority: {cost_patterns.get('optimization_priority', 'N/A')}
"""

    def generate_streaming_comparison(
        self, 
        spec: ProjectSpec, 
        candidates: List[Tuple[Blueprint, Estimate]], 
        optimizations: List[Any], 
        risks: Dict[str, Any],
        cost_patterns: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate streaming comparison results for real-time display.
        
        This method provides immediate feedback as each blueprint is analyzed,
        allowing users to see results as they're generated.
        """
        try:
            logger.info("🚀 Starting streaming strategy comparison")
            
            # Initialize streaming results
            streaming_results = {
                "status": "in_progress",
                "current_step": "initializing",
                "completed_analyses": [],
                "partial_comparison": {},
                "final_recommendation": None,
                "timestamp": datetime.now().isoformat()
            }
            
            # Analyze each blueprint individually for streaming
            for i, (blueprint, estimate) in enumerate(candidates):
                if estimate is None:
                    continue
                
                # Update streaming status
                streaming_results["current_step"] = f"analyzing_blueprint_{i+1}"
                
                # Analyze this blueprint
                blueprint_analysis = self._analyze_single_blueprint(
                    blueprint, estimate, optimizations[i] if i < len(optimizations) else None,
                    risks.get(blueprint.id, []), spec
                )
                
                # Add to completed analyses
                streaming_results["completed_analyses"].append({
                    "blueprint_id": blueprint.id,
                    "analysis": blueprint_analysis,
                    "completed_at": datetime.now().isoformat()
                })
                
                # Update partial comparison
                streaming_results["partial_comparison"][blueprint.id] = blueprint_analysis
            
            # Generate final comprehensive comparison
            streaming_results["current_step"] = "generating_final_recommendation"
            final_comparison = self.compare_strategies(
                spec, candidates, optimizations, risks, cost_patterns
            )
            
            # Update final results
            streaming_results["status"] = "completed"
            streaming_results["current_step"] = "completed"
            streaming_results["final_recommendation"] = final_comparison.dict()
            streaming_results["completed_at"] = datetime.now().isoformat()
            
            logger.info("✅ Streaming strategy comparison completed")
            return streaming_results
            
        except Exception as e:
            logger.error(f"❌ Streaming strategy comparison failed: {e}")
            streaming_results["status"] = "failed"
            streaming_results["error"] = str(e)
            return streaming_results
    
    def _analyze_single_blueprint(
        self, 
        blueprint: Blueprint, 
        estimate: Estimate, 
        optimization: Any, 
        risks: List[Any], 
        spec: ProjectSpec
    ) -> Dict[str, Any]:
        """Analyze a single blueprint for streaming results."""
        
        analysis = {
            "blueprint_id": blueprint.id,
            "name": blueprint.id.upper(),
            "cloud": blueprint.cloud,
            "region": blueprint.region,
            "monthly_cost": estimate.monthly_cost,
            "cost_breakdown": self._extract_cost_breakdown(estimate),
            "optimization": self._extract_optimization_data(optimization),
            "risks": self._extract_risk_data(risks),
            "workload_suitability": self._assess_workload_suitability(blueprint, spec),
            "scalability_assessment": self._assess_scalability(blueprint, estimate),
            "cost_efficiency_score": self._calculate_cost_efficiency_score(estimate, optimization),
            "risk_score": self._calculate_risk_score(risks),
            "overall_score": 0  # Will be calculated
        }
        
        # Calculate overall score
        analysis["overall_score"] = self._calculate_overall_score(analysis)
        
        return analysis
    
    def _assess_workload_suitability(self, blueprint: Blueprint, spec: ProjectSpec) -> Dict[str, Any]:
        """Assess how well the blueprint suits the workload requirements."""
        
        suitability = {
            "score": 0,
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }
        
        # ML Training workload
        if spec.workload.train_gpus > 0:
            if any('gpu' in service.get('service', '').lower() for service in blueprint.services):
                suitability["score"] += 30
                suitability["strengths"].append("GPU support for ML training")
            else:
                suitability["weaknesses"].append("No GPU support for ML training")
                suitability["recommendations"].append("Consider adding GPU instances")
        
        # High-traffic workload
        if spec.workload.inference_qps > 500:
            if any('load' in service.get('service', '').lower() for service in blueprint.services):
                suitability["score"] += 25
                suitability["strengths"].append("Load balancing support")
            else:
                suitability["weaknesses"].append("Limited load balancing capabilities")
                suitability["recommendations"].append("Add load balancer service")
        
        # Data-intensive workload
        if spec.data.size_gb > 5000:
            if any('storage' in service.get('service', '').lower() for service in blueprint.services):
                suitability["score"] += 25
                suitability["strengths"].append("Storage optimization for large datasets")
            else:
                suitability["weaknesses"].append("Limited storage optimization")
                suitability["recommendations"].append("Optimize storage configuration")
        
        # General compute
        if spec.workload.inference_qps <= 100 and spec.workload.train_gpus == 0:
            suitability["score"] += 20
            suitability["strengths"].append("Suitable for general compute workloads")
        
        return suitability
    
    def _assess_scalability(self, blueprint: Blueprint, estimate: Estimate) -> Dict[str, Any]:
        """Assess the scalability characteristics of the blueprint."""
        
        scalability = {
            "score": 0,
            "auto_scaling": False,
            "horizontal_scaling": False,
            "vertical_scaling": False,
            "recommendations": []
        }
        
        # Check for auto-scaling capabilities
        if any('auto' in service.get('service', '').lower() for service in blueprint.services):
            scalability["auto_scaling"] = True
            scalability["score"] += 30
            scalability["strengths"].append("Auto-scaling capabilities")
        
        # Check for horizontal scaling
        if any('load' in service.get('service', '').lower() for service in blueprint.services):
            scalability["horizontal_scaling"] = True
            scalability["score"] += 25
            scalability["strengths"].append("Horizontal scaling support")
        
        # Check for vertical scaling
        if any('instance' in service.get('service', '').lower() for service in blueprint.services):
            scalability["vertical_scaling"] = True
            scalability["score"] += 20
            scalability["strengths"].append("Vertical scaling support")
        
        if scalability["score"] < 50:
            scalability["recommendations"].append("Consider adding scaling capabilities")
        
        return scalability
    
    def _calculate_cost_efficiency_score(self, estimate: Estimate, optimization: Any) -> float:
        """Calculate a cost efficiency score (0-100)."""
        base_score = 50
        
        # Adjust based on optimization potential
        if optimization and hasattr(optimization, 'metadata'):
            savings_potential = getattr(optimization, 'savings_potential', 0)
            if savings_potential > 0:
                savings_percentage = (savings_potential / estimate.monthly_cost) * 100
                if savings_percentage > 20:
                    base_score += 30
                elif savings_percentage > 10:
                    base_score += 20
                elif savings_percentage > 5:
                    base_score += 10
        
        # Adjust based on cost structure
        if hasattr(estimate, 'bom') and estimate.bom:
            total_cost = sum(item.cost for item in estimate.bom)
            compute_cost = sum(item.cost for item in estimate.bom if 'compute' in item.service.lower())
            compute_ratio = (compute_cost / total_cost) if total_cost > 0 else 0
            
            # Prefer balanced cost distribution
            if 0.4 <= compute_ratio <= 0.7:
                base_score += 10
            elif compute_ratio > 0.8:
                base_score -= 10
        
        return min(100, max(0, base_score))
    
    def _calculate_risk_score(self, risks: List[Any]) -> float:
        """Calculate a risk score (0-100, lower is better)."""
        if not risks:
            return 10  # Low risk if no risks identified
        
        risk_score = 0
        for risk in risks:
            if hasattr(risk, 'severity'):
                severity = risk.severity
            elif hasattr(risk, 'get'):
                severity = risk.get('severity', 'medium')
            else:
                severity = 'medium'
            
            if severity == 'critical':
                risk_score += 30
            elif severity == 'high':
                risk_score += 20
            elif severity == 'medium':
                risk_score += 10
            else:
                risk_score += 5
        
        return min(100, risk_score)
    
    def _calculate_overall_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall score for the blueprint."""
        
        # Weighted scoring system
        weights = {
            "cost_efficiency_score": 0.25,
            "workload_suitability_score": 0.25,
            "scalability_score": 0.20,
            "risk_score": 0.15,
            "optimization_score": 0.15
        }
        
        overall_score = 0
        
        # Cost efficiency (25%)
        overall_score += analysis["cost_efficiency_score"] * weights["cost_efficiency_score"]
        
        # Workload suitability (25%)
        overall_score += analysis["workload_suitability"]["score"] * weights["workload_suitability_score"]
        
        # Scalability (20%)
        overall_score += analysis["scalability_assessment"]["score"] * weights["scalability_score"]
        
        # Risk (15%) - invert since lower risk is better
        risk_score = 100 - analysis["risk_score"]
        overall_score += risk_score * weights["risk_score"]
        
        # Optimization potential (15%)
        optimization_score = analysis["optimization"].get('optimization_score', 50)
        overall_score += optimization_score * weights["optimization_score"]
        
        return round(overall_score, 1)
    
    def process(self, *args, **kwargs) -> Any:
        """Main processing method to satisfy abstract class requirement."""
        # This method is required by the parent class but not used in this agent
        # The main functionality is provided by compare_strategies method
        if args and len(args) > 0:
            # If called with arguments, try to use them for comparison
            return self.compare_strategies(*args, **kwargs)
        else:
            raise RuntimeError("Strategy Comparison Agent requires specific arguments. Use compare_strategies() method instead.")
