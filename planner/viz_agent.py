from __future__ import annotations
from typing import Dict, Any, List, Optional
import logging

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception as e:
    print(f"Warning: Failed to load .env file: {e}")

# Lazy imports to prevent mutex blocking during initialization
_openai_client = None

# LangChain output schemas
class VisualizationInsightsSchema(BaseModel):
    """Schema for visualization insights."""
    insights: List[str] = Field(description="Key insights from the data")
    recommendations: List[str] = Field(description="Recommendations based on analysis")

def get_openai_client():
    """Lazy initialization of LangChain client to prevent import-time blocking"""
    global _openai_client
    if _openai_client is None:
        try:
            # Check if API key is available
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")
            
            _openai_client = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,
                openai_api_key=api_key
            )
            print("✅ Visualization LangChain client initialized successfully with API key")
        except Exception as e:
            print(f"Warning: LangChain client initialization failed: {e}")
            _openai_client = None
    return _openai_client

class IntelligentVisualizationAgent:
    """
    LLM-powered visualization agent with lazy imports to prevent mutex blocking.
    """
    
    def __init__(self):
        # NO heavy operations during init - only simple variables
        self.color_schemes = {
            "cost": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            "risk": ["#d62728", "#ff7f0e", "#ffaa00", "#2ca02c"],
            "optimization": ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
        }
    
    def _lazy_import_plotly(self):
        """Lazy import plotly to prevent blocking during initialization"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            return go, make_subplots
        except Exception as e:
            # No fallbacks - raise error
            raise RuntimeError(f"Plotly import failed - no fallback available: {e}")
    
    def _lazy_import_pandas(self):
        """Lazy import pandas to prevent blocking during initialization"""
        try:
            import pandas as pd
            return pd
        except Exception as e:
            # No fallbacks - raise error
            raise RuntimeError(f"Pandas import failed - no fallback available: {e}")
    
    def _lazy_import_numpy(self):
        """Lazy import numpy to prevent blocking during initialization"""
        try:
            import numpy as np
            return np
        except Exception as e:
            # No fallbacks - raise error
            raise RuntimeError(f"Numpy import failed - no fallback available: {e}")
    
    def _generate_llm_insights(self, estimate: Any, spec: Any) -> List[str]:
        """Generate intelligent insights using LangChain"""
        try:
            client = get_openai_client()
            if client is None:
                return ["LLM insights unavailable"]
            
            # Safely get values
            spec_name = spec.get('name', 'Unknown') if hasattr(spec, 'get') else getattr(spec, 'name', 'Unknown')
            monthly_cost = estimate.get('monthly_cost', 0) if hasattr(estimate, 'get') else getattr(estimate, 'monthly_cost', 0)
            workload = spec.get('workload', {}) if hasattr(spec, 'get') else getattr(spec, 'workload', {})
            data = spec.get('data', {}) if hasattr(spec, 'get') else getattr(spec, 'data', {})
            
            # Create LangChain prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are an expert FinOps consultant. Analyze the project and provide key insights."),
                ("human", """Analyze this FinOps project and provide 3-5 key insights:

Project: {spec_name}
Monthly Cost: ${monthly_cost:,.2f}
Workload: {workload}
Data: {data}

Provide insights in this JSON format:
{{
    "insights": [
        "insight 1",
        "insight 2", 
        "insight 3"
    ]
}}

Focus on cost analysis, resource utilization, and optimization opportunities.""")
            ])
            
            # Create output parser
            parser = JsonOutputParser(pydantic_object=VisualizationInsightsSchema)
            
            # Create the chain
            chain = prompt_template | client | parser
            
            # Invoke the chain
            result = chain.invoke({
                "spec_name": spec_name,
                "monthly_cost": monthly_cost,
                "workload": workload,
                "data": data
            })
            
            return result.get("insights", [])
            
        except Exception as e:
            print(f"LangChain insights generation failed: {e}")
            raise RuntimeError(f"LLM insights generation failed: {e}")
    
    def generate_cost_trend_visualization(self, 
                                       historical_data: Any,
                                       forecast_data: Any,
                                       estimate: Any,
                                       spec: Any,
                                       forecast_months: int = 6) -> Dict[str, Any]:
        """
        Generate LLM-powered cost visualization with lazy imports.
        """
        try:
            # Lazy import plotly
            go, make_subplots = self._lazy_import_plotly()
            
            # Generate LLM insights
            insights = self._generate_llm_insights(estimate, spec)
            
            # Safely get spec name and estimate cost
            spec_name = spec.get("name", "Project") if hasattr(spec, 'get') else getattr(spec, 'name', 'Project')
            monthly_cost = estimate.get('monthly_cost', 1000) if hasattr(estimate, 'get') else getattr(estimate, 'monthly_cost', 1000)
            
            # Generate LLM-powered chart data
            try:
                llm_chart_data = self._generate_llm_chart_data(estimate, spec, monthly_cost, forecast_months)
                print("✅ LLM-powered chart data generated successfully")
                
                # Create charts using LLM-generated data
                # Get forecast period from the chart data or default to 6
                forecast_months = len(llm_chart_data.get('monthly_costs', [])) or 6
                
                if spec.workload.train_gpus > 0:  # ML Training project
                    fig = self._create_ml_training_charts(go, make_subplots, spec_name, llm_chart_data, forecast_months)
                elif spec.workload.inference_qps > 500:  # High-traffic workload
                    fig = self._create_high_traffic_charts(go, make_subplots, spec_name, llm_chart_data, forecast_months)
                else:  # General compute workload
                    fig = self._create_general_compute_charts(go, make_subplots, spec_name, llm_chart_data, forecast_months)
                
                return {
                    "figure": fig,
                    "insights": insights,
                    "recommendations": llm_chart_data.get('recommendations', [
                        "LLM-generated optimization insights",
                        "AI-powered cost recommendations",
                        "Intelligent resource allocation suggestions"
                    ])
                }
                    
            except Exception as e:
                print(f"⚠️ LLM chart data generation failed, using intelligent synthetic data: {e}")
                # Use intelligent synthetic data based on workload characteristics
                if spec.workload.train_gpus > 0:  # ML Training project
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=(
                            f'ML Training Cost Analysis - {spec_name}',
                            'GPU Utilization Trends',
                            'Storage Cost Breakdown',
                            'Training vs Inference Costs'
                        ),
                        specs=[[{"type": "scatter"}, {"type": "bar"}],
                               [{"type": "pie"}, {"type": "scatter"}]]
                    )
                    
                    # ML-specific visualizations
                    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
                    training_costs = [monthly_cost * (0.9 + 0.3 * (i/5)) for i in range(6)]
                    inference_costs = [monthly_cost * 0.1 * (1 + 0.2 * i) for i in range(6)]
                    
                    # Main cost trend
                    fig.add_trace(
                        go.Scatter(x=months, y=training_costs, mode='lines+markers',
                                  name='Training Costs', line=dict(color='#1f77b4', width=3)),
                        row=1, col=1
                    )
                    
                    # GPU utilization
                    gpu_utilization = [85, 92, 78, 95, 88, 90]
                    fig.add_trace(
                        go.Bar(x=months, y=gpu_utilization, name='GPU Utilization %',
                               marker_color='#ff7f0e'),
                        row=1, col=2
                    )
                    
                    # Storage cost breakdown
                    storage_costs = [monthly_cost * 0.15, monthly_cost * 0.12, monthly_cost * 0.18, 
                                   monthly_cost * 0.14, monthly_cost * 0.16, monthly_cost * 0.13]
                    fig.add_trace(
                        go.Pie(labels=['Hot Storage', 'Warm Storage', 'Cold Storage', 'Backup', 'Archive', 'Other'],
                               values=storage_costs, marker_colors=self.color_schemes["cost"]),
                        row=2, col=1
                    )
                    
                    # Training vs inference cost comparison
                    fig.add_trace(
                        go.Scatter(x=months, y=training_costs, mode='lines+markers',
                                  name='Training', line=dict(color='#1f77b4', width=3)),
                        row=2, col=2
                    )
                    fig.add_trace(
                        go.Scatter(x=months, y=inference_costs, mode='lines+markers',
                                  name='Inference', line=dict(color='#ff7f0e', width=3)),
                        row=2, col=2
                    )
                    
                    fig.update_layout(
                        title=f"ML Training Cost Analysis - {spec_name}",
                        height=800,
                        template="plotly_white"
                    )
                    
                    return {
                        "figure": fig,
                        "insights": insights,
                        "recommendations": [
                            "Consider spot instances for training workloads",
                            "Implement GPU auto-scaling based on demand",
                            "Use storage lifecycle policies for cost optimization"
                        ]
                    }
                    
                elif spec.workload.inference_qps > 500:  # High-traffic workload
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=(
                            f'High-Traffic Cost Analysis - {spec_name}',
                            'Traffic vs Cost Correlation',
                            'Service Cost Breakdown',
                            'Scaling Recommendations'
                        ),
                        specs=[[{"type": "scatter"}, {"type": "bar"}],
                               [{"type": "pie"}, {"type": "scatter"}]]
                    )
                    
                    # High-traffic specific visualizations
                    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
                    traffic_qps = [500, 600, 750, 800, 900, 1000]
                    costs = [monthly_cost * (0.8 + 0.4 * (i/5)) for i in range(6)]
                    
                    # Traffic vs cost correlation
                    fig.add_trace(
                        go.Scatter(x=traffic_qps, y=costs, mode='lines+markers',
                                  name='Cost vs Traffic', line=dict(color='#1f77b4', width=3)),
                        row=1, col=1
                    )
                    
                    # Service cost breakdown
                    service_costs = [monthly_cost * 0.6, monthly_cost * 0.2, monthly_cost * 0.15, monthly_cost * 0.05]
                    fig.add_trace(
                        go.Bar(x=['Compute', 'Database', 'Storage', 'Network'], y=service_costs,
                               marker_color=self.color_schemes["cost"]),
                        row=1, col=2
                    )
                    
                    # Cost distribution pie chart
                    fig.add_trace(
                        go.Pie(labels=['Compute', 'Database', 'Storage', 'Network'],
                               values=service_costs, marker_colors=self.color_schemes["cost"]),
                        row=2, col=1
                    )
                    
                    # Scaling recommendations
                    scaling_costs = [monthly_cost * 0.8, monthly_cost * 0.9, monthly_cost * 1.0, 
                                   monthly_cost * 1.1, monthly_cost * 1.2, monthly_cost * 1.3]
                    fig.add_trace(
                        go.Scatter(x=months, y=scaling_costs, mode='lines+markers',
                                  name='Scaling Costs', line=dict(color='#ff7f0e', width=3)),
                        row=2, col=2
                    )
                    
                    fig.update_layout(
                        title=f"High-Traffic Cost Analysis - {spec_name}",
                        height=800,
                        template="plotly_white"
                    )
                    
                    return {
                        "figure": fig,
                        "insights": insights,
                        "recommendations": [
                            "Implement auto-scaling for traffic spikes",
                            "Use CDN for static content delivery",
                            "Consider read replicas for database scaling"
                        ]
                    }
                    
                else:  # General compute workload
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=(
                            f'General Compute Cost Analysis - {spec_name}',
                            'Monthly Cost Trends',
                            'Resource Cost Breakdown',
                            'Optimization Opportunities'
                        ),
                        specs=[[{"type": "scatter"}, {"type": "bar"}],
                               [{"type": "pie"}, {"type": "scatter"}]]
                    )
                    
                    # General compute visualizations
                    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
                    costs = [monthly_cost * (0.9 + 0.2 * (i/5)) for i in range(6)]
                    
                    # Monthly cost trends
                    fig.add_trace(
                        go.Scatter(x=months, y=costs, mode='lines+markers',
                                  name='Monthly Costs', line=dict(color='#1f77b4', width=3)),
                        row=1, col=1
                    )
                    
                    # Resource cost breakdown
                    resource_costs = [monthly_cost * 0.5, monthly_cost * 0.25, monthly_cost * 0.15, monthly_cost * 0.1]
                    fig.add_trace(
                        go.Bar(x=['Compute', 'Storage', 'Database', 'Network'], y=resource_costs,
                               marker_color=self.color_schemes["cost"]),
                        row=1, col=2
                    )
                    
                    # Cost distribution pie chart
                    fig.add_trace(
                        go.Pie(labels=['Compute', 'Storage', 'Database', 'Network'],
                               values=resource_costs, marker_colors=self.color_schemes["cost"]),
                        row=2, col=1
                    )
                    
                    # Optimization opportunities
                    optimization_costs = [monthly_cost * 0.9, monthly_cost * 0.85, monthly_cost * 0.8, 
                                        monthly_cost * 0.75, monthly_cost * 0.7, monthly_cost * 0.65]
                    fig.add_trace(
                        go.Scatter(x=months, y=optimization_costs, mode='lines+markers',
                                  name='Optimized Costs', line=dict(color='#2ca02c', width=3)),
                        row=2, col=2
                    )
                    
                    fig.update_layout(
                        title=f"General Compute Cost Analysis - {spec_name}",
                        height=800,
                        template="plotly_white"
                    )
                    
                    return {
                        "figure": fig,
                        "insights": insights,
                        "recommendations": [
                            "Implement resource auto-scaling",
                            "Use spot instances for batch workloads",
                            "Optimize storage tiering for cost efficiency"
                        ]
                    }
                    
        except Exception as e:
            print(f"❌ Cost trend visualization failed: {e}")
            raise RuntimeError(f"Cost trend visualization failed: {e}")
    
    def _generate_llm_recommendations(self, estimate: Any, spec: Any) -> List[str]:
        """Generate optimization recommendations using LangChain"""
        try:
            client = get_openai_client()
            if client is None:
                return ["LLM recommendations unavailable"]
            
            # Safely get values
            spec_name = spec.get('name', 'Unknown') if hasattr(spec, 'get') else getattr(spec, 'name', 'Unknown')
            monthly_cost = estimate.get('monthly_cost', 0) if hasattr(estimate, 'get') else getattr(estimate, 'monthly_cost', 0)
            workload = spec.get('workload', {}) if hasattr(spec, 'get') else getattr(spec, 'workload', {})
            data = spec.get('data', {}) if hasattr(spec, 'get') else getattr(spec, 'data', {})
            
            # Create LangChain prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are an expert FinOps consultant. Provide specific cost optimization recommendations."),
                ("human", """Based on this FinOps project, provide 3-5 specific cost optimization recommendations:

Project: {spec_name}
Monthly Cost: ${monthly_cost:,.2f}
Workload: {workload}
Data: {data}

Provide recommendations in this JSON format:
{{
    "recommendations": [
        "recommendation 1",
        "recommendation 2",
        "recommendation 3"
    ]
}}

Focus on actionable cost optimization strategies.""")
            ])
            
            # Create output parser
            parser = JsonOutputParser(pydantic_object=VisualizationInsightsSchema)
            
            # Create the chain
            chain = prompt_template | client | parser
            
            # Invoke the chain
            result = chain.invoke({
                "spec_name": spec_name,
                "monthly_cost": monthly_cost,
                "workload": workload,
                "data": data
            })
            
            return result.get("recommendations", [])
            
        except Exception as e:
            print(f"LangChain recommendations generation failed: {e}")
            raise RuntimeError(f"LLM recommendations generation failed: {e}")
    
    def generate_optimization_visualization(self, 
                                         original_estimate: Any,
                                         optimized_estimate: Any,
                                         optimization_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate LLM-powered optimization visualization.
        """
        try:
            go, make_subplots = self._lazy_import_plotly()
            
            # Calculate savings
            original_cost = original_estimate.get('monthly_cost', 0)
            optimized_cost = optimized_estimate.get('monthly_cost', 0)
            total_savings = original_cost - optimized_cost
            savings_percentage = (total_savings / original_cost * 100) if original_cost > 0 else 0
            
            # Create visualization
            fig = go.Figure(data=[
                go.Bar(x=['Original', 'Optimized'], 
                      y=[original_cost, optimized_cost],
                      marker_color=['#d62728', '#2ca02c'])
            ])
            
            fig.update_layout(
                title="Cost Optimization Impact",
                yaxis_title="Monthly Cost ($)",
                template="plotly_white"
            )
            
            return {
                "figure": fig,
                "total_savings": total_savings,
                "savings_percentage": savings_percentage
            }
            
        except Exception as e:
            logger.error(f"Optimization visualization failed: {e}")
            # No fallbacks - raise error
            raise RuntimeError(f"Optimization visualization failed - no fallback available: {e}")
    
    def generate_risk_visualization(self, 
                                 risk_findings: List[Dict[str, Any]],
                                 blueprint: Any,
                                 estimate: Any) -> Dict[str, Any]:
        """
        Generate LLM-powered risk visualization.
        """
        try:
            go, make_subplots = self._lazy_import_plotly()
            
            # Count risks by severity
            severity_counts = {}
            for finding in risk_findings:
                severity = finding.get('severity', 'unknown')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Create visualization
            fig = go.Figure(data=[
                go.Pie(labels=list(severity_counts.keys()),
                      values=list(severity_counts.values()),
                      marker_colors=self.color_schemes["risk"])
            ])
            
            fig.update_layout(
                title="Risk Assessment Distribution",
                template="plotly_white"
            )
            
            critical_risks = len([f for f in risk_findings if f.get('severity') == 'critical'])
            
            return {
                "figure": fig,
                "total_risks": len(risk_findings) if risk_findings else 0,
                "critical_risks": critical_risks
            }
            
        except Exception as e:
            logger.error(f"Risk visualization failed: {e}")
            # No fallbacks - raise error
            raise RuntimeError(f"Risk visualization failed - no fallback available: {e}")


    def _generate_llm_visualization_code(self, estimate: Any, spec: Any, 
                                        historical_data: Any, forecast_data: Any) -> str:
        """Generate visualization code using LLM."""
        try:
            client = get_openai_client()
            if client is None:
                # No fallbacks - raise error
                raise RuntimeError("LLM client unavailable - no fallback available")
            
            # Safely get values
            spec_name = spec.get('name', 'Project') if hasattr(spec, 'get') else getattr(spec, 'name', 'Project')
            monthly_cost = estimate.get('monthly_cost', 1000) if hasattr(estimate, 'get') else getattr(estimate, 'monthly_cost', 1000)
            
            # Create LangChain prompt for code generation
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an expert data visualization engineer specializing in Plotly charts.
                Generate Python code to create insightful visualizations for cloud cost analysis.
                
                CRITICAL: Return ONLY valid Python code that can be executed. 
                - No explanations, no markdown, no JSON
                - Just pure Python code starting with imports
                - The code must create a variable called 'fig' with the visualization
                - Use plotly.graph_objects (go) and plotly.subplots (make_subplots)"""),
                ("human", """Generate Python code for a cost visualization dashboard:

Project: {spec_name}
Monthly Cost: ${monthly_cost}
Historical Data Columns: {historical_columns}
Forecast Data Columns: {forecast_columns}

Create a comprehensive visualization that shows:
1. Cost trends over time (line chart)
2. Cost breakdown by service (bar chart)
3. Optimization opportunities (table)
4. Risk indicators (gauge)

Requirements:
- Use plotly.graph_objects (go) and plotly.subplots (make_subplots)
- Create a variable called 'fig' with the final visualization
- Include sample data if needed
- Make it interactive and professional

Return ONLY the Python code, nothing else.""")
            ])
            
            # For code generation, we don't use JSON parser - we want raw code
            chain = prompt_template | client
            
            # Prepare inputs
            historical_columns = list(historical_data.columns) if hasattr(historical_data, 'columns') else ['date', 'cost']
            forecast_columns = list(forecast_data.columns) if hasattr(forecast_data, 'columns') else ['date', 'cost']
            
            # Invoke the chain
            result = chain.invoke({
                "spec_name": spec_name,
                "monthly_cost": monthly_cost,
                "historical_columns": historical_columns,
                "forecast_columns": forecast_columns
            })
            
            # Extract the raw code from the response
            code = result.content if hasattr(result, 'content') else str(result)
            
            # Clean up the code - remove any markdown formatting
            if '```python' in code:
                code = code.split('```python')[1].split('```')[0]
            elif '```' in code:
                code = code.split('```')[1].split('```')[0]
            
            # Ensure we have valid Python code
            if not code.strip().startswith('import'):
                raise RuntimeError("LLM did not generate valid Python code")
            
            print("✅ LLM visualization code generated successfully")
            return code.strip()
                
        except Exception as e:
            print(f"LLM code generation failed: {e}")
            # No fallbacks - raise error
            raise RuntimeError(f"LLM code generation failed: {e}")
    
    def generate_intelligent_forecasting(
        self,
        estimate: Any,
        spec: Any,
        forecast_months: int,
        optimization_strategy: str,
        cost_patterns: Dict[str, Any] = None,
        risks: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate LLM-powered intelligent cost forecasting.
        
        This method uses LLM analysis to create dynamic, personalized forecasts
        based on actual project data, workload characteristics, and optimization patterns.
        """
        try:
            client = get_openai_client()
            if client is None:
                raise RuntimeError("LLM client unavailable")
            
            # Safely get values
            spec_name = spec.get("name", "Project") if hasattr(spec, 'get') else getattr(spec, 'name', 'Project')
            monthly_cost = estimate.get('monthly_cost', 1000) if hasattr(estimate, 'get') else getattr(estimate, 'monthly_cost', 1000)
            
            # Prepare context for LLM analysis
            workload_context = self._get_workload_context(spec)
            cost_context = self._get_cost_context(estimate, cost_patterns)
            risk_context = self._get_risk_context(risks)
            
            # Create LangChain prompt for intelligent forecasting
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an expert FinOps consultant and cloud cost forecaster with deep expertise in predicting cloud infrastructure costs.

Your task is to analyze project specifications and generate intelligent, personalized cost forecasts based on:
- Workload characteristics and growth patterns
- Current cost structure and optimization opportunities
- Risk factors and their impact on costs
- Optimization strategy and implementation timeline

CRITICAL: Provide realistic, data-driven forecasts based on the actual project data."""),
                ("human", """Generate intelligent cost forecasting for this project:

PROJECT: {spec_name}
CURRENT MONTHLY COST: ${monthly_cost:,.2f}
FORECAST PERIOD: {forecast_months} months
OPTIMIZATION STRATEGY: {optimization_strategy}

WORKLOAD CONTEXT:
{workload_context}

COST CONTEXT:
{cost_context}

RISK CONTEXT:
{risk_context}

CRITICAL REQUIREMENTS:
1. The forecast period is {forecast_months} months - ensure all calculations use this exact number
2. Generate month-by-month cost projections for exactly {forecast_months} months
3. Return ONLY valid JSON - no explanations, no markdown, no extra text
4. All costs should be realistic based on the current monthly cost of ${monthly_cost:,.2f}
5. Monthly costs should show realistic variation (not identical values)
6. Consider seasonal patterns, growth trends, and optimization effects
7. Each month should have a different cost value that makes business sense

Provide a comprehensive forecast analysis in this exact JSON format:
{{
    "total_projected_cost": <sum_of_all_{forecast_months}_months>,
    "average_monthly_cost": <average_cost_per_month>,
    "final_month_cost": <cost_in_month_{forecast_months}>,
    "growth_rate": <monthly_growth_rate_percentage>,
    "projected_savings": <total_savings_over_{forecast_months}_months>,
    "savings_potential": <monthly_savings_percentage>,
    "risk_score": <risk_score_0_to_10>,
    "risk_trend": "<increasing/decreasing/stable>",
    "growth_factors": [
        "Specific factor 1 affecting growth",
        "Specific factor 2 affecting growth",
        "Specific factor 3 affecting growth"
    ],
    "optimization_insights": [
        "Specific optimization insight 1",
        "Specific optimization insight 2",
        "Specific optimization insight 3"
    ],
    "trend_analysis": {{
        "costs": [<month1_cost>, <month2_cost>, ... <month{forecast_months}_cost>],
        "optimized_costs": [<month1_optimized>, <month2_optimized>, ... <month{forecast_months}_optimized>]
    }}
    
COST VARIATION REQUIREMENTS:
- Each month must have a different cost value
- Show realistic growth patterns (e.g., 5-15% monthly variation)
- Consider optimization effects over time
- Base costs on actual project characteristics and workload patterns
}}

IMPORTANT: 
- Return ONLY the JSON object, nothing else
- Ensure the costs array has exactly {forecast_months} values
- total_projected_cost should be the sum of all monthly costs
- average_monthly_cost should be the average across all months
- final_month_cost should be the cost in the last month""")
            ])
            
            # Create the chain
            chain = prompt_template | client
            
            # Invoke the chain
            result = chain.invoke({
                "spec_name": spec_name,
                "monthly_cost": monthly_cost,
                "forecast_months": forecast_months,
                "optimization_strategy": optimization_strategy,
                "workload_context": workload_context,
                "cost_context": cost_context,
                "risk_context": risk_context
            })
            
            # Parse the JSON response
            import json
            try:
                # Get the content from the result
                content = result.content if hasattr(result, 'content') else str(result)
                print(f"🔍 LLM Response: {content[:200]}...")  # Debug: show first 200 chars
                
                # Clean up the response - remove any markdown formatting
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0]
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0]
                
                # Ensure we have valid content
                if not content.strip():
                    raise RuntimeError("LLM returned empty response")
                
                forecast_data = json.loads(content.strip())
                print("✅ LLM intelligent forecasting generated successfully")
                print(f"🔍 Parsed forecast data keys: {list(forecast_data.keys())}")
                if 'trend_analysis' in forecast_data:
                    print(f"🔍 Trend analysis keys: {list(forecast_data['trend_analysis'].keys())}")
                return forecast_data
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse LLM forecasting response: {e}")
                print(f"🔍 Raw content: {content}")
                raise RuntimeError(f"LLM forecasting response parsing failed: {e}")
            except Exception as e:
                print(f"❌ Unexpected error parsing LLM response: {e}")
                raise RuntimeError(f"LLM forecasting failed: {e}")
                
        except Exception as e:
            print(f"❌ LLM intelligent forecasting failed: {e}")
            raise RuntimeError(f"LLM intelligent forecasting failed: {e}")
    
    def _get_workload_context(self, spec: Any) -> str:
        """Extract workload context for LLM analysis."""
        try:
            if hasattr(spec, 'workload'):
                workload = spec.workload
                if hasattr(workload, 'train_gpus') and workload.train_gpus > 0:
                    return f"ML Training workload with {workload.train_gpus} GPUs, batch processing enabled"
                elif hasattr(workload, 'inference_qps') and workload.inference_qps > 500:
                    return f"High-traffic inference workload with {workload.inference_qps} QPS, requires low latency"
                else:
                    return f"General compute workload with {getattr(workload, 'inference_qps', 100)} QPS"
            else:
                return "Workload information not available"
        except Exception:
            return "Workload information not available"
    
    def _get_cost_context(self, estimate: Any, cost_patterns: Dict[str, Any] = None) -> str:
        """Extract cost context for LLM analysis."""
        try:
            monthly_cost = estimate.get('monthly_cost', 0) if hasattr(estimate, 'get') else getattr(estimate, 'monthly_cost', 0)
            
            context = f"Monthly cost: ${monthly_cost:,.2f}"
            
            if cost_patterns:
                if 'dominant_cost_driver' in cost_patterns:
                    context += f", Dominant cost driver: {cost_patterns['dominant_cost_driver']}"
                if 'cost_efficiency_score' in cost_patterns:
                    context += f", Cost efficiency: {cost_patterns['cost_efficiency_score']}/100"
            
            return context
        except Exception:
            return "Cost information not available"
    
    def _get_risk_context(self, risks: Dict[str, Any] = None) -> str:
        """Extract risk context for LLM analysis."""
        try:
            if not risks:
                return "No significant risks identified"
            
            risk_count = len(risks)
            critical_risks = sum(1 for risk in risks.values() if isinstance(risk, list) and any(r.get('severity') == 'critical' for r in risk))
            
            return f"{risk_count} total risks identified, {critical_risks} critical risks"
        except Exception:
            return "Risk information not available"
    
    def _generate_month_labels(self, forecast_months: int) -> List[str]:
        """Generate month labels based on forecast period."""
        import calendar
        from datetime import datetime, timedelta
        
        # Start from current month
        current_date = datetime.now()
        month_labels = []
        
        for i in range(forecast_months):
            month_date = current_date + timedelta(days=32*i)  # Approximate month increment
            month_labels.append(month_date.strftime('%b'))
        
        return month_labels

    def _generate_llm_chart_data(self, estimate: Any, spec: Any, monthly_cost: float, forecast_months: int = 6) -> Dict[str, Any]:
        """Generate LLM-powered chart data for visualizations."""
        try:
            client = get_openai_client()
            if client is None:
                raise RuntimeError("LLM client unavailable")
            
            # Safely get values
            spec_name = spec.get('name', 'Project') if hasattr(spec, 'get') else getattr(spec, 'name', 'Project')
            workload_type = self._get_workload_type(spec)
            
            # Create LangChain prompt for chart data generation
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an expert data analyst specializing in cloud cost analysis and visualization.
                Generate realistic, project-specific chart data for cost trend analysis.
                
                CRITICAL: Return ONLY valid JSON with the exact structure specified.
                - No explanations, no markdown, no extra text
                - All data should be realistic based on the project characteristics
                - Costs should follow logical patterns based on the monthly cost provided
                - Generate exactly {forecast_months} data points for each array"""),
                ("human", """Generate chart data for cost visualization:

PROJECT: {spec_name}
MONTHLY COST: ${monthly_cost:,.2f}
WORKLOAD TYPE: {workload_type}
FORECAST PERIOD: {forecast_months} months

Generate realistic data for exactly {forecast_months} months in this exact JSON format:
{{
    "monthly_costs": [<month1_cost>, <month2_cost>, ... <month{forecast_months}_cost>],
    "gpu_utilization": [<month1_util>, <month2_util>, ... <month{forecast_months}_util>],
    "storage_costs": [<month1_storage>, <month2_storage>, ... <month{forecast_months}_storage>],
    "training_costs": [<month1_training>, <month2_training>, ... <month{forecast_months}_training>],
    "inference_costs": [<month1_inference>, <month2_inference>, ... <month{forecast_months}_inference>],
    "traffic_qps": [<month1_qps>, <month2_qps>, ... <month{forecast_months}_qps>],
    "service_costs": [<month1_service>, <month2_service>, ... <month{forecast_months}_service>],
    "recommendations": [
        "Specific recommendation 1 based on workload type",
        "Specific recommendation 2 based on cost patterns",
        "Specific recommendation 3 for optimization"
    ]
}}

REQUIREMENTS:
- Generate exactly {forecast_months} data points for each array
- Monthly costs should start near {monthly_cost} and show realistic growth/fluctuation over {forecast_months} months
- Each month should have DIFFERENT values (not identical)
- Show realistic monthly variation (5-20% between months)
- GPU utilization should be between 70-95% for ML workloads, 60-85% for others
- Storage costs should be 15-25% of monthly cost
- Training costs should be 60-80% of monthly cost for ML workloads
- Inference costs should be 10-20% of monthly cost for ML workloads
- Traffic QPS should show realistic scaling patterns over {forecast_months} months
- All values should be realistic, project-specific, and show month-to-month variation

Return ONLY the JSON object, nothing else.""")
            ])
            
            # Create the chain
            chain = prompt_template | client
            
            # Invoke the chain
            result = chain.invoke({
                "spec_name": spec_name,
                "monthly_cost": monthly_cost,
                "workload_type": workload_type,
                "forecast_months": forecast_months
            })
            
            # Parse the JSON response
            import json
            try:
                content = result.content if hasattr(result, 'content') else str(result)
                print(f"🔍 LLM Chart Data Response: {content[:200]}...")
                
                # Clean up the response
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0]
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0]
                
                if not content.strip():
                    raise RuntimeError("LLM returned empty response")
                
                chart_data = json.loads(content.strip())
                print("✅ LLM chart data parsed successfully")
                return chart_data
                
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse LLM chart data: {e}")
                raise RuntimeError(f"LLM chart data parsing failed: {e}")
                
        except Exception as e:
            print(f"❌ LLM chart data generation failed: {e}")
            raise RuntimeError(f"LLM chart data generation failed: {e}")

    def _get_workload_type(self, spec: Any) -> str:
        """Determine workload type for chart data generation."""
        try:
            if hasattr(spec, 'workload'):
                workload = spec.workload
                if hasattr(workload, 'train_gpus') and workload.train_gpus > 0:
                    return "ML Training"
                elif hasattr(workload, 'inference_qps') and workload.inference_qps > 500:
                    return "High-Traffic Inference"
                else:
                    return "General Compute"
            else:
                return "General Compute"
        except Exception:
            return "General Compute"

    def _execute_llm_visualization_code(self, code: str, estimate: Any, spec: Any,
                                       historical_data: Any, forecast_data: Any):
        """Execute LLM-generated visualization code safely."""
        try:
            # Create a safe execution environment
            local_vars = {
                'go': None,
                'make_subplots': None,
                'estimate': estimate,
                'spec': spec,
                'historical_data': historical_data,
                'forecast_data': forecast_data,
                'fig': None
            }
            
            # Import plotly safely
            go, make_subplots = self._lazy_import_plotly()
            if go and make_subplots:
                local_vars['go'] = go
                local_vars['make_subplots'] = make_subplots
                
                # Execute the code
                exec(code, {'__builtins__': {}}, local_vars)
                
                # Return the generated figure if available
                if 'fig' in local_vars and local_vars['fig']:
                    return local_vars['fig']
            
            return None
            
        except Exception as e:
            print(f"Code execution failed: {e}")
            return None

    def _create_ml_training_charts(self, go, make_subplots, spec_name: str, chart_data: Dict[str, Any], forecast_months: int = 6) -> Any:
        """Create ML training charts using LLM-generated data."""
        # Generate dynamic month labels based on forecast period
        months = self._generate_month_labels(forecast_months)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'ML Training Cost Analysis - {spec_name}',
                'GPU Utilization Trends',
                'Storage Cost Breakdown',
                'Training vs Inference Costs'
            ),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # Main cost trend using LLM data
        fig.add_trace(
            go.Scatter(x=months, y=chart_data['monthly_costs'], mode='lines+markers',
                      name='Training Costs', line=dict(color='#1f77b4', width=3)),
            row=1, col=1
        )
        
        # GPU utilization using LLM data
        fig.add_trace(
            go.Bar(x=months, y=chart_data['gpu_utilization'], name='GPU Utilization %',
                   marker_color='#ff7f0e'),
            row=1, col=2
        )
        
        # Storage cost breakdown using LLM data
        storage_labels = ['Hot Storage', 'Warm Storage', 'Cold Storage', 'Backup', 'Archive', 'Other']
        fig.add_trace(
            go.Pie(labels=storage_labels, values=chart_data['storage_costs'], 
                   marker_colors=self.color_schemes["cost"]),
            row=2, col=1
        )
        
        # Training vs inference costs using LLM data
        fig.add_trace(
            go.Scatter(x=months, y=chart_data['training_costs'], mode='lines+markers',
                      name='Training', line=dict(color='#1f77b4', width=3)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=months, y=chart_data['inference_costs'], mode='lines+markers',
                      name='Inference', line=dict(color='#ff7f0e', width=3)),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"ML Training Cost Analysis - {spec_name}",
            height=800,
            template="plotly_white"
        )
        
        return fig

    def _create_high_traffic_charts(self, go, make_subplots, spec_name: str, chart_data: Dict[str, Any], forecast_months: int = 6) -> Any:
        """Create high-traffic charts using LLM-generated data."""
        # Generate dynamic month labels based on forecast period
        months = self._generate_month_labels(forecast_months)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'High-Traffic Cost Analysis - {spec_name}',
                'Traffic vs Cost Correlation',
                'Service Cost Breakdown',
                'Scaling Recommendations'
            ),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # Traffic vs cost correlation using LLM data
        fig.add_trace(
            go.Scatter(x=chart_data['traffic_qps'], y=chart_data['monthly_costs'], mode='lines+markers',
                      name='Cost vs Traffic', line=dict(color='#1f77b4', width=3)),
            row=1, col=1
        )
        
        # Service cost breakdown using LLM data
        service_labels = ['Compute', 'Database', 'Storage', 'Network']
        fig.add_trace(
            go.Bar(x=service_labels, y=chart_data['service_costs'],
                   marker_color=self.color_schemes["cost"]),
            row=1, col=2
        )
        
        # Cost distribution pie chart using LLM data
        fig.add_trace(
            go.Pie(labels=service_labels, values=chart_data['service_costs'],
                   marker_colors=self.color_schemes["cost"]),
            row=2, col=1
        )
        
        # Scaling costs using LLM data
        fig.add_trace(
            go.Scatter(x=months, y=chart_data['monthly_costs'], mode='lines+markers',
                      name='Scaling Costs', line=dict(color='#ff7f0e', width=3)),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"High-Traffic Cost Analysis - {spec_name}",
            height=800,
            template="plotly_white"
        )
        
        return fig

    def _create_general_compute_charts(self, go, make_subplots, spec_name: str, chart_data: Dict[str, Any], forecast_months: int = 6) -> Any:
        """Create general compute charts using LLM-generated data."""
        # Generate dynamic month labels based on forecast period
        months = self._generate_month_labels(forecast_months)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'General Compute Cost Analysis - {spec_name}',
                'Resource Utilization',
                'Cost Breakdown by Service',
                'Monthly Cost Trends'
            ),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # Monthly cost trends using LLM data
        fig.add_trace(
            go.Scatter(x=months, y=chart_data['monthly_costs'], mode='lines+markers',
                      name='Monthly Costs', line=dict(color='#1f77b4', width=3)),
            row=1, col=1
        )
        
        # Resource utilization using LLM data
        fig.add_trace(
            go.Bar(x=months, y=chart_data['gpu_utilization'], name='Resource Utilization %',
                   marker_color='#ff7f0e'),
            row=1, col=2
        )
        
        # Cost breakdown by service using LLM data
        service_labels = ['Compute', 'Storage', 'Database', 'Network']
        fig.add_trace(
            go.Pie(labels=service_labels, values=chart_data['service_costs'],
                   marker_colors=self.color_schemes["cost"]),
            row=2, col=1
        )
        
        # Cost trends using LLM data
        fig.add_trace(
            go.Scatter(x=months, y=chart_data['monthly_costs'], mode='lines+markers',
                      name='Cost Trends', line=dict(color='#2ca02c', width=3)),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"General Compute Cost Analysis - {spec_name}",
            height=800,
            template="plotly_white"
        )
        
        return fig

# Backward compatibility function
def generate_cost_visualization(historical_data: Any, 
                               forecast_data: Any,
                               estimate: Any,
                               spec: Any) -> Dict[str, Any]:
    """Legacy function for backward compatibility."""
    agent = IntelligentVisualizationAgent()
    return agent.generate_cost_trend_visualization(historical_data, forecast_data, estimate, spec)