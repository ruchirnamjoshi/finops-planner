from __future__ import annotations
from typing import Dict, Any, List, Optional

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

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
            print(f"Warning: Plotly import failed: {e}")
            return None, None
    
    def _lazy_import_pandas(self):
        """Lazy import pandas to prevent blocking during initialization"""
        try:
            import pandas as pd
            return pd
        except Exception as e:
            print(f"Warning: Pandas import failed: {e}")
            return None
    
    def _lazy_import_numpy(self):
        """Lazy import numpy to prevent blocking during initialization"""
        try:
            import numpy as np
            return np
        except Exception as e:
            print(f"Warning: Numpy import failed: {e}")
            return None
    
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
                                       spec: Any) -> Dict[str, Any]:
        """
        Generate LLM-powered cost visualization with lazy imports.
        """
        try:
            # Lazy import plotly
            go, make_subplots = self._lazy_import_plotly()
            if go is None:
                return {
                    "figure": None,
                    "insights": ["Plotly unavailable - using LLM insights"],
                    "recommendations": ["Install plotly for visualizations"]
                }
            
            # Generate LLM insights
            insights = self._generate_llm_insights(estimate, spec)
            
            # Safely get spec name and estimate cost
            spec_name = spec.get("name", "Project") if hasattr(spec, 'get') else getattr(spec, 'name', 'Project')
            monthly_cost = estimate.get('monthly_cost', 1000) if hasattr(estimate, 'get') else getattr(estimate, 'monthly_cost', 1000)
            
            # Create dynamic chart structure based on project characteristics
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
                
                # Storage breakdown
                storage_breakdown = ['Hot Data', 'Warm Data', 'Cold Data']
                storage_costs = [monthly_cost * 0.6, monthly_cost * 0.3, monthly_cost * 0.1]
                fig.add_trace(
                    go.Pie(labels=storage_breakdown, values=storage_costs,
                           marker_colors=['#d62728', '#ff7f0e', '#2ca02c']),
                    row=2, col=1
                )
                
                # Training vs Inference
                fig.add_trace(
                    go.Scatter(x=months, y=inference_costs, mode='lines+markers',
                              name='Inference Costs', line=dict(color='#2ca02c', width=3)),
                    row=2, col=2
                )
                
            elif spec.workload.inference_qps > 500:  # High-traffic web app
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        f'High-Traffic Web App - {spec_name}',
                        'Traffic Patterns',
                        'Cost by Service',
                        'Scalability Metrics'
                    ),
                    specs=[[{"type": "scatter"}, {"type": "bar"}],
                           [{"type": "pie"}, {"type": "scatter"}]]
                )
                
                # Web app specific visualizations
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
                traffic_costs = [monthly_cost * (1.0 + 0.5 * (i/5)) for i in range(6)]
                
                # Traffic cost trend
                fig.add_trace(
                    go.Scatter(x=months, y=traffic_costs, mode='lines+markers',
                              name='Traffic-Based Costs', line=dict(color='#d62728', width=3)),
                    row=1, col=1
                )
                
                # Service breakdown
                services = ['Compute', 'Database', 'CDN', 'Storage']
                service_costs = [monthly_cost * 0.4, monthly_cost * 0.3, monthly_cost * 0.2, monthly_cost * 0.1]
                fig.add_trace(
                    go.Bar(x=services, y=service_costs, name='Cost by Service',
                           marker_color='#1f77b4'),
                    row=1, col=2
                )
                
                # Availability metrics
                availability = ['99.9%', '99.95%', '99.99%']
                availability_costs = [monthly_cost * 0.8, monthly_cost * 1.0, monthly_cost * 1.3]
                fig.add_trace(
                    go.Pie(labels=availability, values=availability_costs,
                           marker_colors=['#ff7f0e', '#2ca02c', '#d62728']),
                    row=2, col=1
                )
                
                # Scalability trend
                scalability_costs = [monthly_cost * (1.0 + 0.3 * i) for i in range(6)]
                fig.add_trace(
                    go.Scatter(x=months, y=scalability_costs, mode='lines+markers',
                              name='Scalability Costs', line=dict(color='#9467bd', width=3)),
                    row=2, col=2
                )
                
            else:  # Data warehouse or other
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        f'Data Warehouse Analysis - {spec_name}',
                        'Data Growth Impact',
                        'Storage Tier Breakdown',
                        'Query Performance Costs'
                    ),
                    specs=[[{"type": "scatter"}, {"type": "bar"}],
                           [{"type": "pie"}, {"type": "scatter"}]]
                )
                
                # Data warehouse specific visualizations
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
                data_costs = [monthly_cost * (1.0 + 0.2 * (i/5)) for i in range(6)]
                
                # Data growth trend
                fig.add_trace(
                    go.Scatter(x=months, y=data_costs, mode='lines+markers',
                              name='Data Growth Costs', line=dict(color='#2ca02c', width=3)),
                    row=1, col=1
                )
                
                # Storage tiers
                tiers = ['Hot Storage', 'Warm Storage', 'Cold Storage', 'Archive']
                tier_costs = [monthly_cost * 0.5, monthly_cost * 0.3, monthly_cost * 0.15, monthly_cost * 0.05]
                fig.add_trace(
                    go.Bar(x=tiers, y=tier_costs, name='Storage Tier Costs',
                           marker_color='#1f77b4'),
                    row=1, col=2
                )
                
                # Query performance
                query_types = ['Fast Queries', 'Standard Queries', 'Slow Queries']
                query_costs = [monthly_cost * 0.6, monthly_cost * 0.3, monthly_cost * 0.1]
                fig.add_trace(
                    go.Pie(labels=query_types, values=query_costs,
                           marker_colors=['#d62728', '#ff7f0e', '#2ca02c']),
                    row=2, col=1
                )
                
                # Performance trend
                performance_costs = [monthly_cost * (0.9 + 0.1 * i) for i in range(6)]
                fig.add_trace(
                    go.Scatter(x=months, y=performance_costs, mode='lines+markers',
                              name='Performance Costs', line=dict(color='#9467bd', width=3)),
                    row=2, col=2
                )
            
            fig.update_layout(
                title=f"Intelligent Cost Analysis Dashboard - {spec_name}",
                height=600,
                showlegend=True,
                template="plotly_white"
            )
            
            # Use LLM to generate visualization code
            viz_code = self._generate_llm_visualization_code(estimate, spec, historical_data, forecast_data)
            
            # Execute the LLM-generated code
            try:
                enhanced_fig = self._execute_llm_visualization_code(viz_code, estimate, spec, historical_data, forecast_data)
                if enhanced_fig:
                    fig = enhanced_fig
            except Exception as e:
                print(f"LLM visualization code execution failed: {e}")
                # Continue with default visualization
            
            return {
                "figure": fig,
                "insights": insights,
                "recommendations": self._generate_llm_recommendations(estimate, spec),
                "llm_code": viz_code
            }
            
        except Exception as e:
            print(f"Visualization generation failed: {e}")
            return {
                "figure": None,
                "insights": ["Visualization generation failed"],
                "recommendations": ["Check system configuration"]
            }
    
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
            if go is None:
                return {
                    "figure": None,
                    "total_savings": 0,
                    "savings_percentage": 0
                }
            
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
            print(f"Optimization visualization failed: {e}")
            return {
                "figure": None,
                "total_savings": 0,
                "savings_percentage": 0
            }
    
    def generate_risk_visualization(self, 
                                 risk_findings: List[Dict[str, Any]],
                                 blueprint: Any,
                                 estimate: Any) -> Dict[str, Any]:
        """
        Generate LLM-powered risk visualization.
        """
        try:
            go, make_subplots = self._lazy_import_plotly()
            if go is None:
                return {
                    "figure": None,
                    "total_risks": len(risk_findings) if risk_findings else 0,
                    "critical_risks": 0
                }
            
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
            print(f"Risk visualization failed: {e}")
            return {
                "figure": None,
                "total_risks": len(risk_findings) if risk_findings else 0,
                "critical_risks": 0
            }


    def _generate_llm_visualization_code(self, estimate: Any, spec: Any, 
                                        historical_data: Any, forecast_data: Any) -> str:
        """Generate visualization code using LLM."""
        try:
            client = get_openai_client()
            if client is None:
                return "# LLM unavailable - using default visualization"
            
            # Safely get values
            spec_name = spec.get('name', 'Project') if hasattr(spec, 'get') else getattr(spec, 'name', 'Project')
            monthly_cost = estimate.get('monthly_cost', 1000) if hasattr(estimate, 'get') else getattr(estimate, 'monthly_cost', 1000)
            
            # Create LangChain prompt for code generation
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an expert data visualization engineer specializing in Plotly charts.
                Generate Python code to create insightful visualizations for cloud cost analysis.
                
                IMPORTANT: Return ONLY valid Python code that can be executed. No explanations or markdown."""),
                ("human", """Generate Python code for a cost visualization dashboard:

Project: {spec_name}
Monthly Cost: ${monthly_cost}
Historical Data Columns: {historical_columns}
Forecast Data Columns: {forecast_columns}

Create a comprehensive visualization that shows:
1. Cost trends over time
2. Cost breakdown by service
3. Optimization opportunities
4. Risk indicators

Use plotly.graph_objects (go) and plotly.subplots (make_subplots).
Return ONLY the Python code.""")
            ])
            
            # Create output parser
            parser = JsonOutputParser(pydantic_object=VisualizationInsightsSchema)
            
            # Create the chain
            chain = prompt_template | client | parser
            
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
            
            # Extract code from insights
            if isinstance(result, dict) and 'insights' in result:
                return '\n'.join(result['insights'])
            else:
                return "# LLM code generation failed - using default visualization"
                
        except Exception as e:
            print(f"LLM code generation failed: {e}")
            return "# LLM code generation failed - using default visualization"
    
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

# Backward compatibility function
def generate_cost_visualization(historical_data: Any, 
                               forecast_data: Any,
                               estimate: Any,
                               spec: Any) -> Dict[str, Any]:
    """Legacy function for backward compatibility."""
    agent = IntelligentVisualizationAgent()
    return agent.generate_cost_trend_visualization(historical_data, forecast_data, estimate, spec)