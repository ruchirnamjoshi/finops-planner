import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
import sys
import re

# Add the planner directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'planner'))

# Page configuration
st.set_page_config(
    page_title="FinOps Planner - AI-Powered Cloud Cost Optimization",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize planner service
def get_planner_service():
    try:
        from planner.planner import PlannerService
        return PlannerService()
    except Exception as e:
        st.error(f"Failed to initialize PlannerService: {e}")
        return None

# Initialize visualization agent
def get_viz_agent():
    try:
        from planner.viz_agent import IntelligentVisualizationAgent
        return IntelligentVisualizationAgent()
    except Exception as e:
        st.error(f"Failed to initialize VisualizationAgent: {e}")
        return None

def create_project_spec_from_brief(brief: str) -> 'ProjectSpec':
    """Intelligently create a ProjectSpec object from a natural language brief."""
    from planner.schemas import ProjectSpec, Workload, DataSpec, Constraints
    
    # Default values
    workload = Workload(
        train_gpus=0,
        inference_qps=100,
        latency_ms=100,
        batch=False
    )
    
    data = DataSpec(
        size_gb=100,
        growth_gb_per_month=10,
        hot_fraction=0.7,
        egress_gb_per_month=100
    )
    
    constraints = Constraints(
        clouds=["aws"],
        regions=["us-east-1"]
    )
    
    # Parse the brief for specific requirements
    brief_lower = brief.lower()
    
    # ML Training workloads
    if any(word in brief_lower for word in ["ml", "machine learning", "training", "gpu", "neural", "ai"]):
        if "gpu" in brief_lower or "training" in brief_lower:
            gpu_count = 1
            if "4" in brief or "four" in brief:
                gpu_count = 4
            elif "8" in brief or "eight" in brief:
                gpu_count = 8
            elif "16" in brief or "sixteen" in brief:
                gpu_count = 16
            
            workload = Workload(
                train_gpus=gpu_count,
                inference_qps=50,
                latency_ms=200,
                batch=True
            )
            data = DataSpec(
                size_gb=1000,
                growth_gb_per_month=50,
                hot_fraction=0.6,
                egress_gb_per_month=200
            )
            constraints = Constraints(
                clouds=["aws", "gcp"],
                regions=["us-east-1", "us-central1"]
            )
    
    # High-traffic web applications
    elif any(word in brief_lower for word in ["web", "app", "application", "high traffic", "scalable", "ha", "high availability"]):
        workload = Workload(
            train_gpus=0,
            inference_qps=1000,
            latency_ms=50,
            batch=False
        )
        data = DataSpec(
            size_gb=100,
            growth_gb_per_month=20,
            hot_fraction=0.8,
            egress_gb_per_month=500
        )
        constraints = Constraints(
            clouds=["aws", "azure"],
            regions=["us-east-1", "eastus"]
        )
    
    # Data warehouse/analytics
    elif any(word in brief_lower for word in ["data", "warehouse", "analytics", "big data", "etl", "reporting"]):
        workload = Workload(
            train_gpus=0,
            inference_qps=100,
            latency_ms=300,
            batch=True
        )
        data = DataSpec(
            size_gb=10000,
            growth_gb_per_month=100,
            hot_fraction=0.3,
            egress_gb_per_month=1000
        )
        constraints = Constraints(
            clouds=["aws", "gcp"],
            regions=["us-east-1", "us-central1"]
        )
    
    # Extract project name
    project_name = brief.split()[0] if brief.split() else "Project"
    if len(project_name) > 20:
        project_name = project_name[:20]
    
    return ProjectSpec(
        name=project_name,
        workload=workload,
        data=data,
        constraints=constraints
    )

# Main header
st.markdown('<h1 class="main-header">🚀 FinOps Planner</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Cloud Infrastructure Planning & Cost Optimization</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Project Configuration")
    
    # Project brief input
    project_brief = st.text_area(
        "Describe your project requirements:",
        height=150,
        placeholder="e.g., I need to train a vision transformer with 4 GPUs, 2TB data, max cost $5000/month on AWS us-east-1"
    )
    
    st.markdown("### 📊 Analysis Options")
    include_optimization = st.checkbox("Include Cost Optimization", value=True)
    include_risk_assessment = st.checkbox("Include Risk Assessment", value=True)
    include_forecasting = st.checkbox("Include Cost Forecasting", value=True)
    
    st.markdown("### 🔧 Advanced Settings")
    forecast_months = st.slider("Forecast Period (months)", 3, 12, 6)
    optimization_aggressiveness = st.selectbox(
        "Optimization Strategy",
        ["Conservative", "Balanced", "Aggressive"],
        index=1
    )
    
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    if 'planning_result' in st.session_state:
        result = st.session_state.planning_result
        if not result.get("error"):
            st.metric("Monthly Cost", f"${result.get('monthly_cost', 0):,.2f}")
            st.metric("Optimization Savings", f"${result.get('optimization_savings', 0):,.2f}")
            st.metric("Risk Score", f"{result.get('risk_score', 0):.1f}/10")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h2 class="sub-header">🎯 Generate Intelligent Plan</h2>', unsafe_allow_html=True)
    
    # Plan generation
    if st.button("🎯 Generate Intelligent Plan", type="primary", use_container_width=True):
        if project_brief.strip():
            # Clear any previous results to ensure fresh planning
            if 'planning_result' in st.session_state:
                del st.session_state.planning_result
            
            with st.spinner("🤖 AI agents are analyzing your project..."):
                try:
                    # Get the planner service
                    planner = get_planner_service()
                    if planner is None:
                        st.error("❌ Failed to initialize intelligent agents. Please check the configuration.")
                    else:
                        # Create a ProjectSpec object based on the project brief
                        from planner.schemas import ProjectSpec, Workload, DataSpec, Constraints
                        
                        # Parse the project brief to create intelligent specifications
                        spec = create_project_spec_from_brief(project_brief)
                        
                        # Use the intelligent agents to generate a real plan
                        result = planner.plan(spec)
                        
                        if result.get("error"):
                            st.error(f"❌ Planning failed: {result['error']}")
                        else:
                            # Add timestamp and unique identifier to ensure fresh results
                            result['generated_at'] = datetime.now().isoformat()
                            result['unique_id'] = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            st.session_state.planning_result = result
                            st.success("✅ Intelligent plan generated successfully!")
                            st.rerun()
                            
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 This might be due to missing API keys or configuration. Check the setup instructions.")
        else:
            st.warning("⚠️ Please enter a project description")

with col2:
    st.markdown('<h3 class="sub-header">📋 Project Details</h3>', unsafe_allow_html=True)
    
    if 'planning_result' in st.session_state:
        result = st.session_state.planning_result
        if not result.get("error"):
            spec = result.get("spec", {})
            
            st.markdown("**Project Name:**")
            st.info(spec.get("name", "N/A"))
            
            st.markdown("**Workload Type:**")
            workload = spec.get("workload", {})
            if workload.get("train_gpus", 0) > 0:
                st.success(f"ML Training: {workload.get('train_gpus')} GPUs")
            elif workload.get("inference_qps", 0) > 0:
                st.info(f"Inference: {workload.get('inference_qps')} QPS")
            else:
                st.info("General Compute")
            
            st.markdown("**Data Size:**")
            data = spec.get("data", {})
            st.info(f"{data.get('size_gb', 0):.1f} GB")
            
            st.markdown("**Cloud Provider:**")
            constraints = spec.get("constraints", {})
            clouds = constraints.get("clouds", [])
            st.info(", ".join(clouds) if clouds else "Not specified")

# Display results
if 'planning_result' in st.session_state:
    result = st.session_state.planning_result
    
    if result.get("error"):
        st.error(f"❌ Planning failed: {result['error']}")
    else:
        # Success! Display the intelligent plan
        st.markdown("---")
        
        # Add clear results button and timestamp
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown('<h2 class="sub-header">🎉 Intelligent Plan Generated!</h2>', unsafe_allow_html=True)
        with col2:
            if st.button("🗑️ Clear Results", type="secondary"):
                del st.session_state.planning_result
                st.rerun()
        
        # Show when the plan was generated
        st.info(f"📅 Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Key metrics in cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Project Status</h3>
                <h2>✅ Active</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🏗️ Blueprints</h3>
                <h2>{len(result.get('candidates', []))}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>⚡ Optimizations</h3>
                <h2>{len(result.get('optimized', []))}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>⚠️ Risk Areas</h3>
                <h2>{len(result.get('risks', {}))}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Blueprint Analysis
        st.markdown('<h3 class="sub-header">🏗️ Architecture Blueprints</h3>', unsafe_allow_html=True)
        
        if result.get('candidates') and len(result['candidates']) > 0:
            # Display all candidate blueprints
            for i, (blueprint, estimate) in enumerate(result['candidates']):
                if blueprint and estimate:
                    st.markdown(f"**Blueprint {i+1}:**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Handle both dict and object access
                        bp_id = blueprint.get('id', 'N/A') if hasattr(blueprint, 'get') else getattr(blueprint, 'id', 'N/A')
                        bp_cloud = blueprint.get('cloud', 'N/A') if hasattr(blueprint, 'get') else getattr(blueprint, 'cloud', 'N/A')
                        bp_region = blueprint.get('region', 'N/A') if hasattr(blueprint, 'get') else getattr(blueprint, 'region', 'N/A')
                        
                        st.info(f"**ID:** {bp_id}")
                        st.info(f"**Cloud:** {bp_cloud}")
                        st.info(f"**Region:** {bp_region}")
                    
                    with col2:
                        # Handle both dict and object access
                        if hasattr(estimate, 'monthly_cost'):
                            monthly_cost = estimate.monthly_cost
                        elif hasattr(estimate, 'get'):
                            monthly_cost = estimate.get('monthly_cost', 'N/A')
                        else:
                            monthly_cost = 'N/A'
                            
                        st.info(f"**Monthly Cost:** ${monthly_cost:,.2f}" if isinstance(monthly_cost, (int, float)) else f"**Monthly Cost:** {monthly_cost}")
                    
                    st.markdown("---")
        else:
            st.info("No blueprints generated. This might be due to missing API keys or configuration.")
        
        # Cost Breakdown
        st.markdown('<h3 class="sub-header">💰 Cost Breakdown</h3>', unsafe_allow_html=True)
        
        if result.get('candidates') and len(result['candidates']) > 0:
            # Get the first estimate for cost breakdown
            _, first_estimate = result['candidates'][0]
            
            if first_estimate and hasattr(first_estimate, 'bom') and first_estimate.bom:
                bom_data = []
                for item in first_estimate.bom:
                    bom_data.append({
                        'Service': item.service,
                        'Cost': item.cost
                    })
                
                df_bom = pd.DataFrame(bom_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    fig_pie = px.pie(
                        df_bom, 
                        values='Cost', 
                        names='Service',
                        title="Cost Distribution by Service",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Bar chart
                    fig_bar = px.bar(
                        df_bom,
                        x='Service',
                        y='Cost',
                        title="Cost by Service",
                        color='Service',
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_bar.update_layout(height=400)
                    st.plotly_chart(fig_bar, use_container_width=True)
        
        # Display LLM insights from all agents
        st.markdown('<h3 class="sub-header">🤖 AI-Generated Insights</h3>', unsafe_allow_html=True)
        
        if result.get('candidates') and len(result['candidates']) > 0:
            first_estimate = result['candidates'][0][1]
            
            # Display Cost Engine LLM Insights
            if hasattr(first_estimate, 'metadata') and first_estimate.metadata:
                metadata = first_estimate.metadata
                
                if 'llm_insights' in metadata and metadata['llm_insights']:
                    st.markdown("**💰 Cost Engine AI Insights:**")
                    llm_insights = metadata['llm_insights']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'cost_breakdown_analysis' in llm_insights:
                            breakdown = llm_insights['cost_breakdown_analysis']
                            st.markdown("**📊 Cost Breakdown Analysis:**")
                            st.info(f"• Compute: {breakdown.get('compute_percentage', 'N/A')}%")
                            st.info(f"• Storage: {breakdown.get('storage_percentage', 'N/A')}%")
                            st.info(f"• Network: {breakdown.get('network_percentage', 'N/A')}%")
                            st.info(f"• Dominant Driver: {breakdown.get('dominant_cost_driver', 'N/A')}")
                        
                        if 'optimization_opportunities' in llm_insights:
                            st.markdown("**🚀 Optimization Opportunities:**")
                            for opp in llm_insights['optimization_opportunities'][:3]:  # Show top 3
                                st.success(f"• **{opp.get('category', 'N/A')}**: {opp.get('description', 'N/A')}")
                                st.success(f"  - Potential Savings: {opp.get('potential_savings', 'N/A')}")
                                st.success(f"  - Effort: {opp.get('effort_required', 'N/A')}")
                    
                    with col2:
                        if 'pricing_insights' in llm_insights:
                            st.markdown("**💡 Pricing Insights:**")
                            for insight in llm_insights['pricing_insights'][:2]:  # Show top 2
                                st.info(f"• **{insight.get('insight', 'N/A')}**")
                                st.info(f"  - Impact: {insight.get('impact', 'N/A')}")
                                st.info(f"  - Action: {insight.get('action', 'N/A')}")
                        
                        if 'cost_forecast' in llm_insights:
                            forecast = llm_insights['cost_forecast']
                            st.markdown("**📈 Cost Forecast:**")
                            st.warning(f"• Trend: {forecast.get('trend', 'N/A')}")
                            if 'factors' in forecast:
                                for factor in forecast['factors'][:2]:
                                    st.warning(f"• Factor: {factor}")
                            if 'recommendations' in forecast:
                                for rec in forecast['recommendations'][:2]:
                                    st.warning(f"• Recommendation: {rec}")
                
                # Display LLM cost optimization details
                if 'llm_cost_optimization' in metadata and metadata['llm_cost_optimization']:
                    st.success(f"**🎯 LLM Cost Optimization Applied:** {metadata['llm_cost_optimization']}")
                
                if 'llm_cost_forecast' in metadata and metadata['llm_cost_forecast']:
                    st.warning(f"**📊 LLM Cost Forecast Applied:** {metadata['llm_cost_forecast']}")
                
                # Display cost analysis metadata
                if 'cost_breakdown' in metadata:
                    st.markdown("**📊 Detailed Cost Breakdown:**")
                    breakdown = metadata['cost_breakdown']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Compute", f"${breakdown.get('compute', 0):,.2f}")
                    with col2:
                        st.metric("Storage", f"${breakdown.get('storage', 0):,.2f}")
                    with col3:
                        st.metric("Network", f"${breakdown.get('network', 0):,.2f}")
        
        # Display Optimization Results with LLM insights
        if result.get('optimized') and len(result['optimized']) > 0:
            st.markdown("**🚀 LLM-Powered Optimization Results:**")
            for i, opt_result in enumerate(result['optimized'][:3]):  # Show top 3
                st.markdown(f"**Optimization {i+1}:**")
                
                if hasattr(opt_result, 'metadata') and opt_result.metadata:
                    metadata = opt_result.metadata
                    
                    # Show LLM recommendations
                    if 'llm_recommendations' in metadata and metadata['llm_recommendations']:
                        st.markdown("**🤖 LLM Recommendations:**")
                        for rec in metadata['llm_recommendations'][:3]:
                            if isinstance(rec, dict):
                                st.success(f"• **{rec.get('strategy', 'N/A')}**: {rec.get('description', 'N/A')}")
                                st.success(f"  - Savings: {rec.get('savings_potential', 'N/A')}")
                                st.success(f"  - Effort: {rec.get('implementation_effort', 'N/A')}")
                            else:
                                st.success(f"• {rec}")
                    
                    # Show cost patterns
                    if 'cost_patterns' in metadata:
                        st.markdown("**📊 Cost Patterns:**")
                        patterns = metadata['cost_patterns']
                        st.json(patterns)
                
                st.markdown("---")
        
        # Display Risk Assessment Results with LLM insights
        if result.get('risks') and len(result['risks']) > 0:
            st.markdown("**⚠️ LLM-Powered Risk Assessment:**")
            risks_list = list(result['risks']) if hasattr(result['risks'], '__iter__') else [result['risks']]
            for i, risk in enumerate(risks_list[:3]):  # Show top 3
                if hasattr(risk, 'category'):
                    st.markdown(f"**Risk {i+1}:**")
                    st.warning(f"• **Category**: {risk.category}")
                    st.warning(f"• **Severity**: {risk.severity}")
                    st.warning(f"• **Evidence**: {risk.evidence}")
                    st.warning(f"• **Fix**: {risk.fix}")
                else:
                    st.warning(f"• {risk}")
                st.markdown("---")
        
        # Advanced Visualizations
        if include_forecasting:
            st.markdown('<h3 class="sub-header">📈 Advanced Cost Analytics</h3>', unsafe_allow_html=True)
            
            viz_agent = get_viz_agent()
            if viz_agent and 'estimate' in result and 'spec' in result:
                try:
                    # Generate cost trend visualization
                    viz_result = viz_agent.generate_cost_trend_visualization(
                        historical_data=None,  # Will generate synthetic data
                        forecast_data=None,    # Will generate synthetic data
                        estimate=result['estimate'],
                        spec=result['spec']
                    )
                    
                    # Display the visualization
                    st.plotly_chart(viz_result['figure'], use_container_width=True)
                    
                    # Display insights
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🔍 Key Insights:**")
                        for insight in viz_result.get('insights', []):
                            st.info(insight)
                    
                    with col2:
                        st.markdown("**💡 Recommendations:**")
                        for rec in viz_result.get('recommendations', []):
                            st.success(rec)
                            
                except Exception as e:
                    st.warning(f"Visualization generation failed: {e}")
        
        # LLM-Powered Insights and Visualizations
        st.markdown('<h3 class="sub-header">🤖 AI-Generated Insights</h3>', unsafe_allow_html=True)
        
        # Get visualization agent
        viz_agent = get_viz_agent()
        if viz_agent:
            try:
                # Generate cost trend visualization
                spec = result.get('spec', {})
                first_estimate = result['candidates'][0][1] if result.get('candidates') else None
                
                if first_estimate:
                    viz_result = viz_agent.generate_cost_trend_visualization(
                        historical_data=None,  # We don't have historical data yet
                        forecast_data=None,     # We don't have forecast data yet
                        estimate=first_estimate,
                        spec=spec
                    )
                    
                    # Display the visualization if available
                    if viz_result.get('figure'):
                        st.plotly_chart(viz_result['figure'], use_container_width=True)
                    
                    # Display insights
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🔍 Key Insights:**")
                        for insight in viz_result.get('insights', []):
                            st.info(insight)
                    
                    with col2:
                        st.markdown("**💡 Recommendations:**")
                        for rec in viz_result.get('recommendations', []):
                            st.success(rec)
                else:
                    st.info("No estimates available for visualization")
                    
            except Exception as e:
                st.warning(f"Visualization generation failed: {e}")
        else:
            st.warning("Visualization agent not available")
        
        # Cost Optimization Results
        st.markdown('<h3 class="sub-header">⚡ Cost Optimization</h3>', unsafe_allow_html=True)
        
        if result.get('optimized') and len(result['optimized']) > 0:
            for i, opt in enumerate(result['optimized']):
                if opt:
                    st.markdown(f"**Optimization {i+1}:**")
                    
                    # Handle both dict and object access
                    if hasattr(opt, 'monthly_cost'):
                        monthly_cost = opt.monthly_cost
                    elif hasattr(opt, 'get'):
                        monthly_cost = opt.get('monthly_cost', 'N/A')
                    else:
                        monthly_cost = 'N/A'
                    
                    st.info(f"**Monthly Cost:** ${monthly_cost:,.2f}" if isinstance(monthly_cost, (int, float)) else f"**Monthly Cost:** {monthly_cost}")
                    
                    # Display optimization metadata if available
                    if hasattr(opt, 'metadata') and opt.metadata:
                        for key, value in opt.metadata.items():
                            st.info(f"**{key.title()}:** {value}")
                    
                    st.markdown("---")
        else:
            st.info("No optimizations generated. This might be due to missing API keys or configuration.")
        
        # Risk Assessment
        st.markdown('<h3 class="sub-header">⚠️ Risk Assessment</h3>', unsafe_allow_html=True)
        
        if result.get('risks') and len(result['risks']) > 0:
            for blueprint_id, risk_findings in result['risks'].items():
                if risk_findings:
                    st.markdown(f"**Blueprint: {blueprint_id}**")
                    
                    for finding in risk_findings:
                        # Handle both dict and object access
                        if hasattr(finding, 'severity'):
                            severity = finding.severity
                        elif hasattr(finding, 'get'):
                            severity = finding.get('severity', 'unknown')
                        else:
                            severity = 'unknown'
                            
                        if hasattr(finding, 'category'):
                            category = finding.category
                        elif hasattr(finding, 'get'):
                            category = finding.get('category', 'unknown')
                        else:
                            category = 'unknown'
                            
                        if hasattr(finding, 'evidence'):
                            evidence = finding.evidence
                        elif hasattr(finding, 'get'):
                            evidence = finding.get('evidence', 'No evidence')
                        else:
                            evidence = 'No evidence'
                        
                        if severity == 'critical':
                            st.error(f"🚨 **{category.upper()}** - {evidence}")
                        elif severity == 'high':
                            st.warning(f"⚠️ **{category.upper()}** - {evidence}")
                        elif severity == 'medium':
                            st.info(f"ℹ️ **{category.upper()}** - {evidence}")
                        else:
                            st.success(f"✅ **{category.upper()}** - {evidence}")
                        
                        # Display fix if available
                        if hasattr(finding, 'fix'):
                            fix = finding.fix
                        elif hasattr(finding, 'get'):
                            fix = finding.get('fix')
                        else:
                            fix = None
                            
                        if fix:
                            st.markdown(f"**Fix:** {fix}")
                    
                    st.markdown("---")
        else:
            st.info("No risk assessments generated. This might be due to missing API keys or configuration.")
        
        # Cost Forecasting
        if include_forecasting and 'forecast' in result:
            st.markdown('<h3 class="sub-header">🔮 Cost Forecasting</h3>', unsafe_allow_html=True)
            
            forecast = result['forecast']
            if forecast:
                # Create forecast chart
                months = list(range(1, forecast_months + 1))
                costs = [forecast.get('monthly_costs', {}).get(str(m), 0) for m in months]
                
                df_forecast = pd.DataFrame({
                    'Month': [f"Month {m}" for m in months],
                    'Cost': costs
                })
                
                fig_forecast = px.line(
                    df_forecast,
                    x='Month',
                    y='Cost',
                    title=f"{forecast_months}-Month Cost Forecast",
                    markers=True
                )
                fig_forecast.update_layout(height=400)
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Forecast insights
                st.markdown("**📊 Forecast Insights:**")
                st.info(f"**Trend:** {forecast.get('trend', 'Stable')}")
                st.info(f"**Confidence:** {forecast.get('confidence', 'Medium')}")
                st.info(f"**Key Factors:** {', '.join(forecast.get('factors', []))}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🚀 <strong>FinOps Planner</strong> - AI-Powered Cloud Cost Optimization</p>
    <p>Built with intelligent agents for real-world cloud infrastructure planning</p>
</div>
""", unsafe_allow_html=True)