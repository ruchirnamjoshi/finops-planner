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
    """Get or create the visualization agent."""
    if 'viz_agent' not in st.session_state:
        try:
            from planner.viz_agent import IntelligentVisualizationAgent
            st.session_state.viz_agent = IntelligentVisualizationAgent()
        except Exception as e:
            st.error(f"Failed to initialize VisualizationAgent: {e}")
            return None
    return st.session_state.viz_agent

# Import the insights agent
from planner.insights_agent import IntelligentInsightsAgent

# Import the strategy comparison agent
from planner.strategy_comparison_agent import IntelligentStrategyComparisonAgent

def get_insights_agent():
    """Get or create the insights agent."""
    if 'insights_agent' not in st.session_state:
        try:
            st.session_state.insights_agent = IntelligentInsightsAgent()
        except Exception as e:
            st.error(f"Failed to initialize insights agent: {e}")
            return None
    return st.session_state.insights_agent

def get_strategy_comparison_agent():
    """Get or create the strategy comparison agent."""
    if 'strategy_comparison_agent' not in st.session_state:
        try:
            st.session_state.strategy_comparison_agent = IntelligentStrategyComparisonAgent()
        except Exception as e:
            st.error(f"Failed to initialize strategy comparison agent: {e}")
            return None
    return st.session_state.strategy_comparison_agent

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
    forecast_months = st.slider("Forecast Period (months)", 3, 12, 6, key="forecast_months")
    optimization_aggressiveness = st.selectbox(
        "Optimization Strategy",
        ["Conservative", "Balanced", "Aggressive"],
        index=1,
        key="optimization_aggressiveness"
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
st.markdown('<h2 class="sub-header">🎯 Generate Intelligent Plan</h2>', unsafe_allow_html=True)

# Plan generation and management
col1, col2 = st.columns([3, 1])
with col1:
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
                        # Pass the original project_brief string, not the spec object
                        result = planner.plan(project_brief)
                        
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
    if st.button("🗑️ Clear Results", type="secondary", use_container_width=True):
        if 'planning_result' in st.session_state:
            del st.session_state.planning_result
        st.rerun()

# Project Details Section
st.markdown('<h3 class="sub-header">📋 Project Details</h3>', unsafe_allow_html=True)

if 'planning_result' in st.session_state:
    result = st.session_state.planning_result
    if not result.get("error"):
        spec = result.get("spec", {})
        
        # Use columns for better layout of project details
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
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
        
        with col2:
            st.markdown("**Data Size:**")
            data = spec.get("data", {})
            st.info(f"{data.get('size_gb', 0):.1f} GB")
            
            st.markdown("**Data Growth:**")
            st.info(f"{data.get('growth_gb_per_month', 0):.1f} GB/month")
        
        with col3:
            st.markdown("**Cloud Provider:**")
            constraints = spec.get("constraints", {})
            clouds = constraints.get("clouds", [])
            st.info(", ".join(clouds) if clouds else "Not specified")
            
            st.markdown("**Regions:**")
            regions = constraints.get("regions", [])
            st.info(", ".join(regions) if regions else "Not specified")
        
        with col4:
            st.markdown("**Latency Requirements:**")
            workload = spec.get("workload", {})
            latency = workload.get("latency_ms", 100)
            if latency < 50:
                st.success(f"Ultra-low: {latency}ms")
            elif latency < 100:
                st.info(f"Low: {latency}ms")
            else:
                st.warning(f"Standard: {latency}ms")
            
            st.markdown("**Batch Processing:**")
            batch = workload.get("batch", False)
            st.info("✅ Enabled" if batch else "❌ Disabled")

# Display Results
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
                
                # Display each blueprint with its own detailed analysis
                if result.get('candidates') and len(result['candidates']) > 0:
                    st.markdown('<h2 class="sub-header">🏗️ Architecture Blueprint Analysis</h2>', unsafe_allow_html=True)
                    
                    # Display each blueprint with its own detailed analysis
                    for i, (blueprint, estimate) in enumerate(result['candidates']):
                        if estimate is None:
                            continue
                            
                        # Create a unique section for each blueprint with side-by-side layout
                        st.markdown(f"### 🏗️ Blueprint {i+1}: {blueprint.id.upper()}")
                        
                        # Create two main columns: Left for blueprint details, Right for cost breakdown
                        col_left, col_right = st.columns([1, 1])
                        
                        with col_left:
                            st.markdown("#### 📋 Blueprint Details")
                            # Blueprint details in sub-columns
                            sub_col1, sub_col2 = st.columns(2)
                            with sub_col1:
                                st.info(f"**Cloud:** {blueprint.cloud.upper()}")
                                st.info(f"**Region:** {blueprint.region}")
                                st.info(f"**Monthly Cost:** ${estimate.monthly_cost:,.2f}")
                            with sub_col2:
                                st.info(f"**Services:** {len(blueprint.services) if hasattr(blueprint, 'services') else 0}")
                                # Show workload-specific insights
                                if result.get('spec_obj'):
                                    spec = result['spec_obj']
                                    if spec.workload.train_gpus > 0:
                                        st.success(f"**🎯 ML Training:** {spec.workload.train_gpus} GPUs")
                                    elif spec.workload.inference_qps > 500:
                                        st.success(f"**🌐 High Traffic:** {spec.workload.inference_qps:.0f} QPS")
                                    elif spec.data.size_gb > 5000:
                                        st.success(f"**💾 Data Heavy:** {spec.data.size_gb:,.0f} GB")
                                    else:
                                        st.success(f"**⚡ General Compute:** {spec.workload.inference_qps:.0f} QPS")
                            
                            # Show blueprint services
                            if hasattr(blueprint, 'services') and blueprint.services:
                                st.markdown("**🔧 Services:**")
                                for service in blueprint.services:
                                    service_name = service.get('service', 'unknown') if hasattr(service, 'get') else getattr(service, 'service', 'unknown')
                                    service_sku = service.get('sku', 'N/A') if hasattr(service, 'get') else getattr(service, 'sku', 'N/A')
                                    service_qty = service.get('qty_expr', '1') if hasattr(service, 'get') else getattr(service, 'qty_expr', '1')
                                    st.info(f"• {service_name}: {service_sku} x {service_qty}")
                            
                            # Show data insights
                            if result.get('spec_obj'):
                                spec = result['spec_obj']
                                st.info(f"**📊 Data:** {spec.data.size_gb:,.0f} GB")
                                st.info(f"**📈 Growth:** {spec.data.growth_gb_per_month:,.0f} GB/month")
                        
                        with col_right:
                            st.markdown("#### 💰 Cost Breakdown & Visualization")
                            
                            # Show cost breakdown for this specific blueprint
                            if hasattr(estimate, 'bom') and estimate.bom:
                                st.markdown("**📊 Cost Summary:**")
                                total_cost = sum(item.cost for item in estimate.bom)
                                for item in estimate.bom:
                                    percentage = (item.cost / total_cost * 100) if total_cost > 0 else 0
                                    st.info(f"• {item.service}: ${item.cost:,.2f} ({percentage:.1f}%)")
                                
                                # Cost visualization below the breakdown - Side by side for better visibility
                                bom_data = []
                                for item in estimate.bom:
                                    bom_data.append({
                                        'Service': item.service,
                                        'Cost': item.cost
                                    })
                                
                                df_bom = pd.DataFrame(bom_data)
                                
                                # Single pie chart for cost distribution (full width)
                                fig_pie = px.pie(
                                    df_bom, 
                                    values='Cost', 
                                    names='Service',
                                    title=f"Cost Distribution - {blueprint.id.upper()}",
                                    color_discrete_sequence=px.colors.qualitative.Set3
                                )
                                fig_pie.update_layout(height=400)
                                st.plotly_chart(fig_pie, use_container_width=True)
                            else:
                                st.info("Cost breakdown details not available")
                        
                        st.markdown("---")
                        
                        # Individual AI Insights for this blueprint
                        st.markdown(f"#### 🤖 AI-Powered Analysis for {blueprint.id.upper()}")
                
                        # Get optimization recommendations for this specific blueprint
                        optimization_recs = []
                        if result.get('optimized') and len(result['optimized']) > i and result['optimized'][i]:
                            opt_result = result['optimized'][i]
                            if hasattr(opt_result, 'metadata') and opt_result.metadata:
                                if 'optimization_recommendations' in opt_result.metadata:
                                    optimization_recs = opt_result.metadata['optimization_recommendations']
                        
                        # Get cost patterns for this specific blueprint
                        cost_patterns = {}
                        if result.get('optimized') and len(result['optimized']) > i and result['optimized'][i]:
                            opt_result = result['optimized'][i]
                            if hasattr(opt_result, 'metadata') and opt_result.metadata:
                                if 'cost_patterns' in opt_result.metadata:
                                    cost_patterns = opt_result.metadata['cost_patterns']
                        
                        # Generate comprehensive insights for this specific blueprint
                        insights_agent = get_insights_agent()
                        if insights_agent and result.get('spec_obj'):
                            try:
                                comprehensive_insights = insights_agent.generate_comprehensive_insights(
                                    spec=result['spec_obj'],
                                    estimate=estimate,
                                    cost_patterns=cost_patterns,
                                    optimization_recommendations=optimization_recs
                                )
                                
                                # Display blueprint-specific insights
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**💰 Cost Analysis:**")
                                    for insight in comprehensive_insights.get('cost_analysis', []):
                                        st.info(f"• {insight}")
                                    
                                    st.markdown("**🔧 Resource Optimization:**")
                                    for rec in comprehensive_insights.get('resource_optimization', []):
                                        st.success(f"• {rec}")
                                
                                with col2:
                                    st.markdown("**💡 Cost Savings Opportunities:**")
                                    for opportunity in comprehensive_insights.get('cost_savings', []):
                                        st.warning(f"• {opportunity}")
                                    
                                    st.markdown("**⚠️ Risk Mitigation:**")
                                    for strategy in comprehensive_insights.get('risk_mitigation', []):
                                        st.error(f"• {strategy}")
                                
                                # Performance and strategic insights
                                st.markdown("**📊 Performance Insights:**")
                                for insight in comprehensive_insights.get('performance_insights', []):
                                    st.info(f"• {insight}")
                                
                                st.markdown("**🎯 Strategic Recommendations:**")
                                for rec in comprehensive_insights.get('strategic_recommendations', []):
                                    st.success(f"• {rec}")
                                
                            except Exception as e:
                                st.error(f"❌ AI insights generation failed for {blueprint.id}: {e}")
                                st.error("The system requires LLM connectivity to provide intelligent analysis.")
                        
                        # Display cost patterns for this specific blueprint
                        if cost_patterns:
                            st.markdown(f"#### 📊 Cost Pattern Analysis for {blueprint.id.upper()}")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("💰 Compute Ratio", f"{cost_patterns.get('compute_ratio', 0):.1f}%")
                                st.metric("💾 Storage Ratio", f"{cost_patterns.get('storage_ratio', 0):.1f}%")
                                st.metric("🌐 Network Ratio", f"{cost_patterns.get('network_ratio', 0):.1f}%")
                            
                            with col2:
                                # Dominant cost driver
                                dominant = cost_patterns.get('dominant_cost_driver', 'unknown')
                                if dominant == 'compute':
                                    st.warning(f"🔥 **Dominant Cost Driver**: Compute ({cost_patterns.get('compute_ratio', 0):.1f}%)")
                                elif dominant == 'storage':
                                    st.warning(f"💾 **Dominant Cost Driver**: Storage ({cost_patterns.get('storage_ratio', 0):.1f}%)")
                                elif dominant == 'network':
                                    st.warning(f"🌐 **Dominant Cost Driver**: Network ({cost_patterns.get('network_ratio', 0):.1f}%)")
                                else:
                                    st.info(f"📊 **Dominant Cost Driver**: {dominant}")
                                
                                # Cost efficiency score
                                efficiency = cost_patterns.get('cost_efficiency_score', 0)
                                if efficiency > 80:
                                    st.success(f"✅ **Cost Efficiency**: Excellent ({efficiency:.0f}/100)")
                                elif efficiency > 60:
                                    st.success(f"✅ **Cost Efficiency**: Good ({efficiency:.0f}/100)")
                                elif efficiency > 40:
                                    st.warning(f"⚠️ **Cost Efficiency**: Fair ({efficiency:.0f}/100)")
                                else:
                                    st.error(f"❌ **Cost Efficiency**: Poor ({efficiency:.0f}/100)")
                            
                            # Optimization priority
                            priority = cost_patterns.get('optimization_priority', 'overall')
                            st.info(f"🎯 **Optimization Priority**: Focus on {priority.replace('|', ', ')}")
                            
                            # High cost indicators
                            if cost_patterns.get('high_compute', False):
                                st.warning("⚠️ **High Compute Costs**: Consider spot instances, reserved instances, or auto-scaling")
                            if cost_patterns.get('high_storage', False):
                                st.warning("⚠️ **High Storage Costs**: Implement lifecycle policies, compression, or tiered storage")
                            if cost_patterns.get('high_network', False):
                                st.warning("⚠️ **High Network Costs**: Optimize data transfer, use CDN, or co-locate resources")
                        
                        # Display optimization recommendations for this specific blueprint
                        if optimization_recs:
                            st.markdown(f"#### 🚀 Optimization Recommendations for {blueprint.id.upper()}")
                            
                            for j, rec in enumerate(optimization_recs, 1):
                                if isinstance(rec, dict):
                                    st.success(f"**{j}. {rec.get('strategy', 'N/A')}**")
                                    st.success(f"   Description: {rec.get('description', 'N/A')}")
                                    st.success(f"   Savings: {rec.get('savings_potential', 'N/A')}")
                                    st.success(f"   Effort: {rec.get('implementation_effort', 'N/A')}")
                                else:
                                    st.success(f"**{j}. {rec}**")
                        
                        # Display risk assessment for this specific blueprint
                        if result.get('risks') and blueprint.id in result['risks']:
                            risks = result['risks'][blueprint.id]
                            if risks:
                                st.markdown(f"#### ⚠️ Risk Assessment for {blueprint.id.upper()}")
                                
                                if isinstance(risks, list):
                                    for k, risk in enumerate(risks, 1):
                                        if hasattr(risk, 'category'):
                                            st.warning(f"**Risk {k}:**")
                                            st.warning(f"• **Category**: {risk.category}")
                                            st.warning(f"• **Severity**: {risk.severity}")
                                            st.warning(f"• **Evidence**: {risk.evidence}")
                                            st.warning(f"• **Fix**: {risk.fix}")
                                        else:
                                            st.warning(f"• {risk}")
                                else:
                                    if hasattr(risks, 'category'):
                                        st.warning(f"**Risk:**")
                                        st.warning(f"• **Category**: {risks.category}")
                                        st.warning(f"• **Severity**: {risks.severity}")
                                        st.warning(f"• **Evidence**: {risks.evidence}")
                                        st.warning(f"• **Fix**: {risks.fix}")
                                    else:
                                        st.warning(f"• {risks}")
                        
                        # Compact spacing between blueprints
                        st.markdown("---")
        else:
            st.info("No blueprints generated. This might be due to missing API keys or configuration.")
        
        # Strategy Comparison and Recommendation
        st.markdown('<h3 class="sub-header">🏆 Strategy Comparison & Recommendation</h3>', unsafe_allow_html=True)
        
        strategy_agent = get_strategy_comparison_agent()
        if strategy_agent and result.get('spec_obj'):
            try:
                with st.spinner("🤖 Analyzing strategies and generating recommendations..."):
                    # Get all the data needed for comparison
                    candidates = result.get('candidates', [])
                    optimizations = result.get('optimized', [])
                    risks = result.get('risks', {})
                    
                    # Get cost patterns from optimizations if available
                    cost_patterns = {}
                    if optimizations:
                        for opt in optimizations:
                            if opt and hasattr(opt, 'metadata') and opt.metadata:
                                if 'cost_patterns' in opt.metadata:
                                    cost_patterns.update(opt.metadata['cost_patterns'])
                    
                    # Generate comprehensive strategy comparison
                    comparison_result = strategy_agent.compare_strategies(
                        spec=result['spec_obj'],
                        candidates=candidates,
                        optimizations=optimizations,
                        risks=risks,
                        cost_patterns=cost_patterns
                    )
                    
                    # Display the winner
                    if isinstance(comparison_result, dict):
                        winner_id = comparison_result.get('winner_blueprint_id', 'N/A')
                        winner_reason = comparison_result.get('winner_reason', 'N/A')
                        comparison_matrix = comparison_result.get('comparison_matrix', {})
                        ranking = comparison_result.get('ranking', [])
                        cost_analysis = comparison_result.get('cost_analysis', {})
                        performance_analysis = comparison_result.get('performance_analysis', {})
                        risk_analysis = comparison_result.get('risk_analysis', {})
                        strategic_recommendations = comparison_result.get('strategic_recommendations', [])
                        implementation_roadmap = comparison_result.get('implementation_roadmap', {})
                    else:
                        winner_id = comparison_result.winner_blueprint_id
                        winner_reason = comparison_result.winner_reason
                        comparison_matrix = comparison_result.comparison_matrix
                        ranking = comparison_result.ranking
                        cost_analysis = comparison_result.cost_analysis
                        performance_analysis = comparison_result.performance_analysis
                        risk_analysis = comparison_result.risk_analysis
                        strategic_recommendations = comparison_result.strategic_recommendations
                        implementation_roadmap = comparison_result.implementation_roadmap
                    
                    # Find the highest scoring blueprint
                    highest_scoring = max(
                        comparison_matrix.items(),
                        key=lambda x: x[1].get('overall_score', 0)
                    )
                    winner_id = highest_scoring[0]
                    winner_score = highest_scoring[1].get('overall_score', 0)
                    
                    st.success(f"🏆 **Recommended Strategy: {winner_id.upper()}** (Score: {winner_score}/100)")
                    st.info(f"**Reason:** Highest overall score based on cost, performance, risk, scalability, and implementation factors")
                    
                    # Display comparison matrix
                    st.markdown("#### 📊 Strategy Comparison Matrix")
                    
                    # Create comparison table
                    comparison_data = []
                    for blueprint_id, comparison in comparison_matrix.items():
                        comparison_data.append({
                            "Blueprint": blueprint_id.upper(),
                            "Overall Score": f"{comparison.get('overall_score', 'N/A')}/100",
                            "Cost Score": f"{comparison.get('cost_score', 'N/A')}/100",
                            "Performance Score": f"{comparison.get('performance_score', 'N/A')}/100",
                            "Risk Score": f"{comparison.get('risk_score', 'N/A')}/100",
                            "Scalability Score": f"{comparison.get('scalability_score', 'N/A')}/100",
                            "Implementation Score": f"{comparison.get('implementation_score', 'N/A')}/100"
                        })
                    
                    import pandas as pd
                    df_comparison = pd.DataFrame(comparison_data)
                    st.dataframe(df_comparison, use_container_width=True)
                    
                    # Display ranking based on overall scores
                    st.markdown("#### 🏅 Strategy Ranking (Based on Overall Scores)")
                    
                    # Sort blueprints by overall score (highest first)
                    sorted_blueprints = sorted(
                        comparison_matrix.items(),
                        key=lambda x: x[1].get('overall_score', 0),
                        reverse=True
                    )
                    
                    for i, (blueprint_id, comparison) in enumerate(sorted_blueprints, 1):
                        overall_score = comparison.get('overall_score', 0)
                        if i == 1:
                            st.success(f"🥇 **{i}. {blueprint_id.upper()}** - Best Overall Strategy (Score: {overall_score}/100)")
                        elif i == 2:
                            st.info(f"🥈 **{i}. {blueprint_id.upper()}** - Strong Alternative (Score: {overall_score}/100)")
                        else:
                            st.warning(f"🥉 **{i}. {blueprint_id.upper()}** - Consider for specific use cases (Score: {overall_score}/100)")
                    
                    # Display detailed analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 💰 Cost Analysis")
                        st.info(f"**Cost Efficiency:** {cost_analysis.get('cost_efficiency', 'N/A')}")
                        st.info(f"**Optimization Potential:** {cost_analysis.get('optimization_potential', 'N/A')}")
                        st.info(f"**Long-term Trends:** {cost_analysis.get('long_term_cost_trends', 'N/A')}")
                        
                        st.markdown("#### ⚡ Performance Analysis")
                        st.info(f"**Scalability:** {performance_analysis.get('scalability', 'N/A')}")
                        st.info(f"**Latency Analysis:** {performance_analysis.get('latency_analysis', 'N/A')}")
                        st.info(f"**Resource Utilization:** {performance_analysis.get('resource_utilization', 'N/A')}")
                    
                    with col2:
                        st.markdown("#### ⚠️ Risk Analysis")
                        st.info(f"**Overall Risk:** {risk_analysis.get('overall_risk_assessment', 'N/A')}")
                        st.info(f"**Compliance:** {risk_analysis.get('compliance_considerations', 'N/A')}")
                        
                        st.markdown("#### 🎯 Strategic Recommendations")
                        for i, rec in enumerate(strategic_recommendations, 1):
                            st.success(f"**{i}.** {rec}")
                    
                    # Display implementation roadmap
                    st.markdown("#### 🗺️ Implementation Roadmap")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"**Phase 1 (0-30 days):** {implementation_roadmap.get('phase_1', 'N/A')}")
                    with col2:
                        st.info(f"**Phase 2 (1-3 months):** {implementation_roadmap.get('phase_2', 'N/A')}")
                    with col3:
                        st.info(f"**Phase 3 (3-12 months):** {implementation_roadmap.get('phase_3', 'N/A')}")
                    
                    st.markdown("**Key Milestones:**")
                    milestones = implementation_roadmap.get('key_milestones', [])
                    for milestone in milestones:
                        st.info(f"• {milestone}")
                    
                    st.info(f"**Resource Requirements:** {implementation_roadmap.get('resource_requirements', 'N/A')}")
                    
            except Exception as e:
                st.error(f"❌ Strategy comparison failed: {e}")
                st.info("💡 This might be due to missing API keys or configuration.")
        else:
            st.warning("Strategy comparison agent not available")
            if not strategy_agent:
                st.error("❌ Strategy comparison agent failed to initialize")
            if not result.get('spec_obj'):
                st.error("❌ Project specification not available")
        
        # Advanced Cost Analytics with Forecasting
        st.markdown('<h3 class="sub-header">📈 Advanced Cost Analytics & Forecasting</h3>', unsafe_allow_html=True)
        
        # Get forecast settings from sidebar
        forecast_months = st.session_state.get('forecast_months', 6)
        optimization_aggressiveness = st.session_state.get('optimization_aggressiveness', 'Balanced')
        
        # Display forecast configuration
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**📅 Forecast Period:** {forecast_months} months")
        with col2:
            st.info(f"**🎯 Optimization Strategy:** {optimization_aggressiveness}")
        with col3:
            if result.get('candidates') and len(result['candidates']) > 0:
                total_monthly_cost = sum(est.monthly_cost for _, est in result['candidates'] if est)
                projected_yearly = total_monthly_cost * 12
                st.info(f"**💰 Projected Yearly:** ${projected_yearly:,.2f}")
        
        viz_agent = get_viz_agent()
        if viz_agent and result.get('spec_obj'):
            try:
                # Generate cost trend visualization for the first estimate
                if result.get('candidates') and len(result['candidates']) > 0:
                    first_estimate = result['candidates'][0][1]
                    
                    # Create enhanced visualization with forecasting
                    st.markdown("#### 🔮 Cost Trend Analysis & Forecasting")
                    
                    viz_result = viz_agent.generate_cost_trend_visualization(
                        historical_data=None,  # Will generate synthetic data
                        forecast_data=None,    # Will generate synthetic data
                        estimate=first_estimate,
                        spec=result['spec_obj'],  # Use the actual ProjectSpec object
                        forecast_months=forecast_months  # Pass forecast period directly
                    )
                    
                    # Display the visualization
                    st.plotly_chart(viz_result['figure'], use_container_width=True)
                    
                    # Enhanced insights with forecasting context
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🔍 Key Insights:**")
                        for insight in viz_result.get('insights', []):
                            st.info(insight)
                        
                        # Add forecast-specific insights
                        if forecast_months > 6:
                            st.success(f"**📈 Long-term Trend:** {forecast_months}-month forecast shows cost evolution patterns")
                        if optimization_aggressiveness == 'Aggressive':
                            st.warning("**⚡ Aggressive Optimization:** Higher savings potential but increased implementation complexity")
                    
                    with col2:
                        st.markdown("**💡 Recommendations:**")
                        for rec in viz_result.get('recommendations', []):
                            st.success(rec)
                        
                        # Add forecast-specific recommendations
                        if forecast_months >= 12:
                            st.info("**🎯 Annual Planning:** Consider reserved instances for predictable workloads")
                        if optimization_aggressiveness == 'Conservative':
                            st.info("**🛡️ Conservative Approach:** Lower risk, gradual optimization over time")
                    
                    # LLM-Powered Dynamic Forecasting Analysis
                    st.markdown("#### 📊 LLM-Powered Forecasting Analysis")
                    
                    # Generate intelligent forecasting using LLM
                    try:
                        with st.spinner("🤖 Generating intelligent cost forecasts..."):
                            forecast_analysis = viz_agent.generate_intelligent_forecasting(
                                estimate=first_estimate,
                                spec=result['spec_obj'],
                                forecast_months=forecast_months,
                                optimization_strategy=optimization_aggressiveness,
                                cost_patterns=cost_patterns,
                                risks=result.get('risks', {})
                            )
                        
                        # Display dynamic forecasting results
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            # Total projected cost over forecast period
                            total_projected_cost = forecast_analysis.get('total_projected_cost', 0)
                            final_month_cost = forecast_analysis.get('final_month_cost', 0)
                            st.metric(
                                f"Total Projected Cost ({forecast_months} months)", 
                                f"${total_projected_cost:,.0f}", 
                                f"Final month: ${final_month_cost:,.0f}"
                            )
                        
                        with col2:
                            # Dynamic optimization potential
                            projected_savings = forecast_analysis.get('projected_savings', 0)
                            savings_potential = forecast_analysis.get('savings_potential', 0)
                            st.metric(
                                "Potential Savings", 
                                f"${projected_savings:,.0f}", 
                                f"{savings_potential:.1f}% monthly"
                            )
                        
                        with col3:
                            # Dynamic risk assessment
                            risk_score = forecast_analysis.get('risk_score', 0)
                            risk_trend = forecast_analysis.get('risk_trend', 'stable')
                            st.metric(
                                "Risk Score", 
                                f"{risk_score:.1f}/10", 
                                f"Trend: {risk_trend}"
                            )
                        
                        with col4:
                            # Average monthly cost and growth rate
                            average_monthly_cost = forecast_analysis.get('average_monthly_cost', 0)
                            growth_rate = forecast_analysis.get('growth_rate', 0)
                            st.metric(
                                "Average Monthly Cost", 
                                f"${average_monthly_cost:,.0f}", 
                                f"Growth: {growth_rate:.1f}% monthly"
                            )
                        
                        # Display LLM-generated insights
                        st.markdown("#### 🔮 LLM Forecasting Insights")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**📈 Growth Factors:**")
                            growth_factors = forecast_analysis.get('growth_factors', [])
                            for factor in growth_factors:
                                st.info(f"• {factor}")
                        
                        with col2:
                            st.markdown("**💰 Optimization Insights:**")
                            optimization_insights = forecast_analysis.get('optimization_insights', [])
                            for insight in optimization_insights:
                                st.success(f"• {insight}")
                        
                        # Display trend analysis
                        if forecast_analysis.get('trend_analysis'):
                            st.markdown("#### 📊 Trend Analysis")
                            trend_data = forecast_analysis['trend_analysis']
                            if trend_data and 'costs' in trend_data:
                                # Create dynamic trend chart
                                import plotly.graph_objects as go
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=list(range(1, forecast_months + 1)),
                                    y=trend_data['costs'],
                                    mode='lines+markers',
                                    name='Projected Costs',
                                    line=dict(color='#1f77b4', width=3)
                                ))
                                
                                if 'optimized_costs' in trend_data:
                                    fig.add_trace(go.Scatter(
                                        x=list(range(1, forecast_months + 1)),
                                        y=trend_data['optimized_costs'],
                                        mode='lines+markers',
                                        name='Optimized Costs',
                                        line=dict(color='#2ca02c', width=3, dash='dash')
                                    ))
                                
                                fig.update_layout(
                                    title=f"LLM-Powered Cost Forecasting ({forecast_months} months)",
                                    xaxis_title="Months",
                                    yaxis_title="Monthly Cost ($)",
                                    height=400,
                                    template="plotly_white"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("❌ LLM forecasting failed: Missing trend data")
                                st.error("The LLM did not provide complete trend analysis. Please try regenerating the forecast.")
                    
                    except Exception as e:
                        st.error(f"❌ LLM forecasting failed: {e}")
                        st.error("No fallback available. The system requires LLM-powered forecasting to work properly.")
                        st.info("💡 Please check your API configuration and try again.")
                        
                else:
                    st.info("No estimates available for visualization")
                    
            except Exception as e:
                st.warning(f"Visualization generation failed: {e}")
                st.error(f"Error details: {str(e)}")
        else:
            st.warning("Visualization agent not available")
            st.info("💡 This might be due to missing API keys or configuration.")
        
        # Summary and Next Steps
        st.markdown('<h3 class="sub-header">📋 Summary and Next Steps</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Key Findings:**")
            if result.get('candidates') and len(result['candidates']) > 0:
                st.success(f"• Generated {len(result['candidates'])} architecture blueprints")
                st.success(f"• Cost estimates range from ${min(est.monthly_cost for _, est in result['candidates'] if est):,.2f} to ${max(est.monthly_cost for _, est in result['candidates'] if est):,.2f}")
                st.success(f"• Identified {len(result.get('risks', {}))} risk areas")
                st.success(f"• Generated {len(result.get('optimized', []))} optimization strategies")
        
        with col2:
            st.markdown("**🚀 Next Steps:**")
            st.info("• Review and compare the proposed blueprints")
            st.info("• Analyze cost optimization opportunities")
            st.info("• Assess and mitigate identified risks")
            st.info("• Implement the chosen architecture")
        
        st.markdown("---")
        st.markdown("**💡 Pro Tip:** Use the 'Clear Results' button above to start a new analysis with different requirements.")
        
    else:
        st.info("No planning results available. Please generate a plan first.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🚀 <strong>FinOps Planner</strong> - AI-Powered Cloud Cost Optimization</p>
    <p>Built with intelligent agents for real-world cloud infrastructure planning</p>
</div>
""", unsafe_allow_html=True)