#!/usr/bin/env python3
"""
Debug script to test all LLM agents without Streamlit
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the planner directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'planner'))

def test_imports():
    """Test if all modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from planner.schemas import ProjectSpec, Blueprint, Estimate
        print("✅ Schemas imported successfully")
    except Exception as e:
        print(f"❌ Failed to import schemas: {e}")
        return False
    
    try:
        from planner.config import load_settings
        print("✅ Config imported successfully")
    except Exception as e:
        print(f"❌ Failed to import config: {e}")
        return False
    
    try:
        from planner.blueprint_bot import IntelligentBlueprintAgent
        print("✅ Blueprint agent imported successfully")
    except Exception as e:
        print(f"❌ Failed to import blueprint agent: {e}")
        return False
    
    try:
        from planner.cost_engine import IntelligentCostEngineAgent
        print("✅ Cost engine agent imported successfully")
    except Exception as e:
        print(f"❌ Failed to import cost engine agent: {e}")
        return False
    
    try:
        from planner.optimizer_bot import IntelligentCostOptimizerAgent
        print("✅ Optimizer agent imported successfully")
    except Exception as e:
        print(f"❌ Failed to import optimizer agent: {e}")
        return False
    
    try:
        from planner.risk_bot import IntelligentRiskAssessmentAgent
        print("✅ Risk assessment agent imported successfully")
    except Exception as e:
        print(f"❌ Failed to import risk assessment agent: {e}")
        return False
    
    try:
        from planner.viz_agent import IntelligentVisualizationAgent
        print("✅ Visualization agent imported successfully")
    except Exception as e:
        print(f"❌ Failed to import visualization agent: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading"""
    print("\n🔧 Testing configuration...")
    
    try:
        from planner.config import load_settings
        settings = load_settings()
        print(f"✅ Settings loaded: {settings}")
        return True
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")
        return False

def test_agents():
    """Test agent initialization"""
    print("\n🤖 Testing agent initialization...")
    
    try:
        from planner.blueprint_bot import IntelligentBlueprintAgent
        from planner.cost_engine import IntelligentCostEngineAgent
        from planner.optimizer_bot import IntelligentCostOptimizerAgent
        from planner.risk_bot import IntelligentRiskAssessmentAgent
        from planner.viz_agent import IntelligentVisualizationAgent
        
        # Test blueprint agent
        blueprint_agent = IntelligentBlueprintAgent()
        print("✅ Blueprint agent initialized")
        
        # Test cost engine agent
        cost_agent = IntelligentCostEngineAgent()
        print("✅ Cost engine agent initialized")
        
        # Test optimizer agent
        optimizer_agent = IntelligentCostOptimizerAgent()
        print("✅ Optimizer agent initialized")
        
        # Test risk assessment agent
        risk_agent = IntelligentRiskAssessmentAgent()
        print("✅ Risk assessment agent initialized")
        
        # Test visualization agent
        viz_agent = IntelligentVisualizationAgent()
        print("✅ Visualization agent initialized")
        
        return True
    except Exception as e:
        print(f"❌ Failed to initialize agents: {e}")
        return False

def test_data_files():
    """Test if required data files exist"""
    print("\n📁 Testing data files...")
    
    required_files = [
        "data/price_snapshot.csv",
        "data/history.csv",
        "settings.yaml"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_simple_planning():
    """Test a simple planning operation"""
    print("\n🚀 Testing simple planning...")
    
    try:
        from planner.planner import PlannerService
        
        planner = PlannerService()
        print("✅ Planner service initialized")
        
        # Test with a simple project spec
        spec = {
            "name": "Test Project",
            "description": "A simple test project",
            "budget": 1000,
            "requirements": ["web server", "database"],
            "cloud_preference": "aws"
        }
        
        result = planner.plan_project(spec)
        print(f"✅ Planning completed: {type(result)}")
        
        return True
    except Exception as e:
        print(f"❌ Planning failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 FinOps Planner Debug Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Cannot continue.")
        return
    
    # Test configuration
    if not test_config():
        print("\n❌ Configuration test failed.")
        return
    
    # Test data files
    if not test_data_files():
        print("\n❌ Data file tests failed.")
        return
    
    # Test agent initialization
    if not test_agents():
        print("\n❌ Agent initialization failed.")
        return
    
    # Test simple planning
    if not test_simple_planning():
        print("\n❌ Simple planning test failed.")
        return
    
    print("\n🎉 All tests passed! The FinOps Planner is working correctly.")
    print("\nThe issue is likely with Streamlit, not the core application.")
    print("Try running: streamlit run app.py --server.port 8501")

if __name__ == "__main__":
    main()
