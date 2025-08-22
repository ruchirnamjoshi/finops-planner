#!/usr/bin/env python3
"""
Simple test to check basic functionality
"""

import sys
import os

def test_basic_imports():
    """Test basic imports"""
    print("🧪 Testing basic imports...")
    
    try:
        import streamlit as st
        print("✅ streamlit imported")
    except Exception as e:
        print(f"❌ streamlit import failed: {e}")
    
    try:
        import pandas as pd
        print("✅ pandas imported")
    except Exception as e:
        print(f"❌ pandas import failed: {e}")
    
    try:
        import plotly.express as px
        print("✅ plotly imported")
    except Exception as e:
        print(f"❌ plotly import failed: {e}")

def test_planner_imports():
    """Test planner imports"""
    print("\n🧪 Testing planner imports...")
    
    try:
        # Add planner to path
        sys.path.append(os.path.join(os.getcwd(), 'planner'))
        
        from planner.schemas import ProjectSpec, Blueprint, Estimate
        print("✅ schemas imported")
        
        from planner.blueprint_bot import IntelligentBlueprintAgent
        print("✅ blueprint_bot imported")
        
        from planner.cost_engine import IntelligentCostEngineAgent
        print("✅ cost_engine imported")
        
        from planner.optimizer_bot import IntelligentCostOptimizerAgent
        print("✅ optimizer_bot imported")
        
        from planner.risk_bot import IntelligentRiskAssessmentAgent
        print("✅ risk_bot imported")
        
    except Exception as e:
        print(f"❌ planner import failed: {e}")
        return False
    
    return True

def test_planner_service():
    """Test planner service creation"""
    print("\n🧪 Testing planner service...")
    
    try:
        from planner.planner import PlannerService
        print("✅ PlannerService imported")
        
        # Try to create instance
        planner = PlannerService()
        print("✅ PlannerService created")
        
        return True
        
    except Exception as e:
        print(f"❌ PlannerService failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 FinOps Planner - Simple Test")
    print("=" * 40)
    
    test_basic_imports()
    planner_imports_ok = test_planner_imports()
    planner_service_ok = test_planner_service()
    
    print("\n" + "=" * 40)
    if planner_imports_ok and planner_service_ok:
        print("🎉 Basic functionality is working!")
        print("You can now run: streamlit run app.py")
    else:
        print("⚠️ There are issues with the planner modules.")
    
    return planner_imports_ok and planner_service_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
