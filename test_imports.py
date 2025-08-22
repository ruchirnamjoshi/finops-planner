#!/usr/bin/env python3
"""
🧪 FinOps Planner Import Test
==================================================
This script tests that all the core modules can be imported without errors.
Run this before starting the Streamlit app to catch import issues early.
"""

import sys
import os

def test_basic_imports():
    """Test basic Python imports that don't involve pandas/NumPy"""
    print("🧪 Testing basic imports...")
    
    try:
        # Test basic Python modules
        import json
        import yaml
        import logging
        print("  ✅ Basic modules imported successfully")
        return True
    except ImportError as e:
        print(f"  ❌ Basic import failed: {e}")
        return False

def test_schema_imports():
    """Test schema imports"""
    print("🧪 Testing schema imports...")
    
    try:
        from planner.schemas import (
            ProjectSpec, Workload, DataSpec, Constraints,
            Blueprint, Estimate, OptimizationResult, RiskFinding
        )
        print("  ✅ Schemas imported successfully")
        return True
    except ImportError as e:
        print(f"  ❌ Schema import failed: {e}")
        return False

def test_config_imports():
    """Test configuration imports"""
    print("🧪 Testing config imports...")
    
    try:
        from planner.config import load_settings
        print("  ✅ Config imported successfully")
        return True
    except ImportError as e:
        print(f"  ❌ Config import failed: {e}")
        return False

def test_agent_imports():
    """Test agent imports without initializing them"""
    print("🧪 Testing agent imports...")
    
    try:
        # Import the classes without instantiating
        from planner.blueprint_bot import IntelligentBlueprintAgent
        print("  ✅ Blueprint agent imported successfully")
        
        from planner.cost_engine import IntelligentCostEngineAgent
        print("  ✅ Cost engine agent imported successfully")
        
        from planner.optimizer_bot import IntelligentCostOptimizerAgent
        print("  ✅ Optimizer agent imported successfully")
        
        from planner.risk_bot import IntelligentRiskAssessmentAgent
        print("  ✅ Risk assessment agent imported successfully")
        
        from planner.viz_agent import IntelligentVisualizationAgent
        print("  ✅ Visualization agent imported successfully")
        
        return True
    except ImportError as e:
        print(f"  ❌ Agent import failed: {e}")
        return False

def test_planner_service_import():
    """Test planner service import without initialization"""
    print("🧪 Testing planner service import...")
    
    try:
        # Import the class without instantiating
        from planner.planner import PlannerService
        print("  ✅ Planner service imported successfully")
        return True
    except ImportError as e:
        print(f"  ❌ Planner service import failed: {e}")
        return False

def test_imports():
    """Run all import tests"""
    print("🧪 FinOps Planner Import Test")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_schema_imports,
        test_config_imports,
        test_agent_imports,
        test_planner_service_import,
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    if all_passed:
        print("\n🎉 All imports successful!")
        print("\n💡 You can now try running the Streamlit app:")
        print("   streamlit run app.py")
        return True
    else:
        print("\n❌ Some imports failed. Please fix the errors above.")
        return False

if __name__ == "__main__":
    if not test_imports():
        sys.exit(1)
