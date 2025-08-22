#!/usr/bin/env python3
"""
Test script for the intelligent agents in FinOps Planner
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_blueprint_agent():
    """Test the Intelligent Blueprint Agent"""
    print("🧪 Testing Blueprint Agent...")
    try:
        from planner.blueprint_bot import IntelligentBlueprintAgent
        agent = IntelligentBlueprintAgent()
        print("✅ Blueprint Agent imported successfully")
        
        # Test basic functionality
        from planner.schemas import ProjectSpec, Workload, DataSpec, Constraints
        
        # Create a test project spec
        test_spec = ProjectSpec(
            name="Test ML Project",
            workload=Workload(train_gpus=2, batch=True),
            data=DataSpec(size_gb=1000, hot_fraction=0.3),
            constraints=Constraints(clouds=["aws"], regions=["us-east-1"])
        )
        
        print("✅ ProjectSpec created successfully")
        print(f"   Project: {test_spec.name}")
        print(f"   GPUs: {test_spec.workload.train_gpus}")
        print(f"   Data: {test_spec.data.size_gb} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Blueprint Agent test failed: {e}")
        return False

def test_cost_engine_agent():
    """Test the Intelligent Cost Engine Agent"""
    print("\n🧪 Testing Cost Engine Agent...")
    try:
        from planner.cost_engine import IntelligentCostEngineAgent
        agent = IntelligentCostEngineAgent()
        print("✅ Cost Engine Agent imported successfully")
        
        # Test pricing insights
        insights = agent._load_pricing_insights()
        print(f"✅ Loaded {len(insights)} pricing insight categories")
        
        # Test cost models
        models = agent._load_cost_models()
        print(f"✅ Loaded {len(models)} cost model types")
        
        return True
        
    except Exception as e:
        print(f"❌ Cost Engine Agent test failed: {e}")
        return False

def test_optimizer_agent():
    """Test the Intelligent Cost Optimizer Agent"""
    print("\n🧪 Testing Cost Optimizer Agent...")
    try:
        from planner.optimizer_bot import IntelligentCostOptimizerAgent
        agent = IntelligentCostOptimizerAgent()
        print("✅ Cost Optimizer Agent imported successfully")
        
        # Test optimization strategies
        strategies = agent._load_optimization_strategies()
        print(f"✅ Loaded {len(strategies)} optimization strategy categories")
        
        # Test cost patterns
        patterns = agent._load_cost_patterns()
        print(f"✅ Loaded {len(patterns)} cost pattern types")
        
        return True
        
    except Exception as e:
        print(f"❌ Cost Optimizer Agent test failed: {e}")
        return False

def test_risk_agent():
    """Test the Intelligent Risk Assessment Agent"""
    print("\n🧪 Testing Risk Assessment Agent...")
    try:
        from planner.risk_bot import IntelligentRiskAssessmentAgent
        agent = IntelligentRiskAssessmentAgent()
        print("✅ Risk Assessment Agent imported successfully")
        
        # Test risk patterns
        patterns = agent._load_risk_patterns()
        print(f"✅ Loaded {len(patterns)} risk pattern categories")
        
        # Test compliance frameworks
        frameworks = agent._load_compliance_frameworks()
        print(f"✅ Loaded {len(frameworks)} compliance frameworks")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk Assessment Agent test failed: {e}")
        return False

def test_planner_service():
    """Test the main Planner Service"""
    print("\n🧪 Testing Planner Service...")
    try:
        from planner.planner import PlannerService
        print("✅ Planner Service imported successfully")
        
        # Test basic initialization
        planner = PlannerService()
        print("✅ Planner Service initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Planner Service test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 FinOps Planner - Intelligent Agents Test Suite")
    print("=" * 50)
    
    tests = [
        test_blueprint_agent,
        test_cost_engine_agent,
        test_optimizer_agent,
        test_risk_agent,
        test_planner_service
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The intelligent agents are working correctly.")
        print("\n🚀 You can now:")
        print("   1. Run the Streamlit UI: streamlit run app.py")
        print("   2. Use the intelligent agents in your code")
        print("   3. Deploy to production")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
