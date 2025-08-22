#!/usr/bin/env python3
"""
Test to verify NO mutex blocking errors occur.
This test specifically addresses the mutex.cc:452 Lock blocking issue.
"""

def test_basic_imports():
    """Test basic imports that should never cause mutex blocking"""
    print("Testing basic imports...")
    try:
        import uuid
        from typing import Dict, Any, Optional
        print("✅ Basic imports successful (no mutex blocking)")
        return True
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_planner_import():
    """Test importing PlannerService without mutex blocking"""
    print("Testing PlannerService import...")
    try:
        from planner.planner import PlannerService
        print("✅ PlannerService import successful (no mutex blocking)")
        return True
    except Exception as e:
        print(f"❌ PlannerService import failed with potential mutex issue: {e}")
        return False

def test_viz_agent_import():
    """Test importing VisualizationAgent without mutex blocking"""
    print("Testing VisualizationAgent import...")
    try:
        from planner.viz_agent import IntelligentVisualizationAgent
        print("✅ VisualizationAgent import successful (no mutex blocking)")
        return True
    except Exception as e:
        print(f"❌ VisualizationAgent import failed with potential mutex issue: {e}")
        return False

def test_service_creation():
    """Test creating service instances without mutex blocking"""
    print("Testing service creation...")
    try:
        from planner.planner import PlannerService
        from planner.viz_agent import IntelligentVisualizationAgent
        
        # These should be instant with no blocking operations
        planner = PlannerService()
        viz_agent = IntelligentVisualizationAgent()
        
        print("✅ Service creation successful (no mutex blocking)")
        return True
    except Exception as e:
        print(f"❌ Service creation failed with potential mutex issue: {e}")
        return False

def test_basic_operations():
    """Test basic operations without mutex blocking"""
    print("Testing basic operations...")
    try:
        from planner.planner import PlannerService
        from planner.viz_agent import IntelligentVisualizationAgent
        
        # Test basic operations that should not block
        planner = PlannerService()
        result = planner.plan("Simple test project")
        
        viz_agent = IntelligentVisualizationAgent()
        viz_result = viz_agent.generate_cost_trend_visualization(None, None, None, None)
        
        print("✅ Basic operations successful (no mutex blocking)")
        print(f"   Planner message: {result.get('message', 'N/A')}")
        print(f"   Viz insights: {viz_result.get('insights', ['N/A'])[0]}")
        return True
    except Exception as e:
        print(f"❌ Basic operations failed with potential mutex issue: {e}")
        return False

def main():
    print("🔒 Mutex Blocking Test - FinOps Planner")
    print("=" * 60)
    print("Testing for mutex.cc:452 Lock blocking errors...")
    print()
    
    tests = [
        test_basic_imports,
        test_planner_import,
        test_viz_agent_import,
        test_service_creation,
        test_basic_operations,
    ]
    
    all_passed = True
    for i, test in enumerate(tests, 1):
        print(f"Test {i}/5:", end=" ")
        if not test():
            all_passed = False
            print("🚨 MUTEX BLOCKING DETECTED - Stopping tests")
            break
        print()
        
    print("-" * 60)
    if all_passed:
        print("🎉 SUCCESS: No mutex blocking detected!")
        print("✅ All tests passed - safe to run Streamlit app")
        print("\n💡 You can now try: streamlit run app.py")
        return True
    else:
        print("❌ FAILURE: Potential mutex blocking detected")
        print("🔧 Need to investigate and fix blocking operations")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
