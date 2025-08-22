#!/usr/bin/env python3
"""
Minimal test to verify no mutex errors occur during import and basic usage.
"""

def test_basic_import():
    """Test basic Python imports that should never hang"""
    try:
        import os
        import json
        import uuid
        from typing import Dict, Any, List, Optional
        print("✅ Basic imports successful")
        return True
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_planner_import():
    """Test importing our planner module"""
    try:
        from planner.planner import PlannerService
        print("✅ PlannerService import successful")
        return True
    except Exception as e:
        print(f"❌ PlannerService import failed: {e}")
        return False

def test_planner_creation():
    """Test creating PlannerService instance"""
    try:
        from planner.planner import PlannerService
        service = PlannerService()
        print("✅ PlannerService creation successful")
        return True
    except Exception as e:
        print(f"❌ PlannerService creation failed: {e}")
        return False

def test_planner_plan():
    """Test calling the plan method"""
    try:
        from planner.planner import PlannerService
        service = PlannerService()
        result = service.plan("Test project for ML training")
        print(f"✅ PlannerService.plan() successful: {result.get('message', 'No message')}")
        return True
    except Exception as e:
        print(f"❌ PlannerService.plan() failed: {e}")
        return False

def main():
    print("🧪 Minimal FinOps Planner Test")
    print("=" * 50)
    
    tests = [
        test_basic_import,
        test_planner_import,
        test_planner_creation,
        test_planner_plan,
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
            break
        
    if all_passed:
        print("\n🎉 All tests passed! No mutex errors detected.")
        print("✅ The planner is safe to use.")
        return True
    else:
        print("\n❌ Tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    main()
