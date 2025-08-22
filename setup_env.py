#!/usr/bin/env python3
"""
Environment setup and dependency check for FinOps Planner
"""

import sys
import os

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'pandas', 
        'plotly',
        'openai',
        'yaml',
        'pydantic'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️ Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    else:
        print("\n🎉 All dependencies available!")
        return True

def check_environment():
    """Check environment variables and configuration"""
    print("\n🔍 Checking environment...")
    
    # Check OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print("✅ OPENAI_API_KEY found")
    else:
        print("❌ OPENAI_API_KEY not found")
        print("   Set with: export OPENAI_API_KEY='your-key-here'")
    
    # Check current directory
    print(f"📁 Current directory: {os.getcwd()}")
    
    # Check if planner directory exists
    planner_dir = os.path.join(os.getcwd(), 'planner')
    if os.path.exists(planner_dir):
        print("✅ planner/ directory found")
    else:
        print("❌ planner/ directory not found")
    
    # Check if settings.yaml exists
    settings_file = os.path.join(os.getcwd(), 'settings.yaml')
    if os.path.exists(settings_file):
        print("✅ settings.yaml found")
    else:
        print("❌ settings.yaml not found")
    
    # Check if data directory exists
    data_dir = os.path.join(os.getcwd(), 'data')
    if os.path.exists(data_dir):
        print("✅ data/ directory found")
    else:
        print("❌ data/ directory not found")

def test_imports():
    """Test if we can import the planner modules"""
    print("\n🔍 Testing imports...")
    
    try:
        sys.path.append(os.path.join(os.getcwd(), 'planner'))
        from planner.planner import PlannerService
        print("✅ PlannerService imported successfully")
        
        # Try to create an instance
        try:
            planner = PlannerService()
            print("✅ PlannerService instantiated successfully")
        except Exception as e:
            print(f"❌ PlannerService instantiation failed: {e}")
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    return True

def main():
    """Run all checks"""
    print("🚀 FinOps Planner - Environment Check")
    print("=" * 50)
    
    deps_ok = check_dependencies()
    check_environment()
    imports_ok = test_imports()
    
    print("\n" + "=" * 50)
    if deps_ok and imports_ok:
        print("🎉 Environment is ready! You can run the app.")
        print("\n🚀 Next steps:")
        print("   1. Set OPENAI_API_KEY if not already set")
        print("   2. Run: streamlit run app.py")
    else:
        print("⚠️ Environment has issues. Please fix them before running the app.")
    
    return deps_ok and imports_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
