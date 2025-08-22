#!/usr/bin/env python3
"""
Test script to verify LLM usage in agents.
"""

def test_llm_capabilities():
    """Test if LLM calls are working in our agents"""
    print("🧪 Testing LLM Capabilities in FinOps Agents")
    print("=" * 60)
    
    try:
        # Load environment variables first
        print("Loading environment variables...")
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✅ Environment variables loaded from .env file")
        except ImportError:
            print("❌ python-dotenv not installed")
            return False
        except Exception as e:
            print(f"❌ Failed to load .env file: {e}")
            return False
        
        # Test 1: Check if OpenAI client can be created
        print("\nTest 1: OpenAI Client Creation")
        from openai import OpenAI
        import os
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY environment variable is not set")
            return False
        
        print(f"✅ API key found: {api_key[:20]}...")
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client created successfully")
        
        # Test 2: Check if we can make a simple API call
        print("\nTest 2: Simple API Call")
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=50,
                messages=[{"role": "user", "content": "Say 'Hello World' in one word"}]
            )
            result = response.choices[0].message.content.strip()
            print(f"✅ API call successful: {result}")
        except Exception as e:
            print(f"❌ API call failed: {e}")
            return False
            
        # Test 3: Test our planner with LLM
        print("\nTest 3: Planner LLM Usage")
        from planner.planner import PlannerService
        
        planner = PlannerService()
        
        # Test with different inputs
        test_inputs = [
            "ML training project with 4 GPUs for computer vision",
            "Web application with high availability requirements",
            "Data warehouse for analytics with 1TB storage"
        ]
        
        for i, test_input in enumerate(test_inputs, 1):
            print(f"\n  Input {i}: {test_input}")
            try:
                result = planner.plan(test_input)
                
                if result.get("error"):
                    print(f"    ❌ Planning failed: {result['error']}")
                else:
                    spec_name = result.get("spec", {}).get("name", "Unknown")
                    candidates = len(result.get("candidates", []))
                    optimized = len(result.get("optimized", []))
                    risks = len(result.get("risks", {}))
                    
                    print(f"    ✅ Planning successful:")
                    print(f"      - Project: {spec_name}")
                    print(f"      - Blueprints: {candidates}")
                    print(f"      - Optimizations: {optimized}")
                    print(f"      - Risk areas: {risks}")
                    
                    # Check if results are different
                    if i == 1:
                        first_result = result
                    else:
                        if result == first_result:
                            print(f"    ⚠️  WARNING: Result {i} is identical to first result!")
                        else:
                            print(f"    ✅ Result {i} is different from first result")
                            
            except Exception as e:
                print(f"    ❌ Planning error: {e}")
        
        print("\n🎉 LLM capability test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_llm_capabilities()
    exit(0 if success else 1)
