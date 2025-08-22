#!/usr/bin/env python3
"""
Test script to simulate different Streamlit app inputs and verify they generate different results.
"""

import os
import sys
from dotenv import load_dotenv

# Add the planner directory to the path
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), 'planner')))

def test_streamlit_inputs():
    """Test that different Streamlit inputs generate different results."""
    
    print("🧪 Testing Streamlit App Inputs")
    print("=" * 50)
    
    try:
        from planner.schemas import ProjectSpec, Workload, DataSpec, Constraints
        
        # Load environment variables
        load_dotenv()
        
        # Simulate different user inputs (like what they'd type in Streamlit)
        user_inputs = [
            "ML training project with 4 GPUs for computer vision",
            "High-traffic web application with 2000 QPS requirements",
            "Data warehouse for analytics with 10TB storage"
        ]
        
        # Import the app function
        sys.path.append(os.path.dirname(__file__))
        from app import create_project_spec_from_brief
        
        print("✅ Testing project spec creation from user inputs...")
        
        for i, user_input in enumerate(user_inputs):
            print(f"\n🧪 Test {i+1}: {user_input}")
            print("-" * 50)
            
            try:
                # Create ProjectSpec from user input (like the app does)
                spec = create_project_spec_from_brief(user_input)
                
                print(f"✅ Project spec created successfully!")
                print(f"  Name: {spec.name}")
                print(f"  Workload: {spec.workload.train_gpus} GPUs, {spec.workload.inference_qps} QPS")
                print(f"  Data: {spec.data.size_gb} GB, {spec.data.growth_gb_per_month} GB/month growth")
                print(f"  Clouds: {spec.constraints.clouds}")
                print(f"  Regions: {spec.constraints.regions}")
                
                # Now test the full planning pipeline
                from planner.planner import PlannerService
                planner = PlannerService()
                
                result = planner.plan(spec)
                
                if result.get("error"):
                    print(f"❌ Planning failed: {result['error']}")
                    continue
                
                print(f"✅ Planning successful!")
                
                # Extract key metrics
                candidates = result.get('candidates', [])
                if candidates:
                    first_estimate = candidates[0][1]
                    cost = first_estimate.monthly_cost
                    
                    # Get LLM insights
                    llm_insights = None
                    llm_optimization = "None"
                    llm_forecast = "None"
                    
                    if hasattr(first_estimate, 'metadata') and first_estimate.metadata:
                        metadata = first_estimate.metadata
                        llm_insights = metadata.get('llm_insights')
                        llm_optimization = metadata.get('llm_cost_optimization', 'None')
                        llm_forecast = metadata.get('llm_cost_forecast', 'None')
                    
                    print(f"📊 Final Cost: ${cost}")
                    print(f"🎯 LLM Optimization: {llm_optimization}")
                    print(f"📈 LLM Forecast: {llm_forecast}")
                    
                    if llm_insights:
                        print(f"🤖 LLM Insights: {len(llm_insights)} keys")
                        if 'cost_breakdown_analysis' in llm_insights:
                            breakdown = llm_insights['cost_breakdown_analysis']
                            print(f"  - Dominant cost driver: {breakdown.get('dominant_cost_driver', 'N/A')}")
                
                print("✅ Test completed successfully")
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n🎉 Streamlit input testing completed!")
        print("\n💡 Now test the actual Streamlit app with these inputs:")
        for i, user_input in enumerate(user_inputs):
            print(f"  {i+1}. '{user_input}'")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_streamlit_inputs()
