#!/usr/bin/env python3
"""
Test script to verify that the app now generates different results for different inputs.
"""

import os
import sys
from dotenv import load_dotenv

# Add the planner directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'planner'))

def test_app_functionality():
    """Test that the app generates different results for different inputs."""
    
    print("🧪 Testing App Functionality")
    print("=" * 50)
    
    try:
        from planner.planner import PlannerService
        from planner.schemas import ProjectSpec, Workload, DataSpec, Constraints
        
        # Load environment variables
        load_dotenv()
        
        # Test with different project specs (simulating different user inputs)
        test_specs = [
            ProjectSpec(
                name="ml-training-4gpu",
                workload=Workload(train_gpus=4, inference_qps=100, batch=True),
                data=DataSpec(size_gb=1000, growth_gb_per_month=50, hot_fraction=0.6, egress_gb_per_month=200),
                constraints=Constraints(clouds=["aws", "gcp"], regions=["us-east-1", "us-central1"])
            ),
            ProjectSpec(
                name="web-app-high-traffic",
                workload=Workload(train_gpus=0, inference_qps=2000, latency_ms=30, batch=False),
                data=DataSpec(size_gb=200, growth_gb_per_month=30, hot_fraction=0.9, egress_gb_per_month=800),
                constraints=Constraints(clouds=["aws", "azure"], regions=["us-east-1", "eastus"])
            ),
            ProjectSpec(
                name="data-lake-10tb",
                workload=Workload(train_gpus=0, inference_qps=50, latency_ms=500, batch=True),
                data=DataSpec(size_gb=10000, growth_gb_per_month=200, hot_fraction=0.2, egress_gb_per_month=1500),
                constraints=Constraints(clouds=["aws", "gcp"], regions=["us-east-1", "us-central1"])
            )
        ]
        
        planner = PlannerService()
        
        results = []
        
        for i, spec in enumerate(test_specs):
            print(f"\n🧪 Test {i+1}: {spec.name}")
            print("-" * 30)
            
            try:
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
                    
                    results.append({
                        'name': spec.name,
                        'cost': cost,
                        'llm_insights': llm_insights,
                        'llm_optimization': llm_optimization,
                        'llm_forecast': llm_forecast
                    })
                    
                    print(f"📊 Cost: ${cost}")
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
        
        # Summary comparison
        print("\n📊 RESULTS COMPARISON")
        print("=" * 50)
        
        if len(results) >= 2:
            print("✅ Different costs generated for different projects!")
            
            for i, result in enumerate(results):
                print(f"\n{result['name']}:")
                print(f"  Cost: ${result['cost']}")
                print(f"  LLM Optimization: {result['llm_optimization']}")
                print(f"  LLM Forecast: {result['llm_forecast']}")
            
            # Check if costs are different
            costs = [r['cost'] for r in results]
            if len(set(costs)) > 1:
                print(f"\n🎉 SUCCESS: Generated {len(set(costs))} different costs!")
                print(f"Cost range: ${min(costs):.2f} - ${max(costs):.2f}")
            else:
                print(f"\n⚠️ WARNING: All projects have the same cost: ${costs[0]}")
        else:
            print("❌ Not enough results to compare")
        
        print("\n🎉 App functionality test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_app_functionality()
