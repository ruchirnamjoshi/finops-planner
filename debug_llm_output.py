#!/usr/bin/env python3
"""
Debug script to see exactly what LLM outputs are being generated and stored.
"""

import os
import sys
from dotenv import load_dotenv

# Add the planner directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'planner'))

def debug_llm_outputs():
    """Debug what LLM outputs are actually being generated and stored."""
    
    print("🔍 Debugging LLM Outputs")
    print("=" * 50)
    
    try:
        from planner.planner import PlannerService
        from planner.schemas import ProjectSpec, Workload, DataSpec, Constraints
        
        # Load environment variables
        load_dotenv()
        
        # Test with different project specs
        test_specs = [
            ProjectSpec(
                name="ml-training-gpu",
                workload=Workload(train_gpus=4, inference_qps=100, batch=True),
                data=DataSpec(size_gb=1000, growth_gb_per_month=50, hot_fraction=0.6, egress_gb_per_month=200),
                constraints=Constraints(clouds=["aws", "gcp"], regions=["us-east-1", "us-central1"])
            ),
            ProjectSpec(
                name="web-app-ha",
                workload=Workload(train_gpus=0, inference_qps=1000, latency_ms=50, batch=False),
                data=DataSpec(size_gb=100, growth_gb_per_month=10, hot_fraction=0.8, egress_gb_per_month=500),
                constraints=Constraints(clouds=["aws", "azure"], regions=["us-east-1", "eastus"])
            ),
            ProjectSpec(
                name="data-warehouse",
                workload=Workload(train_gpus=0, inference_qps=100, batch=True),
                data=DataSpec(size_gb=10000, growth_gb_per_month=100, hot_fraction=0.3, egress_gb_per_month=1000),
                constraints=Constraints(clouds=["aws", "gcp"], regions=["us-east-1", "us-central1"])
            )
        ]
        
        planner = PlannerService()
        
        for i, spec in enumerate(test_specs):
            print(f"\n🧪 Test {i+1}: {spec.name}")
            print("-" * 30)
            
            try:
                result = planner.plan(spec)
                
                if result.get("error"):
                    print(f"❌ Planning failed: {result['error']}")
                    continue
                
                print(f"✅ Planning successful!")
                
                # Check candidates
                candidates = result.get('candidates', [])
                if candidates:
                    first_estimate = candidates[0][1]
                    print(f"📊 First estimate cost: ${first_estimate.monthly_cost}")
                    
                    # Check metadata
                    if hasattr(first_estimate, 'metadata') and first_estimate.metadata:
                        metadata = first_estimate.metadata
                        print(f"🔍 Metadata keys: {list(metadata.keys())}")
                        
                        # Check LLM insights
                        if 'llm_insights' in metadata:
                            llm_insights = metadata['llm_insights']
                            print(f"🤖 LLM insights type: {type(llm_insights)}")
                            
                            if isinstance(llm_insights, dict):
                                for key, value in llm_insights.items():
                                    if isinstance(value, list):
                                        print(f"  {key}: {len(value)} items")
                                        if value and isinstance(value[0], dict):
                                            print(f"    Sample: {value[0]}")
                                    else:
                                        print(f"  {key}: {value}")
                        
                        # Check LLM cost optimization
                        if 'llm_cost_optimization' in metadata:
                            print(f"🎯 LLM Cost Optimization: {metadata['llm_cost_optimization']}")
                        
                        if 'llm_cost_forecast' in metadata:
                            print(f"📊 LLM Cost Forecast: {metadata['llm_cost_forecast']}")
                
                # Check optimization results
                optimized = result.get('optimized', [])
                if optimized:
                    print(f"🚀 Optimization results: {len(optimized)}")
                    for j, opt in enumerate(optimized[:1]):  # Check first one
                        if hasattr(opt, 'metadata') and opt.metadata:
                            opt_metadata = opt.metadata
                            print(f"  Optimization {j+1} metadata keys: {list(opt_metadata.keys())}")
                            
                            if 'llm_recommendations' in opt_metadata:
                                llm_recs = opt_metadata['llm_recommendations']
                                print(f"    LLM recommendations: {len(llm_recs)} items")
                                if llm_recs and isinstance(llm_recs[0], dict):
                                    print(f"    Sample: {llm_recs[0]}")
                
                # Check risk assessment
                risks = result.get('risks', [])
                if risks:
                    print(f"⚠️ Risk assessments: {len(risks)}")
                    for j, risk in enumerate(risks[:1]):  # Check first one
                        if hasattr(risk, 'category'):
                            print(f"  Risk {j+1}: {risk.category} - {risk.sevidence}")
                        else:
                            print(f"  Risk {j+1}: {risk}")
                
                print("✅ Test completed successfully")
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n🎉 Debug completed!")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_llm_outputs()
