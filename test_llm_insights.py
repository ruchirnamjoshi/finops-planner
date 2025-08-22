#!/usr/bin/env python3
"""
Test script to verify LLM insights are being generated and stored properly.
"""

import os
import sys
from dotenv import load_dotenv

# Add the planner directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'planner'))

def test_llm_insights():
    """Test if LLM insights are being generated and stored properly."""
    
    print("🧪 Testing LLM Insights Generation and Storage")
    print("=" * 50)
    
    try:
        from planner.planner import PlannerService
        from planner.schemas import ProjectSpec, Workload, DataSpec, Constraints
        
        # Load environment variables
        load_dotenv()
        
        # Create a test project spec
        spec = ProjectSpec(
            name="test-ml-project",
            workload=Workload(
                train_gpus=4,
                inference_qps=100,
                batch=True
            ),
            data=DataSpec(
                size_gb=1000,
                hot_fraction=0.6,
                egress_gb_per_month=200,
                growth_gb_per_month=50,
                retention_days=90
            ),
            constraints=Constraints(
                clouds=["aws", "gcp"],
                regions=["us-east-1", "us-central1"],
                budget_monthly=5000.0
            )
        )
        
        print(f"✅ Created test project spec: {spec.name}")
        
        # Initialize planner service
        planner = PlannerService()
        print("✅ Planner service initialized")
        
        # Generate plan
        print("\n🔄 Generating plan...")
        result = planner.plan(spec)
        
        if result.get("error"):
            print(f"❌ Planning failed: {result['error']}")
            return False
        
        print("✅ Planning successful!")
        
        # Check if we have candidates
        candidates = result.get('candidates', [])
        if not candidates:
            print("❌ No candidates generated")
            return False
        
        print(f"✅ Generated {len(candidates)} candidates")
        
        # Check the first estimate for LLM insights
        first_estimate = candidates[0][1]
        
        if hasattr(first_estimate, 'metadata') and first_estimate.metadata:
            metadata = first_estimate.metadata
            print(f"✅ Estimate has metadata with {len(metadata)} keys")
            
            # Check for LLM insights
            if 'llm_insights' in metadata:
                llm_insights = metadata['llm_insights']
                print(f"✅ LLM insights found: {type(llm_insights)}")
                
                if isinstance(llm_insights, dict):
                    print("📊 LLM Insights Content:")
                    for key, value in llm_insights.items():
                        if isinstance(value, list):
                            print(f"  {key}: {len(value)} items")
                            if value and isinstance(value[0], dict):
                                print(f"    Sample: {value[0]}")
                        else:
                            print(f"  {key}: {value}")
                else:
                    print(f"  Raw content: {llm_insights}")
            else:
                print("❌ No LLM insights in metadata")
                print(f"Available keys: {list(metadata.keys())}")
        else:
            print("❌ Estimate has no metadata")
        
        # Check optimization results
        optimized = result.get('optimized', [])
        if optimized:
            print(f"\n🚀 Found {len(optimized)} optimization results")
            for i, opt in enumerate(optimized[:2]):  # Check first 2
                if hasattr(opt, 'metadata') and opt.metadata:
                    print(f"  Optimization {i+1} metadata keys: {list(opt.metadata.keys())}")
        
        # Check risk assessment
        risks = result.get('risks', [])
        if risks:
            print(f"\n⚠️ Found {len(risks)} risk assessments")
            for i, risk in enumerate(list(risks)[:2]):  # Check first 2
                if hasattr(risk, 'category'):
                    print(f"  Risk {i+1}: {risk.category} - {risk.evidence}")
                else:
                    print(f"  Risk {i+1}: {risk}")
        
        print("\n🎉 LLM insights test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_llm_insights()
    sys.exit(0 if success else 1)
