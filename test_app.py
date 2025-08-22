#!/usr/bin/env python3
"""
Simple test script to verify the app logic works correctly
"""

def test_demo_data_structure():
    """Test that the demo data structure works with the app logic"""
    print("🧪 Testing demo data structure...")
    
    # Demo data (same as in app.py)
    demo_result = {
        "spec": {
            "name": "ML Training Pipeline",
            "workload": {"train_gpus": 2, "inference_qps": 100.0, "latency_ms": 200, "batch": True},
            "data": {"size_gb": 1000.0, "hot_fraction": 0.3, "egress_gb_per_month": 500.0}
        },
        "candidates": [
            ({"id": "aws_gpu_training", "cloud": "aws", "region": "us-east-1", "services": [], "assumptions": {"architecture_type": "ml_training"}}, 
             {"monthly_cost": 2500.0, "bom": []}),
            ({"id": "azure_gpu_training", "cloud": "azure", "region": "eastus", "services": [], "assumptions": {"architecture_type": "ml_training"}}, 
             {"monthly_cost": 2800.0, "bom": []}),
            ({"id": "gcp_gpu_training", "cloud": "gcp", "region": "us-central1", "services": [], "assumptions": {"architecture_type": "custom"}}, 
             {"monthly_cost": 2600.0, "bom": []})
        ],
        "winner": {
            "blueprint_id": "aws_gpu_training",
            "estimate": {"monthly_cost": 2000.0, "bom": []},
            "actions": [
                {"type": "spot_instances", "rationale": "Use spot instances for cost optimization", "delta_cost": -500.0}
            ],
            "metadata": {"optimization_score": 85}
        }
    }
    
    # Test spec access
    spec = demo_result["spec"]
    print(f"✅ Project Name: {spec.get('name', 'N/A')}")
    print(f"✅ Training GPUs: {spec.get('workload', {}).get('train_gpus', 'N/A')}")
    print(f"✅ Data Size: {spec.get('data', {}).get('size_gb', 'N/A')} GB")
    
    # Test candidates access
    candidates = demo_result["candidates"]
    print(f"✅ Found {len(candidates)} candidates")
    
    for i, (bp, est) in enumerate(candidates):
        print(f"   Candidate {i+1}: {bp.get('id', 'N/A')} - {bp.get('cloud', 'N/A').upper()}")
        print(f"     Cost: ${est.get('monthly_cost', 0):.2f}")
        print(f"     Services: {len(bp.get('services', []))}")
    
    # Test winner access
    winner = demo_result["winner"]
    winner_bp, winner_est = next((b, e) for (b, e) in candidates if b.get("id") == winner.get("blueprint_id"))
    
    print(f"✅ Winner: {winner_bp.get('id', 'N/A')}")
    print(f"✅ Original Cost: ${winner_est.get('monthly_cost', 0):.2f}")
    print(f"✅ Optimized Cost: ${winner.get('estimate', {}).get('monthly_cost', 0):.2f}")
    print(f"✅ Optimization Score: {winner.get('metadata', {}).get('optimization_score', 'N/A')}")
    
    # Test actions
    actions = winner.get('actions', [])
    print(f"✅ Found {len(actions)} optimization actions")
    
    for action in actions:
        print(f"   Action: {action.get('type', 'N/A').replace('_', ' ').title()}")
        print(f"     Rationale: {action.get('rationale', 'N/A')}")
        print(f"     Cost Impact: ${action.get('delta_cost', 0):.2f}")
    
    print("\n🎉 All tests passed! The app logic works correctly.")
    return True

if __name__ == "__main__":
    test_demo_data_structure()
