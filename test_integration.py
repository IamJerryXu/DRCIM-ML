#!/usr/bin/env python3
"""
Test GMA-MFEA Integration.

Quick test to verify the complete pipeline works.
"""

import numpy as np
import networkx as nx
import os
import sys

def test_gma_mfea_integration():
    """Test the full GMA-MFEA integration."""
    print("=" * 60)
    print("GMA-MFEA Integration Test")
    print("=" * 60)
    
    # 1. Create mock data
    print("\n[1] Creating mock GMA outputs...")
    os.makedirs('data/alignment/similarity_matrix', exist_ok=True)
    
    num_nodes = 50
    s_align = np.random.rand(num_nodes, num_nodes)
    s_align = (s_align + s_align.T) / 2
    np.save('data/alignment/similarity_matrix/S_align.npy', s_align)
    
    alpha_l1 = np.random.rand(num_nodes)
    alpha_l2 = np.random.rand(num_nodes)
    np.save('data/alignment/alpha_l1.npy', alpha_l1)
    np.save('data/alignment/alpha_l2.npy', alpha_l2)
    
    embeddings = np.random.rand(num_nodes, 64)
    np.save('data/alignment/embeddings_l1.npy', embeddings)
    np.save('data/alignment/embeddings_l2.npy', embeddings)
    print("  Mock GMA outputs created.")
    
    # 2. Test loading
    print("\n[2] Testing load_gma_outputs...")
    from code.main import load_gma_outputs
    
    config = {
        'paths': {'data_root': './data'},
        'gma': {'checkpoint_prefix': 'test'},
    }
    outputs = load_gma_outputs(config)
    assert outputs['s_align'] is not None, "s_align not loaded"
    assert outputs['alpha_l1'] is not None, "alpha_l1 not loaded"
    print(f"  s_align: {outputs['s_align'].shape}")
    print(f"  alpha_l1: {outputs['alpha_l1'].shape}")
    print("  ✓ load_gma_outputs works!")
    
    # 3. Test MFEA with GMA outputs
    print("\n[3] Testing MFEA with GMA outputs...")
    from code.mfea_core import run_mfea_solver
    
    # Create mock graphs
    g1 = nx.erdos_renyi_graph(num_nodes, 0.15, seed=42)
    g2 = nx.erdos_renyi_graph(num_nodes, 0.15, seed=43)
    adj_l1 = [list(g1.neighbors(i)) for i in range(num_nodes)]
    adj_l2 = [list(g2.neighbors(i)) for i in range(num_nodes)]
    
    mfea_config = {
        'mfea': {
            'num_nodes': num_nodes,
            'budget_l1': 5,
            'budget_l2': 5,
            'population_size': 20,
            'num_generations': 5,
            'local_search_enabled': True,
            'local_search_interval': 2,
            'seed': 42,
        },
        'attack': {'attack_mode': '2hop', 'attack_ratio': 0.1}
    }
    
    result = run_mfea_solver(
        config=mfea_config,
        s_align=outputs['s_align'],
        alpha_l1=outputs['alpha_l1'],
        alpha_l2=outputs['alpha_l2'],
        embeddings_l1=outputs['embeddings_l1'],
        embeddings_l2=outputs['embeddings_l2'],
        graph_l1=adj_l1,
        graph_l2=adj_l2,
        adj_l1=adj_l1,
        adj_l2=adj_l2,
        verbose=True,
    )
    
    assert result['best_t1'] is not None, "No best T1 solution"
    assert result['best_t3'] is not None, "No best T3 solution"
    print(f"\n  Best T1: {result['best_scores']['T1']:.4f}")
    print(f"  Best T3: {result['best_scores']['T3']:.4f}")
    print("  ✓ MFEA with GMA works!")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ All integration tests passed!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = test_gma_mfea_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
