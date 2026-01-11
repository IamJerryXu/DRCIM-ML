import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix

def compute_rrwp(edge_index, num_nodes, k_steps=5):
    """
    Compute Relative Random Walk Probabilities (RRWP) for the given graph.
    
    Args:
        edge_index (LongTensor): Graph connectivity in COO format [2, E].
        num_nodes (int): Number of nodes.
        k_steps (int): Number of random walk steps.
        
    Returns:
        rrwp_attr (Tensor): RRWP features for each edge in edge_index [E, k_steps].
        (Note: This implementation assumes we only need RRWP for existing edges. 
         For full Transformer attention, dense RRWP would be needed.)
    """
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.from_numpy(edge_index)
    edge_index = edge_index.cpu()
    
    # Convert to sparse matrix
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    
    # 2. Compute Transition Matrix P = D^-1 A
    # Calculate degrees
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv = np.power(deg, -1, where=deg != 0)
    deg_inv[np.isinf(deg_inv)] = 0
    D_inv = sp.diags(deg_inv)
    
    P = D_inv.dot(adj)
    
    # 3. Compute powers of P: P^1, P^2, ..., P^K
    # We want to extract P_ij^k for all (i, j) in edge_index.
    # Since we only need values for existing edges, we can just compute P^k (sparse)
    # and extract values.
    
    # However, P^k becomes denser as k increases.
    # For small k (e.g. 3-5), it's manageable.
    
    # Store P^k matrices
    P_powers = [P]
    for k in range(1, k_steps):
        P_next = P_powers[-1].dot(P)
        P_powers.append(P_next)
        
    # 4. Extract values for each edge in edge_index
    # edge_index is [2, E]
    row = np.array(edge_index[0].numpy(), copy=True)
    col = np.array(edge_index[1].numpy(), copy=True)
    
    rrwp_features = []
    for k in range(k_steps):
        vals = np.array(P_powers[k][row, col]).flatten()
        rrwp_features.append(torch.from_numpy(vals).float())
        
    # Stack to get [E, k_steps]
    rrwp_attr = torch.stack(rrwp_features, dim=1)
    
    return rrwp_attr

def get_rrwp_dense(edge_index, num_nodes, k_steps=5):
    """
    Compute dense RRWP for all pairs (for full Transformer).
    Returns [N, N, k_steps]
    """
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv = np.power(deg, -1)
    deg_inv[np.isinf(deg_inv)] = 0
    D_inv = sp.diags(deg_inv)
    P = D_inv.dot(adj)
    
    P_dense = P.toarray()
    P_powers = [torch.from_numpy(P_dense).float()]
    
    curr_P = P_dense
    for _ in range(1, k_steps):
        curr_P = curr_P @ P_dense
        P_powers.append(torch.from_numpy(curr_P).float())
        
    return torch.stack(P_powers, dim=-1)


def attach_rrwp(data, k_steps=5):
    """
    Attach RRWP edge features to a PyG Data object.
    """
    rrwp = compute_rrwp(data.edge_index, data.num_nodes, k_steps=k_steps)
    data.rrwp = rrwp.to(data.edge_index.device)
    return data
