import numpy as np
def get_edge_index(G, f_path):
    """给GCN产生输入的边信息

    Args:
        graph_name (str): 节点名称
        args (ArgumentParser): 参数
    """
    edge_np = np.array([list(edge) for edge in G.edges()])
    edge_np_T = edge_np.T
    edge_np_T_1 = edge_np_T[0]
    edge_np_T_2 = edge_np_T[1]
    temp_edge = [edge_np_T_2, edge_np_T_1]
    edge_index = np.concatenate((edge_np_T, temp_edge), axis=1)
    np.save(f_path, edge_index)

