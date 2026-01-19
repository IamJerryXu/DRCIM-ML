import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

# Laplacian matrix
def normalized_laplacian_matrix(G, f_path):
    """产生图的归一化拉普拉斯矩阵，保存到对应的npy文件中

    Args:
        G (networkx.Grpah()): 输入图
    """
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G,nodelist=range(n))
    D = sp.diags(np.ravel(np.sum(A, axis=1)))
    L = D - A
    # 计算D的平方根的逆矩阵
    D_sqrt_inv = sp.diags(np.ravel(1 / np.sqrt(np.sum(A, axis=1))))
    # 计算归一化拉普拉斯矩阵
    L_norm = D_sqrt_inv @ L @ D_sqrt_inv

    np.save(f_path, np.array(L_norm.toarray()))


def compute_lap_pe(G, k, f_path, normalized=True):
    """
    计算 Laplacian Positional Encoding (LapPE) 并保存到文件。

    Args:
        G (networkx.Graph): 输入图
        k (int): 需要的特征向量维度
        f_path (str): 保存路径
        normalized (bool): 是否使用归一化拉普拉斯
    """
    n = G.number_of_nodes()
    if n == 0:
        np.save(f_path, np.zeros((0, k), dtype=np.float32))
        return

    A = nx.adjacency_matrix(G, nodelist=range(n))
    deg = np.array(A.sum(axis=1)).flatten()

    if normalized:
        deg_inv_sqrt = np.power(deg, -0.5, where=deg != 0)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0
        D_inv_sqrt = sp.diags(deg_inv_sqrt)
        L = sp.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        D = sp.diags(deg)
        L = D - A

    # 取最小的 k+1 个特征向量，跳过常数特征
    k_eff = min(k + 1, n)
    try:
        vals, vecs = eigsh(L, k=k_eff, which="SM")
    except Exception:
        vals, vecs = np.linalg.eigh(L.toarray())

    idx = np.argsort(vals)
    vecs = vecs[:, idx]

    start = 1 if vecs.shape[1] > 1 else 0
    pe = vecs[:, start:start + k]

    # 若维度不足则补零
    if pe.shape[1] < k:
        pad = np.zeros((n, k - pe.shape[1]), dtype=pe.dtype)
        pe = np.concatenate([pe, pad], axis=1)

    np.save(f_path, pe.astype(np.float32))
