import numpy as np
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from tools.tool_fun import ICm
import time
from tqdm import tqdm

def celf(G, graph_name,p,mc):
    N = G[0].number_of_nodes()
    marg_gain = [ICm([node], G,p,mc) for node in range(N)]
    # Create the sorted list of nodes and their marginal gain
    Q = sorted(zip(range(N), marg_gain), key=lambda x: x[1], reverse=True)

    # Select the first node and remove from candidate list
    labeb, S, spread = {Q[0][0]:Q[0][1]},[Q[0][0]], Q[0][1]
    Q = Q[1:]

    for _ in tqdm(range(N-1)):

        check = False

        while not check:
            # Recalculate spread of top node
            current = Q[0][0]

            # Evaluate the spread function and store the marginal gain in the list
            Q[0] = (current, ICm(S + [current], G,p,mc) - spread)
            # Re-sort the list
            Q = sorted(Q, key=lambda x: x[1], reverse=True)

            # Check if previous top node stayed on top after the sort
            check = (Q[0][0] == current)

        # Select the next node
        spread += Q[0][1]
        labeb[Q[0][0]]=Q[0][1]#*(N-i)/N
        S.append(Q[0][0])

        # Remove the selected node from the list
        Q = Q[1:]

    labels = []
    for i in range(N):
        labels.append([labeb[i]])

    # 后续归一化
    np.save('label/' + graph_name + '_lb.npy', np.array(labels))