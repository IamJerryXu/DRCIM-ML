import numpy as np
import networkx as nx

# 重构自 Multi_RCIM_Tabu_Node/Multi_RCIM_Tabu/Multi_Influ_R.cpp

class RobustnessEvaluator:
    def __init__(self, graph_a, graph_b):
        self.graph_a = graph_a # NetworkX Graph
        self.graph_b = graph_b
        self.num_nodes = len(graph_a.nodes)

    def calculate_robust_influence(self, seeds, task_id, num_attacks=10):
        """
        计算鲁棒影响力 (Robust Influence)
        对应 C++ 中的 cal_R 和 cal_Node_Inf
        """
        # 1. 确定目标网络
        if task_id == 0:
            target_graph = self.graph_a
        elif task_id == 1:
            target_graph = self.graph_b
        else:
            # Task 3: Multiplex (简化处理，取平均或联合)
            target_graph = self.graph_a 
            
        current_graph = target_graph.copy()
        total_influence = 0.0
        
        # 2. 模拟攻击序列
        for _ in range(num_attacks):
            # 2.1 识别关键节点 (基于 2-hop 度信息，简化为 Degree Centrality)
            # C++ 代码中 cal_Node_Inf 实现了复杂的 2-hop 逻辑，这里先用度中心性近似
            degrees = dict(current_graph.degree())
            if not degrees:
                break
            target_node = max(degrees, key=degrees.get)
            
            # 2.2 移除节点
            current_graph.remove_node(target_node)
            
            # 2.3 计算当前网络的竞争影响力 (使用二跳近似)
            inf = self._two_hop_approximation(current_graph, seeds)
            total_influence += inf
            
        return total_influence / num_attacks

    def _two_hop_approximation(self, graph, seeds):
        """
        二跳近似公式计算影响力
        Ref: Multi_Influ_R.cpp 中的 cal_Node_Inf 逻辑
        """
        influence = 0
        # 简化的二跳逻辑：种子节点 + 邻居 + 邻居的邻居 (未被竞争对手占据)
        # 这里仅做示意，完整逻辑需完全翻译 C++ 代码
        
        active_nodes = set(seeds)
        # 1-hop
        for seed in seeds:
            if seed in graph:
                neighbors = list(graph.neighbors(seed))
                active_nodes.update(neighbors)
                
        # 2-hop (简化)
        # ...
        
        return len(active_nodes)
