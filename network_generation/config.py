"""
    网络数据生成参数配置
    1.节点数量
    2.层数
    3.层间节点差异
    4.平均度（连接密度）
    5.连接类型
"""

import os

# 节点数量配置列表，每个子列表定义了一个节点数量范围
nodes_num = [[700,1200]]
# 层数配置列表，每个子列表定义了一个层数范围
layer_num = [[2,5]]
# 层间节点差异配置列表，每个子列表定义了一个节点差异比例范围
layer_nodes_cover = [[0.3,0.6]]
# 平均度配置列表，每个子列表定义了一个平均度的范围
avg_degree = [[1.8,3]]
# 连接类型（用于选择生成边的函数）'scale_free','small_world','hybrid','er'
graph_type = ['scale_free', 'small_world', 'hybrid', 'er']


"""
    数据集基本配置
"""
_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 保存路径
save_path = os.path.join(_base_dir, "dataset", "train")
# 每类多层网络的数量
graph_num = 100
# 文件编号起始值与补零宽度
id_start = 1
id_width = 4
