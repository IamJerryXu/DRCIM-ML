import numpy as np
import matplotlib.pyplot as plt

def normalize_labels(label, method='minmax', **kwargs):
    """
    对节点影响力增益标签进行归一化

    参数:
    labels: 节点影响力增益数组，索引表示节点，值表示节点的增益影响力
    method: 归一化方法，可选 'minmax', 'standard', 'quantile'
    **kwargs: 各归一化方法的额外参数

    返回:
    归一化后的标签数组
    """

    if method == 'minmax':
        return minmax_normalize(label, **kwargs)
    elif method == 'standard':
        return standard_normalize(label, **kwargs)
    elif method == 'quantile':
        return quantile_normalize(label, **kwargs)
    else:
        raise ValueError(f"不支持的归一化方法: {method}")


def minmax_normalize(label, feature_range=(0, 1)):
    """
    Min-Max 归一化: 将数据缩放到指定范围

    参数:
    labels: 原始标签数组
    feature_range: 归一化后的范围，默认为(0, 1)

    返回:
    归一化后的标签数组
    """
    # 计算最小值和最大值
    min_val = np.min(label)
    max_val = np.max(label)

    # Min-Max 归一化
    normalized = (label - min_val) / (max_val - min_val)

    # 缩放到指定范围
    if feature_range != (0, 1):
        normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]

    return normalized


def standard_normalize(label, robust=False, epsilon=1e-8):
    """
    标准化 (Z-score): 将数据转换为均值为0，标准差为1的分布

    参数:
    labels: 原始标签数组
    robust: 是否使用稳健统计量（中位数和MAD）
    epsilon: 小值，避免除零

    返回:
    标准化后的标签数组
    """

    if robust:
        # 使用稳健的标准化（对异常值不敏感）
        median = np.median(label)
        mad = np.median(np.abs(label - median))  # 中位数绝对偏差

        if mad == 0:
            # 如果MAD为0，回退到常规标准化
            mean_val = np.mean(label)
            std_val = np.std(label)
            if std_val == 0:
                return np.zeros_like(label)
            return (label - mean_val) / (std_val + epsilon)

        # 稳健标准化
        normalized = (label - median) / (mad + epsilon)

    else:
        # 常规Z-score标准化
        mean_val = np.mean(label)
        std_val = np.std(label)

        normalized = (label - mean_val) / (std_val + epsilon)

    return normalized


def quantile_normalize(label, alpha=0.1):
    """
    分位数归一化: 将数据转换为指定分布

    参数:
    labels: 原始标签数组
    reference_distribution: 参考分布，可选 'uniform', 'normal'
    n_quantiles: 分位数数量

    返回:
    分位数归一化后的标签数组
    """

    # 按增益排序
    sorted_with_indices = sorted(enumerate(label), key=lambda x: x[1], reverse=True)
    sorted_indices = [index for index, value in sorted_with_indices]
    # 基于排名的权重
    ranks = list(range(1, len(sorted_indices) + 1))
    weights = [1 / (r ** alpha) for r in ranks]  # 排名越高，权重越大
    weights = minmax_normalize(np.array(weights))
    for i in range(len(weights)):
        label[i] = weights[sorted_indices.index(i)]
        # label[i] = (len(label)-sorted_indices.index(i)+1)/len(label)
    return label