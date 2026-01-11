from node2vec import Node2Vec
import numpy as np

# 获取node2vec图嵌入
def node2vec(G, f_path, args):
    """根据node2vec产生节点的embedding

    Args:
        G (networkx.Graph): 输入图
        args (ArgumentParser): 参数
    """
    node2vec_1 = Node2Vec(G,
                          dimensions=args.graph_embedding,
                          p=args.p,
                          q=args.q,
                          walk_length=args.walk_length,
                          num_walks=args.num_walks,
                          workers=args.workers,
                          quiet=True)
    model = node2vec_1.fit(window=args.window_size, min_count=5, batch_words=32)
    x = model.wv.vectors
    # print(model.wv.vectors.shape)
    features = list()
    for i in range(model.wv.vectors.shape[0]):
        temp = model.wv.get_vector(i)
        features.append(np.array(temp))
    features = np.array(features)
    # print(features)
    # print(features.shape)
    np.save(f_path, features)

