import numpy as np
import pandas as pd

# 生成 1000 个草药-靶点数据，每个 10 维
num_samples = 1000
feature_dim = 10

# 随机生成草药和靶点的特征
herb_features = np.random.rand(num_samples, feature_dim)
target_features = np.random.rand(num_samples, feature_dim)

# 生成边索引（edge_index）：草药指向靶点
edge_index = np.array([
    np.arange(num_samples),  # 发送端（草药索引）
    np.arange(num_samples)   # 接收端（靶点索引）
])

# 生成二分类标签（0或1）
edge_label = np.random.randint(0, 2, size=num_samples)

# 存储为 npz 文件
np.savez("data.npz",
         herb_features=herb_features,
         target_features=target_features,
         edge_index=edge_index,
         edge_label=edge_label)

# 存储为 csv 文件（分为两个文件存储）
herb_df = pd.DataFrame(herb_features)
target_df = pd.DataFrame(target_features)
edge_index_df = pd.DataFrame(edge_index.T, columns=["herb_index", "target_index"])
edge_label_df = pd.DataFrame(edge_label, columns=["label"])

# 保存 CSV
herb_df.to_csv("herb_features.csv", index=False)
target_df.to_csv("target_features.csv", index=False)
edge_index_df.to_csv("edge_index.csv", index=False)
edge_label_df.to_csv("edge_label.csv", index=False)

print("数据已生成并存储为 npz 和 csv 文件。")
