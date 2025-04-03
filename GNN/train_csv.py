import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from gnn_model import train_model

if __name__ == '__main__':
    # 读取 CSV 数据
    herb_df = pd.read_csv("data/herb_features.csv")
    target_df = pd.read_csv("data/target_features.csv")
    edge_label_df = pd.read_csv("data/edge_label.csv")

    # 转换为 Tensor
    herb_features = torch.tensor(herb_df.values, dtype=torch.float32)
    target_features = torch.tensor(target_df.values, dtype=torch.float32)
    edge_label = torch.tensor(edge_label_df.values.flatten(), dtype=torch.float32)

    # 组合输入特征
    X = torch.cat((herb_features, target_features), dim=1)  # [1000, 20]
    y = edge_label  # [1000]

    # 创建数据加载器
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 训练模型（同上）
    train_model(dataloader)
