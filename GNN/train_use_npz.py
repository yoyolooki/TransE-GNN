import torch
from torch.utils.data import DataLoader, TensorDataset
from gnn_model import train_model
import numpy as np

if __name__ == '__main__':
    # 读取 npz 文件
    data = np.load("./data/data.npz")

    # 提取数据
    herb_features = torch.tensor(data["herb_features"], dtype=torch.float32)
    target_features = torch.tensor(data["target_features"], dtype=torch.float32)
    edge_label = torch.tensor(data["edge_label"], dtype=torch.float32)

    # 拼接草药和靶点特征
    X = torch.cat((herb_features, target_features), dim=1)  # [1000, 20]
    y = edge_label  # [1000]

    # 创建数据加载器
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    train_model(dataloader)
