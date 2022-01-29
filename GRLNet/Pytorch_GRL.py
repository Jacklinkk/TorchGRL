import pfrl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse


# 该函数用来将环境中的observation转换成pytorch可以接收的float32的数据类型
def feature_extractor(states):
    features = states[0].astype(np.float32, copy=False)
    adjacency = states[1].astype(np.float32, copy=False)
    mask = states[2].astype(np.float32, copy=False)

    output_states = np.array([features, adjacency, mask])
    return output_states


# 下述为pytorch框架下的GRL网络的主程序
class torch_GRL(nn.Module):
    # N为智能体数量(40)，F为每个智能体的特征长度(8)，A为可选择的动作数量(3)
    def __init__(self, N, F, obs_space, action_space, A):
        super(torch_GRL, self).__init__()
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = A

        # 定义编码器
        self.encoder_1 = nn.Linear(F, 32)
        self.encoder_2 = nn.Linear(32, 32)

        # 定义图卷积网络
        # self.GraphConv = GraphConv(40, 32)
        self.GraphConv = GCNConv(32, 32)
        self.GraphConv_Dense = nn.Linear(32, 32)

        # 定义策略层
        self.policy_1 = nn.Linear(64, 32)
        self.policy_2 = nn.Linear(32, 32)
        self.policy_output = nn.Linear(32, A)

    def forward(self, X_in, A_in_Dense, RL_indice):
        # X_in为节点特征矩阵，A_in_Dense为稠密邻接矩阵（NxN）(原始输入)
        # A_in_Sparse为稀疏邻接矩阵COO（2xnum），RL_indice为强化学习索引

        # 计算X_in解码后的结果
        X = self.encoder_1(X_in)
        X = F.relu(X)
        X = self.encoder_2(X)
        X = F.relu(X)

        # 计算图卷积网络后的结果
        A_in_Sparse, _ = dense_to_sparse(A_in_Dense)  # 将observation的邻接矩阵转换成稀疏矩阵
        X_graph = self.GraphConv(X, A_in_Sparse)
        X_graph = F.relu(X_graph)
        X_graph = self.GraphConv_Dense(X_graph)
        X_graph = F.relu(X_graph)

        # 特征按列聚合
        F_concat = torch.cat((X_graph, X), 1)

        # print("F_concat:", F_concat)
        # print("F_concat.shape:", F_concat.shape)

        # 计算策略层的结果
        X_policy = self.policy_1(F_concat)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_output(X_policy)

        # 重新规定RL_indice的维度
        mask = torch.reshape(RL_indice, (40, 1))

        # 计算网络最终输出
        output = torch.mul(X_policy, mask)

        return pfrl.action_value.DiscreteActionValue(output)

