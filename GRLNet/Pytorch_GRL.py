import pfrl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse


class torch_GRL(nn.Module):
    # N is the number of agents (40)
    # F is the feature length of each agent (8)
    # A is the number of actions that can be selected (3)
    def __init__(self, N, F, obs_space, action_space, A):
        super(torch_GRL, self).__init__()
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = A

        # encoder
        self.encoder_1 = nn.Linear(F, 32)
        self.encoder_2 = nn.Linear(32, 32)

        # GCN
        self.GraphConv = GCNConv(32, 32)
        self.GraphConv_Dense = nn.Linear(32, 32)

        # policy network
        self.policy_1 = nn.Linear(64, 32)
        self.policy_2 = nn.Linear(32, 32)
        self.policy_output = nn.Linear(32, A)

    def forward(self, X_in, A_in_Dense, RL_indice):
        # X_in represents node feature matrix
        # A_in_Dense represents dense adjacency matrix
        # A_in_Sparse represents sparse adjacency matrix
        # RL_indice represents mask matrix

        # X_in in encoder
        X = self.encoder_1(X_in)
        X = F.relu(X)
        X = self.encoder_2(X)
        X = F.relu(X)

        # graph convolution operation
        A_in_Sparse, _ = dense_to_sparse(A_in_Dense)
        X_graph = self.GraphConv(X, A_in_Sparse)
        X_graph = F.relu(X_graph)
        X_graph = self.GraphConv_Dense(X_graph)
        X_graph = F.relu(X_graph)

        # Features are aggregated by column
        F_concat = torch.cat((X_graph, X), 1)

        # calculate policy
        X_policy = self.policy_1(F_concat)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_2(X_policy)
        X_policy = F.relu(X_policy)
        X_policy = self.policy_output(X_policy)

        # reshape RL_indice
        mask = torch.reshape(RL_indice, (40, 1))

        # calculate final output
        output = torch.mul(X_policy, mask)

        return pfrl.action_value.DiscreteActionValue(output)

