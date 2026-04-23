import numpy as np
import torch

from torch_geometric.utils import to_networkx, to_dense_adj


def edge_substitution_cost(edge_weight_1, edge_weight_2):
    return abs(edge_weight_1 - edge_weight_2)


def weighted_distance(adj_matrix_1, node_features_1, adj_matrix_2, node_features_2):
    n = len(node_features_1)
    m = len(node_features_2)

    # Initialize the DP table
    dp = np.zeros((n + 1, m + 1))

    # Cost of transforming empty graph to non-empty graph
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + 1
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + 1

    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost_delete = dp[i - 1][j] + 1
            cost_insert = dp[i][j - 1] + 1
            cost_substitute = dp[i - 1][j - 1] + 1
            dp[i][j] = min(cost_delete, cost_insert, cost_substitute)

            # Consider edge edit operations if the nodes are substituted
            if dp[i][j] == cost_substitute:
                for k in range(n):
                    for l in range(m):
                        if adj_matrix_1[k][i - 1] != 0 or adj_matrix_2[l][j - 1] != 0:
                            edge_weight_1 = adj_matrix_1[k][i - 1]
                            edge_weight_2 = adj_matrix_2[l][j - 1]

                            edge_delete = edge_weight_1
                            edge_insert = edge_weight_2
                            edge_substitute = edge_substitution_cost(edge_weight_1, edge_weight_2)

                            dp[i][j] += min(edge_delete, edge_insert, edge_substitute)

    norm = adj_matrix_1.sum() + len(node_features_1) + adj_matrix_2.sum() + len(node_features_2)
    return dp[n][m] / norm


def weighted_ged_distance(init_d, after_d):
    x, adj = (
        after_d.x,
        to_dense_adj(after_d.edge_index, edge_attr=after_d.edge_attr, max_num_nodes=len(after_d.x),)[0],
    )
    d = weighted_distance(
        adj.detach().cpu(),
        x.detach().cpu() if init_d.x is not None else torch.ones(len(x), 1),
        to_dense_adj(init_d.edge_index)[0].detach().cpu(),
        init_d.x.detach().cpu() if init_d.x is not None else torch.ones(init_d.num_nodes, 1),
    )
    return d


def compare_size(init_d, after_d):
    return (after_d.x.shape[0] + after_d.edge_index.shape[1]) / (init_d.x.shape[0] + init_d.edge_index.shape[1])


def iou_distance(init_d, after_d):
    if hasattr(init_d, "expl_nodes_subset"):
        init_d = init_d.expl_nodes_subset
        after_d = after_d.expl_nodes_subset
    init_d = set(init_d.tolist())
    after_d = set(after_d.tolist())
    iou = len(init_d.intersection(after_d)) / len(init_d.union(after_d))
    return iou
