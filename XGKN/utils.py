import json
import math
import os

import torch
import networkx as nx
import numpy as np

from torch_geometric.utils import to_dense_adj, subgraph
from torch_geometric.data import Dataset


def to_torch_sparse_tensor(x, size=None):
    return to_dense_adj(x, max_num_nodes=size)[0].to_sparse()


class Metric:
    def __init__(self):
        self.values = {"train": [], "val": [], "test": []}
        self.best = {"train": 0, "val": 0, "test": 0}

    def add(self, value, split):
        self.values[split] += [value.item()]

    def get(self, split):
        if len(self.values[split]):
            return np.mean(self.values[split])
        else:
            return 0

    def save_higher(self):
        updated = False
        if self.best["val"] < self.get("val"):
            self.best = {
                "train": self.get("train"),
                "val": self.get("val"),
                "test": self.get("test"),
            }
            updated = True
        self.values = {"train": [], "val": [], "test": []}
        return updated

    def restart(self):
        self.values = {"train": [], "val": [], "test": []}
        self.best = {"train": 0, "val": 0, "test": 0}

    def get_best(self, split):
        return self.best[split]


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def accuracy(pred, y):
    return (pred.argmax(dim=-1) == y).float().mean()


def save_model(path, state_dict, model_args, args):
    torch.save({"state_dict": state_dict, "model_args": model_args, "args": args}, path)


def split_data(dataset, dataset_name, seed=0, split=-1, splits_dir=None):
    generator = torch.Generator()
    generator.manual_seed(seed)
    np.random.seed(seed)
    perm = torch.randperm(len(dataset), generator=generator)

    if split < 0:
        val_idx, test_idx = int(len(dataset) * 0.8), int(len(dataset) * 0.9)
        train_idxs, val_idxs, test_idxs = (
            perm[:val_idx],
            perm[val_idx:test_idx],
            perm[test_idx:],
        )
    else:
        with open(os.path.join(splits_dir, f"{dataset_name}.json")) as f:
            splits = json.load(f)
        split = splits[split]
        test_idxs = split["test"]
        train_idxs, val_idxs = (
            split["model_selection"][0]["train"],
            split["model_selection"][0]["validation"],
        )
    return train_idxs, val_idxs, test_idxs


def get_edges_dict(edge_index, num_nodes):
    edges = dict()
    edge_index = [e.tolist() for e in edge_index.T]
    for node in range(num_nodes):
        edges[node] = torch.tensor(list(e[1] for e in edge_index if e[0] == node))
    return edges


def transform(data, k, subgraph_size, node_degree_label=False):
    data.num_nodes += 1
    if node_degree_label:
        adj = to_dense_adj(data.edge_index, max_num_nodes=len(data.x))
        data.x = adj.sum(-1).squeeze(0).unsqueeze(-1)
    x, adj, nidx = extract_subgraphs(data.edge_index, data.x, k, subgraph_size)
    data.subgraphs_x = x
    data.x = x
    data.subgraphs_adj = adj
    data.subgraphs_nidx = nidx
    return data


def k_hop_n(node_idx, num_hops, edge_index, max_size, num_nodes, relabel_nodes=True):
    device = edge_index.device
    col, row = edge_index
    node_mask = row.new_empty(num_nodes, dtype=torch.bool, device=device)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool, device=device)
    subset = torch.tensor([node_idx], dtype=torch.long, device=device)
    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subset] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        s = col[edge_mask]
        s = torch.cat((subset, s))
        _, inverse = np.unique(s.cpu(), return_index=True)
        subset = torch.tensor([s[index.item()] for index in sorted(inverse)], dtype=torch.long, device=device,)
        if len(subset) > max_size:
            break

    subset = subset[:max_size]

    node_mask.fill_(False)
    node_mask[subset] = True

    edge_mask = node_mask[row] & node_mask[col]
    edge_index = edge_index[:, edge_mask]
    if relabel_nodes:
        mapping = row.new_full((num_nodes,), -1)
        mapping[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = mapping[edge_index]

    return subset, edge_index, edge_mask, mapping


def extract_subgraphs(edge_index, x, k, subgraph_size):
    device = edge_index.device
    node_idxs = torch.arange(len(x), device=device)
    num_nodes = len(x)

    def _k_hop_subgraph(idx):
        nidx, sub_edge_index, _, mapping = k_hop_n(
            idx, num_hops=k, edge_index=edge_index, max_size=subgraph_size, num_nodes=num_nodes,
        )
        sub_adj = to_dense_adj(sub_edge_index, max_num_nodes=subgraph_size)
        return nidx, sub_adj, mapping

    subgraphs = [_k_hop_subgraph(idx) for idx in range(len(x))]
    nidx, adj, mapping = zip(*subgraphs)
    adj = torch.cat(adj)
    x = torch.cat([x, torch.zeros(1, x.shape[-1], device=device)])
    adj = torch.cat([adj, torch.zeros(1, adj.shape[-1], adj.shape[-1], device=device)])

    nidx = [torch.nn.functional.pad(n, pad=(0, subgraph_size - len(n)), value=len(x) - 1) for n in nidx] + [
        torch.full((subgraph_size,), fill_value=len(x) - 1, device=device)
    ]
    nidx = torch.stack(nidx)
    return x, adj, nidx


def get_optimizer(model, args):
    optimizer = torch.optim.Adam(list(set(model.parameters())), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    return optimizer, scheduler
