import copy
import json
import os
import random

from tqdm import tqdm

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset, BA2MotifDataset, BAMultiShapesDataset
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import Sequential, GAT, GCN, GIN
from torch_geometric.utils import to_dense_adj, subgraph
from torch_geometric.data import Data, Batch

import networkx as nx

from scipy.stats import spearmanr
import shap


METRICS = {
    "A": "Accuracy",
    "A1": "Accuracy (instance-level)",  # IoU instance-level to GT
    "A2": "Accuracy (model-level)",  # GED between model-level explanation and GT
    "I1": "Sufficiency",  # with + noise (sufficiency)
    "I2": "Necessity",  # without + noise (necessity)
    "I3": "Robustness (nodes)",  # IoU after nodes are altered (should not change prediction) (robustness)
    "I4": "Robustness (edges)",  # IoU after edges are altered (should not change prediction) (robustness)
    "I5": "Consistency",  # Consistency of explainer IoU
    "M1": "Correctness (nodes)",  # prototype nodes affect explanations
    "M2": "Correctness (edges)",  # prototype edges affect explanations
    "M3": "Redundancy",  # Redundancy of prototypes
}

EXPLAINERS = [
    "GNNExplainer",
    "PGExplainer",
    "IntegratedGradients",
    "AttentionExplainer",
    "GraphMaskExplainer",
    "ShapleyValueSampling",
]


def find_five_node_cycles(graph_data, return_all=False):
    G = nx.Graph()
    G.add_edges_from(graph_data.edge_index.t().tolist())
    cycles = list()
    for cycle in nx.simple_cycles(nx.DiGraph(G)):
        if len(cycle) == 5:
            if return_all:
                cycles.append(cycle)
            else:
                return cycle
    return cycles if return_all else None


def find_house_motif(graph_data):
    G = nx.Graph()
    G.add_edges_from(graph_data.edge_index.t().tolist())
    cycles = find_five_node_cycles(graph_data, return_all=True)
    for cycle in cycles:
        for node in cycle:
            neighbors = [v for v in list(G.neighbors(node)) if v in cycle]
            if len(neighbors) == 3:
                return cycle
    return []


def find_wheel_motif(graph_data):
    G = nx.Graph()
    G.add_edges_from(graph_data.edge_index.t().tolist())
    cycles = find_five_node_cycles(graph_data, return_all=True)
    for cycle in cycles:
        for node in G.nodes:
            if (node not in cycle) and sorted(G.neighbors(node)) == sorted(cycle):
                return cycle + [node]
    return []


def is_3x3_grid(subgraph):
    expected_edges = set(
        [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8),]
    )

    mapping = {node: idx for idx, node in enumerate(subgraph.nodes())}
    relabeled_subgraph = nx.relabel_nodes(subgraph, mapping)
    actual_edges = set([tuple(sorted(edge)) for edge in relabeled_subgraph.edges()])
    return expected_edges.issubset(actual_edges)


# Function to find 3x3 grids in the graph
def find_grid_motif(graph_data):
    G = nx.Graph()
    G.add_edges_from(graph_data.edge_index.t().tolist())

    for cycle in nx.simple_cycles(nx.DiGraph(G)):
        if len(cycle) == 8:
            for node in G.nodes:
                if node not in cycle:
                    subgraph = G.subgraph(cycle + [node])
                    if is_3x3_grid(subgraph):  # Check if the subgraph matches the 3x3 grid
                        return cycle + [node]
    return []


def get_mutag_gt(data):
    G = nx.Graph()
    G.add_edges_from(data.edge_index.t().tolist())
    nx.simple_cycles(nx.DiGraph(G))
    explanation_nodes_subset = list()

    def find_atom(x, idx, label):
        idx = torch.tensor(idx, dtype=torch.long)
        one_hot = F.one_hot(torch.tensor([label], dtype=torch.long), num_classes=7).to(x.device)
        mask = (x[idx] == one_hot).amin(dim=-1).cpu()
        return idx[mask].tolist()

    for cycle in nx.simple_cycles(nx.DiGraph(G)):
        nodes_C = find_atom(data.x, cycle, 0)
        if len(nodes_C) == len(cycle):
            explanation_nodes_subset += cycle

    if data.y == 0:
        no2_group = None
        nodes_N = find_atom(data.x, list(range(len(data.x))), 1)
        for n in nodes_N:
            neighbors = [i for i in G.neighbors(n)]
            nodes_O = find_atom(data.x, neighbors, 2)
            if len(nodes_O) > 0:
                assert len(nodes_O) == 2
                no2_group = nodes_O + [n]
                explanation_nodes_subset += no2_group
        if no2_group is None:
            explanation_nodes_subset = []
    explanation_nodes_subset = torch.tensor(list(set(explanation_nodes_subset)), dtype=torch.long)
    expl_node_mask = torch.zeros(len(data.x), dtype=torch.bool, device=data.x.device)
    expl_node_mask[explanation_nodes_subset] = True
    data.gt_expl_node_mask = expl_node_mask
    return data


def get_gt_explanation_ba_2motif(data):
    expl_node_mask = torch.zeros(len(data.x), dtype=torch.bool, device=data.x.device)
    expl_node_mask[torch.tensor([20, 21, 22, 23, 24]).long()] = True
    data.gt_expl_node_mask = expl_node_mask
    return data


def get_gt_explanation_ba_multishapes(data):
    gt_explanations = get_gt_explanations_model("BAMultiShapes")

    house = torch.tensor(find_house_motif(data)).long()
    wheel = torch.tensor(find_wheel_motif(data)).long()
    grid = torch.tensor(find_grid_motif(data)).long()
    expl_node_mask = torch.zeros(len(data.x), dtype=torch.bool, device=data.x.device)
    if len(house) == 0 and len(wheel) == 0 and len(grid) == 0:
        data.gt_expl_node_mask = expl_node_mask
    elif len(house) == 5 and len(wheel) == 0 and len(grid) == 0:
        expl_node_mask[house] = True
        data.gt_expl_node_mask = expl_node_mask
    elif len(house) == 0 and len(wheel) == 6 and len(grid) == 0:
        expl_node_mask[wheel] = True
        data.gt_expl_node_mask = expl_node_mask
    elif len(house) == 0 and len(wheel) == 0 and len(grid) == 9:
        expl_node_mask[grid] = True
        data.gt_expl_node_mask = expl_node_mask
    elif len(house) == 5 and len(wheel) == 6 and len(grid) == 9:
        expl_node_mask[house] = True
        expl_node_mask[wheel] = True
        expl_node_mask[grid] = True
        data.gt_expl_node_mask = expl_node_mask
    elif len(house) == 5 and len(wheel) == 6 and len(grid) == 0:
        expl_node_mask[house] = True
        expl_node_mask[wheel] = True
        data.gt_expl_node_mask = expl_node_mask
    elif len(house) == 5 and len(wheel) == 0 and len(grid) == 9:
        expl_node_mask[house] = True
        expl_node_mask[grid] = True
        data.gt_expl_node_mask = expl_node_mask
    elif len(house) == 0 and len(wheel) == 6 and len(grid) == 9:
        expl_node_mask[wheel] = True
        expl_node_mask[grid] = True
        data.gt_expl_node_mask = expl_node_mask
    else:
        assert False, (house, wheel, grid)
    return data


def get_dataset(dataset_name, degree_attr):
    def add_degree(data):
        adj = to_dense_adj(data.edge_index)[0]
        data.x = adj.sum(-1).unsqueeze(-1)
        return data

    def add_explanation(data):
        get_gt_fn = {
            "MUTAG": get_mutag_gt,
            "BAMultiShapes": get_gt_explanation_ba_multishapes,
            "BA-2motif": get_gt_explanation_ba_2motif,
        }[dataset_name]
        return get_gt_fn(data)

    add_explanation_fn = add_explanation if dataset_name in ["MUTAG", "BA-2motif", "BAMultiShapes"] else (lambda x: x)
    transform = (lambda x: add_degree(add_explanation_fn(x))) if degree_attr else add_explanation_fn

    if dataset_name == "BA-2motif":
        dataset = BA2MotifDataset(f"../datasets/{dataset_name}", force_reload=True, transform=transform)
    elif dataset_name == "BAMultiShapes":
        dataset = BAMultiShapesDataset(f"../datasets/{dataset_name}", force_reload=True, transform=transform)
    else:
        dataset = TUDataset(f"../datasets/", name=dataset_name, use_node_attr=True, transform=transform,)
    return dataset


def get_splits(dataset_name, size=None, seed=123, split=-1):
    if split < 0:
        generator = torch.Generator()
        generator.manual_seed(seed)
        np.random.seed(seed)
        perm = torch.randperm(size, generator=generator)
        val_idx, test_idx = int(size * 0.8), int(size * 0.9)
        train_idxs, val_idxs, test_idxs = (
            perm[:val_idx],
            perm[val_idx:test_idx],
            perm[test_idx:],
        )
    else:
        with open(os.path.join(f"../datasets/splits", f"{dataset_name}.json")) as f:
            splits = json.load(f)
        split = splits[split]
        test_idxs = split["test"]
        train_idxs, val_idxs = (
            split["model_selection"][0]["train"],
            split["model_selection"][0]["validation"],
        )
    return train_idxs, val_idxs, test_idxs


def get_acc(model, dataloader, forward_fn):
    acc, count = 0, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            out = forward_fn(model, data)
            acc += (out.argmax(dim=-1) == data.y).float().sum()
            count += len(data.y)
    return (acc.item() / count) if count > 0 else 0.0


def get_gt_explanations_model(dataset_name):
    gt_dataset = list()
    cls_0, cls_1 = (
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
    )

    if dataset_name == "MUTAG":
        edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        x = F.one_hot(torch.tensor([1, 2, 2]).long(), num_classes=7)
        gt_dataset.append(Data(edge_index=edge_index, x=x, y=cls_0.clone(), num_nodes=edge_index.max() + 1,))
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 5, 5]], dtype=torch.long)
        x = F.one_hot(torch.tensor([0, 0, 0, 0, 0, 0]).long(), num_classes=7)
        gt_dataset.append(Data(edge_index=edge_index, x=x, y=cls_1.clone(), num_nodes=edge_index.max() + 1,))

    elif dataset_name == "REDDIT-BINARY":
        edge_index = torch.tensor([[0, 0, 0], [1, 2, 3]]).long()
        gt_dataset.append(Data(edge_index=edge_index, y=cls_0.clone(), num_nodes=edge_index.max() + 1))
        edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [2, 3, 4, 5, 2, 3, 4, 5]], dtype=torch.long())
        gt_dataset.append(Data(edge_index=edge_index, y=cls_1.clone(), num_nodes=edge_index.max() + 1))

    elif dataset_name == "BA-2motif":
        edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 4, 4]], dtype=torch.long)
        gt_dataset.append(Data(edge_index=edge_index, y=cls_0.clone(), num_nodes=edge_index.max() + 1))
        edge_index = torch.tensor([[0, 1, 2, 3, 0, 1], [1, 2, 3, 4, 4, 4]], dtype=torch.long)
        gt_dataset.append(Data(edge_index=edge_index, y=cls_1.clone(), num_nodes=edge_index.max() + 1))
    elif dataset_name == "BAMultiShapes":
        house = torch.tensor([[0, 1, 2, 3, 0, 1], [1, 2, 3, 4, 4, 4]], dtype=torch.long)
        wheel = torch.tensor([[0, 1, 2, 3, 0, 1, 0, 1, 2, 3, 4], [1, 2, 3, 4, 4, 4, 5, 5, 5, 5, 5]], dtype=torch.long,)
        grid = torch.tensor(
            [[0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 6, 7], [1, 2, 2, 4, 5, 4, 6, 5, 7, 8, 7, 8],], dtype=torch.long,
        )
        gt_dataset.append(Data(edge_index=house.clone(), y=cls_0.clone(), num_nodes=house.amax() + 1))
        gt_dataset.append(Data(edge_index=wheel.clone(), y=cls_0.clone(), num_nodes=wheel.amax() + 1))
        gt_dataset.append(Data(edge_index=grid.clone(), y=cls_0.clone(), num_nodes=grid.amax() + 1))
    else:
        return []
    return gt_dataset


def compare_lists(init_data, after_data, distance_fn):
    return [distance_fn(init_d, after_d) for init_d, after_d in zip(init_data, after_data)]


def find_elbow(values, softmax=False):
    torch_y = values.sort()[0]
    y = torch_y
    if softmax:
        y = F.softmax(torch_y)
    y = y.cpu().detach().numpy()
    x = np.arange(len(y))
    line_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
    line_vec = line_vec / np.linalg.norm(line_vec)
    vec_from_first = np.array([x - x[0], y - y[0]]).T
    proj_onto_line = np.dot(vec_from_first, line_vec)
    distance_to_line = np.linalg.norm(vec_from_first - proj_onto_line[:, None] * line_vec, axis=1)
    elbow_idx = np.argmax(distance_to_line)
    return torch_y[elbow_idx]


def find_percentile(values, p=0.85, softmax=True):
    y, indices = values.sort()
    if not softmax:
        if not torch.all(y >= 0):
            y -= y.min()
        assert torch.all(y >= 0), y
    cumsum = torch.cumsum(F.softmax(y, dim=0), dim=0) if softmax else torch.cumsum(y / sum(y), dim=0)
    breakpoint = indices[(cumsum <= p)]
    if len(breakpoint) > 0:
        breakpoint = breakpoint[-1]
    else:
        breakpoint = 0
    return values[breakpoint]


def find_topk(values, k):
    topk_value = torch.topk(values, k=min(k, len(values)))[0][-1]
    return topk_value


def get_shap(classifier, inputs, check_additivity=True):
    torch.set_grad_enabled(True)
    explainer = shap.DeepExplainer(classifier, Variable(inputs))
    shap_values = explainer.shap_values(Variable(inputs), check_additivity=check_additivity)
    shap_values = np.stack(shap_values, axis=0)
    return torch.from_numpy(shap_values), explainer


def run_shap(explainer, input):
    try:
        shap = torch.from_numpy(np.stack(explainer.shap_values(Variable(input))))
    except:
        shap = list()
        for i in range(input.shape[0]):
            try:
                shap += [torch.from_numpy(np.stack(explainer.shap_values(Variable(input[i : i + 1]))))]
            except:
                shap += [torch.zeros(explainer.expected_value.shape[0], 1, input.shape[-1])]
        shap = torch.stack(shap, dim=1)
    return shap


def extract_explanation_nodes_subset(e, node_mask_fn=None, edge_mask_fn=None, skip_last=False):
    assert (node_mask_fn is not None) or (edge_mask_fn is not None)
    assert not ((node_mask_fn is not None) and (edge_mask_fn is not None))
    device = e.edge_index.device
    subset = torch.arange(len(e.x) - int(skip_last), device=device)
    if node_mask_fn is not None:
        node_mask = e.node_mask[:-1] if skip_last else e.node_mask
    if node_mask_fn is not None:
        thr = node_mask_fn(node_mask)

        if thr is not None:
            node_mask = node_mask >= thr
            subset = subset[node_mask]

    if edge_mask_fn is not None:
        thr = edge_mask_fn(e.edge_mask)
        if thr is not None:
            edge_mask = e.edge_mask >= thr
            edge_index = e.edge_index[:, edge_mask]
            if node_mask_fn is None:
                subset = torch.unique(edge_index)
    return subset


def pairwise_list_corr(x, to_abs=True):
    correlations = list()
    for i in range(x.shape[-1]):
        for j in range(i + 1, x.shape[-1]):
            corr, _ = spearmanr(x[:, i], x[:, j])
            correlations.append(np.abs(corr) if to_abs else corr)
    corrs = [corr if not np.isnan(corr) else 1.0 for corr in correlations]
    return np.mean(corrs)


def get_edge_mask(edge_index, nidx, isin=True):
    nidx_tensor = torch.tensor(nidx, dtype=edge_index.dtype, device=edge_index.device)

    if isin:
        src_nodes_in_nidx = torch.isin(edge_index[0], nidx_tensor)
        tgt_nodes_in_nidx = torch.isin(edge_index[1], nidx_tensor)
    else:
        device = edge_index.device
        src_nodes_in_nidx = torch.tensor([i in nidx_tensor for i in edge_index[0]], dtype=torch.bool, device=device)
        tgt_nodes_in_nidx = torch.tensor([i in nidx_tensor for i in edge_index[1]], dtype=torch.bool, device=device)

    mask = src_nodes_in_nidx & tgt_nodes_in_nidx
    return mask.bool()


def remove_edges(edge_index, edges_to_remove):
    edge_index_set = set(tuple(sorted(edge_index[:, i].tolist())) for i in range(edge_index.size(1)))
    edges_to_remove_set = set(tuple(sorted(edges_to_remove[:, i].tolist())) for i in range(edges_to_remove.size(1)))

    new_edge_index_set = edge_index_set - edges_to_remove_set
    if len(new_edge_index_set) > 0:
        return torch.tensor(list(new_edge_index_set)).t().contiguous()
    else:
        return torch.zeros([2, 0]).long()


def add_noise_perturb(
    explanations,
    forward_fn,
    model,
    p_x=0.05,
    p_edges_del=0.01,
    p_edges_add=0.01,
    skip_last=False,
    n_samples=500,
    preprocess_graph=None,
    max_samples=5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if skip_last:
        all_x = torch.cat([data.x[:-1] for data in explanations])
    else:
        all_x = torch.cat([data.x for data in explanations])

    p_edges_add = p_edges_add * get_edges_ratio(explanations)

    def _add_noise_perturb(e):
        e = e.to(device)
        x, edge_index, y = e.x.clone(), e.edge_index.clone(), e.y.clone()
        expl_nodes_subset = e.expl_nodes_subset
        n, m = len(x), edge_index.shape[-1]
        if p_x > 0:
            keys = torch.tensor(np.random.choice([0, 1], p=[1 - p_x, p_x], size=n)).bool().to(device)
            keys[expl_nodes_subset] = False
            size = keys.sum().item()
            if size > 0:
                x[keys] = all_x[np.random.randint(low=0, high=len(all_x), size=size)].to(device)
            else:
                return None
        if p_edges_del > 0:
            idx = torch.tensor(np.random.choice([0, 1], p=[1 - p_edges_del, p_edges_del], size=m)).bool()
            if idx.sum() == 0:
                idx[torch.randint(0, len(idx), (1,))] = 1
            keep = [
                ((edge_index[0, i] in expl_nodes_subset) or (edge_index[1, i] in expl_nodes_subset))
                for i in range(edge_index.shape[1])
            ]
            keep = torch.tensor(keep).to(device)
            if keep.sum() == len(idx):
                keep[torch.randint(0, len(keep), (1,))] = False

            idx[keep] = False
            if idx.sum() > 0:
                edge_index = remove_edges(edge_index, edge_index[:, idx.bool()])
            else:
                return None
        if p_edges_add > 0:
            new_edges = torch.tensor([[i, j] for i in range(0, len(x)) for j in range(i + 1, len(x))]).T.to(device)
            new_edges = remove_edges(new_edges, edge_index).to(device)
            idx = (
                torch.tensor(np.random.choice([0, 1], p=[1 - p_edges_add, p_edges_add], size=new_edges.shape[-1],))
                .bool()
                .to(device)
            )
            keep = [
                ((new_edges[0, i] in expl_nodes_subset) or (new_edges[1, i] in expl_nodes_subset))
                for i in range(new_edges.shape[-1])
            ]
            keep = torch.tensor(keep).to(device)
            if keep.sum() == len(idx):
                keep[torch.randint(0, len(keep), (1,))] = False
            idx[keep] = False
            if idx.sum() > 0:
                new_edges = new_edges[:, idx]
                edge_index = edge_index.to(device)
                edge_index = torch.cat([edge_index, edge_index[[1, 0]], new_edges, new_edges[[1, 0]]], dim=-1,)
            else:
                return None

        if hasattr(e, "gt_expl_nodes_subset"):
            gt_expl_node_mask = torch.zeros(n, dtype=torch.bool)
            gt_expl_node_mask[e.gt_expl_nodes_subset] = True
        else:
            gt_expl_node_mask = None

        return Data(x=x.cpu(), edge_index=edge_index.cpu(), y=y.cpu(), gt_expl_node_mask=gt_expl_node_mask,)

    perturbed = list()
    original = list()
    with torch.no_grad():
        for e in explanations:
            count = 0
            prev_pred = e.y_pred.to(device).flatten()
            if torch.abs(prev_pred.sum() - 1) > 1e-6:
                prev_pred = torch.softmax(prev_pred, dim=-1)
            for _ in range(n_samples):
                data = _add_noise_perturb(e)
                if data is None:
                    continue
                if preprocess_graph is not None:
                    data = preprocess_graph(data)
                batch = Batch.from_data_list([data]).to(device)  # if collate_fn is None else collate_fn([data])
                pred = forward_fn(model, batch)[0]
                if pred.argmax() == prev_pred.argmax():
                    perturbed.append(data)
                    original.append(e)
                    count += 1
                    if count == max_samples:
                        break
    return original, perturbed


def print_results(results):
    print(",".join(f"{k}" for k in results.keys()))
    print(",".join(f"{v:.4f}" if v is not None else "nan" for v in results.values()))


def get_fn(str_fn):
    fn, param = str_fn.split(":") if (":" in str_fn) else (str_fn, 0)
    return {
        "elbow": lambda x: find_elbow(x, softmax=False),
        "elbow_softmax": lambda x: find_elbow(x, softmax=True),
        "percentile": lambda x: find_percentile(x, float(param), softmax=False),
        "percentile_softmax": lambda x: find_percentile(x, float(param), softmax=True),
        "topk": lambda x: find_topk(x, int(param)),
        "none": None,
    }[fn]


def get_responses_and_logits(model, dataloader, forward_fn):
    logits, responses = list(), list()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            l, r = forward_fn(model, data)
            responses.append(r), logits.append(l)
    return torch.cat(responses, 0), torch.cat(logits)


def remove_explanation_perturb_old(explanations, node_mask_fn, edge_mask_fn, skip_last_node):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _remove_explanation_perturb(e):
        e = e.to(device)
        subset, edge_index = extract_subset_and_edge_index(
            e.edge_index,
            len(e.x) - int(skip_last_node),
            e.node_mask.sum(1)[:-1] if skip_last_node else e.node_mask.sum(1),
            node_mask_fn,
            e.edge_mask.sum(1),
            edge_mask_fn,
        )
        edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True, num_nodes=len(e.x))

        y_pred = e.pred.argmax().unsqueeze(0)
        if len(subset) > 0:
            with_explanation = Data(edge_index=edge_index, x=e.x[subset], y=y_pred)
        else:
            with_explanation = None
        mask = torch.ones(len(e.x), dtype=torch.bool, device=device)
        mask[subset] = False
        if skip_last_node:
            mask[-1] = False
        subset = torch.arange(len(e.x), device=device)[mask]
        if len(subset) == 0:
            without_explanation = None
        else:
            edge_index, _ = subgraph(subset, e.edge_index, relabel_nodes=True, num_nodes=len(e.x))
            without_explanation = Data(edge_index=edge_index, x=e.x[subset], y=y_pred)
        return with_explanation, without_explanation

    with_without_dataset = list(map(_remove_explanation_perturb, explanations))
    with_without_dataset = zip(*with_without_dataset)
    return with_without_dataset


def with_without_explanation_perturb(explanations, keep_expl=True, num_samples=10, p=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _explanation_perturb(e):
        e = e.to(device)
        num_nodes = len(e.x)
        nodes = e.expl_nodes_subset
        prob = torch.full((num_nodes,), p, device=device)  # Probability of 1 for each element

        mask = torch.bernoulli(prob).bool()
        if mask.sum() == 0:
            mask[torch.randint(0, len(mask), (1,))] = 1
        mask[nodes] = keep_expl
        subset = torch.arange(num_nodes, device=device)[mask]
        edge_index, _ = subgraph(subset, e.edge_index, relabel_nodes=True, num_nodes=num_nodes)
        y_pred = e.y_pred.argmax().unsqueeze(0)
        if edge_index.shape[-1] == 0:
            return None
        data = Data(edge_index=edge_index, x=e.x[subset], y=y_pred)
        return data

    perturbed = list()
    for _ in range(num_samples):
        perturbed += list(map(_explanation_perturb, explanations))
    return perturbed


def closest_prototype_acc(model, cls, dataloader, forward_fn):
    acc, count = 0, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls = cls.to(device)
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            sim = forward_fn(model, data)
            acc += (cls[sim.argmax(-1)] == data.y).float().sum()
            count += len(data.y)
    return (acc.item() / count) if count > 0 else 0.0


def compare_with_gt_model(explanations, ground_truths, distance_fn):
    distances = list()
    for e in explanations:
        e_list = [e] * len(ground_truths)
        distances += [min(compare_lists(ground_truths, e_list, distance_fn))]
    return distances


def compare_with_gt_instance(truncated_explanations, distance_fn):
    distances = list()
    for e in truncated_explanations:
        if hasattr(e, "gt_expl_nodes_subset"):
            distances.append(distance_fn(e.expl_nodes_subset, e.gt_expl_nodes_subset))
    return distances


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def remove_explanation_perturb(explanations, node_mask_fn, edge_mask_fn, features_mask_fn):
    default_features = torch.cat([e.x for e in explanations]).mean(0)

    def _remove_explanation_perturb(e):
        node_mask = e.node_mask.sum(1) if hasattr(e, "node_mask") else None
        subset, edge_index = extract_subset_and_edge_index(
            e.edge_index, len(e.x), node_mask, node_mask_fn, e.edge_mask, edge_mask_fn,
        )
        edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True, num_nodes=len(e.x))

        with_x, without_x = e.x, e.x
        if hasattr(e, "mask"):
            thr = features_mask_fn(e.mask)
            if thr is not None:
                features_keep = torch.arange(len(e.mask))[e.mask >= thr]
                features_other = [i for i in range(len(e.mask)) if i not in features_keep]
                for i in features_other:
                    with_x[:, i] = default_features[i]
                for i in features_keep:
                    without_x[:, i] = default_features[i]

        y_pred = e.pred.argmax(keepdim=True)[0]
        if len(subset) > 0:
            with_explanation = Data(edge_index=edge_index, x=with_x[subset], y=y_pred)
        else:
            with_explanation = None
        mask = torch.ones(len(e.x), dtype=torch.bool, device=device)
        mask[subset] = False
        subset = torch.arange(len(e.x), device=device)[mask]
        if len(subset) == 0:
            without_explanation = None
        else:
            edge_index, _ = subgraph(subset, e.edge_index, relabel_nodes=True, num_nodes=len(e.x))
            without_explanation = Data(edge_index=edge_index, x=without_x[subset], y=y_pred)
        return with_explanation, without_explanation

    with_without_dataset = list(map(_remove_explanation_perturb, explanations))
    with_without_dataset = zip(*with_without_dataset)
    return with_without_dataset


def get_explanation_dataset(explanations):
    dataset = list()
    for e in explanations:
        edge_index, _ = subgraph(e.expl_nodes_subset, e.edge_index, relabel_nodes=True, num_nodes=len(e.x))
        dataset.append(Data(x=e.x[e.expl_nodes_subset], edge_index=edge_index, y=e.y_pred.argmax()))
    return dataset


def measure_size(explanations):
    sizes = list()
    for e in explanations:
        sizes.append(len(e.expl_nodes_subset) / len(e.x))
    return sizes


def get_truncated_explanations(explanations, node_mask_fn, edge_mask_fn, skip_last=False):
    def _get_truncated_explanations(e):
        expl_nodes_subset = extract_explanation_nodes_subset(e, node_mask_fn, edge_mask_fn, skip_last)
        assert len(expl_nodes_subset) > 0
        num_nodes = len(e.x)
        gt_expl_nodes_subset = (
            torch.arange(num_nodes, device=e.gt_expl_node_mask.device)[e.gt_expl_node_mask]
            if hasattr(e, "gt_expl_node_mask")
            else None
        )
        adj = e.adj if hasattr(e, "adj") else None
        nidx = e.nidx if hasattr(e, "nidx") else None
        subgraphs_x = e.subgraphs_x if hasattr(e, "subgraphs_x") else None
        subgraphs_adj = e.subgraphs_adj if hasattr(e, "subgraphs_adj") else None
        subgraphs_nidx = e.subgraphs_x if hasattr(e, "subgraphs_nidx") else None
        return Data(
            edge_index=e.edge_index,
            x=e.x,
            y=e.y,
            y_pred=e.y_pred,
            gt_expl_nodes_subset=gt_expl_nodes_subset,
            expl_nodes_subset=expl_nodes_subset,
            adj=adj,
            nidx=nidx,
            subgraphs_x=subgraphs_x,
            subgraphs_adj=subgraphs_adj,
            subgraphs_nidx=subgraphs_nidx,
        )

    return list(map(_get_truncated_explanations, explanations))


def get_gt_fn(dataset_name):
    get_gt_fns = {
        "MUTAG": get_mutag_gt,
        "BAMultiShapes": get_gt_explanation_ba_multishapes,
        "BA-2motif": get_gt_explanation_ba_2motif,
    }
    if dataset_name in get_gt_fns:
        return get_gt_fns[dataset_name]
    else:
        return None


def get_edges_ratio(dataset):
    p = [d.edge_index.shape[-1] / (len(d.x) * (len(d.x) - 1)) for d in dataset]
    return np.mean(p)
