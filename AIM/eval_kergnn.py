import argparse
from tqdm import tqdm

import sys

sys.path.insert(2, f"../../kergnns")
from model import kergnn

import os
import numpy as np

from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GDataLoader
from torch_geometric.utils import dense_to_sparse, to_dense_adj, subgraph

from distances import iou_distance, weighted_ged_distance
from eval_explainer import get_explanations as explainer_get_explanations
from utils import *

KERGNN_PATH = f"../..//kergnns"
METRICS_NAMES = list(METRICS.keys())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--split", type=int, default=-1, help="Split index")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument(
        "--explainer", type=str, required=True, default="SHAP", choices=EXPLAINERS + ["SHAP"], help="Type of explainer",
    )
    parser.add_argument(
        "--metrics", type=str, nargs="+", choices=METRICS_NAMES, default=METRICS_NAMES, help="Metrics to evaluate",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--node_mask_fn", type=str, default="none", help="Thresholding of nodes",
    )
    parser.add_argument("--edge_mask_fn", type=str, default="none", help="Thresholding of edges")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()
    return args


class WrapperModel(nn.Module):
    def __init__(self, model, forward_fn, k=2, size_subgraph=10):
        super().__init__()
        self.model = model
        self.forward_fn = forward_fn
        self.k = k
        self.size_subgraph = size_subgraph

    def forward(self, x, edge_index, batch):
        x, adj, nidx = extract_subgraphs(edge_index, x, self.k, self.size_subgraph)
        data = Data(x=x, edge_index=edge_index, adj=adj, nidx=nidx, batch=batch)
        out = self.forward_fn(self.model, data)
        return out


class Classifier(torch.nn.Module):
    def __init__(self, model):
        super(Classifier, self).__init__()
        self.model = copy.deepcopy(model)

    def forward(self, x):
        preds = 0
        idx = 0
        for layer in range(len(self.model.linears_prediction)):
            try:
                in_features = self.model.linears_prediction[layer].linear.in_features
            except:
                in_features = self.model.linears_prediction[layer].linears[0].in_features
            h = x[:, idx : idx + in_features]
            idx += in_features
            preds += F.dropout(self.model.linears_prediction[layer](h), self.model.dropout_rate, training=False,)
        return preds


def forward(model, classifier, data):

    x, adj, nidxs, batch = data.x, data.adj, data.nidx, data.batch
    nidx = nidxs.clone()
    nidx[nidx < 0] = len(x)

    unique, counts = torch.unique(batch, return_counts=True)
    n_graphs = unique.size(0)

    responses = [x]
    mlp_inputs = [x]
    h = x

    for layer in range(model.num_layers):
        h = model.ker_layers[layer](adj, h, nidx)
        responses.append(h)
        h = model.batch_norms[layer](h)
        h = F.relu(h)
        mlp_inputs.append(h)

    def pool(x):
        pooled = list()
        for layer, h in enumerate(x):
            pooled_h = torch.zeros(batch.max().item() + 1, h.shape[1], device=device).index_add_(0, batch, h)
            if not model.no_norm:
                norm = counts.unsqueeze(1).repeat(1, pooled_h.shape[1])
                pooled_h = pooled_h / norm
            pooled += [pooled_h]
        return torch.cat(pooled, -1)

    pooled_mlp_inputs = pool(mlp_inputs)
    pooled_responses = pool(responses)
    mlp_inputs = torch.cat(mlp_inputs, dim=1)
    responses = torch.cat(responses, dim=1)
    preds = classifier(pooled_mlp_inputs)
    return preds, responses, pooled_responses, mlp_inputs, pooled_mlp_inputs, batch


def get_responses_and_logits(model, dataloader, forward_fn):
    logits, responses = list(), list()
    for data in dataloader:
        data = data.to(device)
        l, r = forward_fn(model, data)
        responses.append(r), logits.append(l)
    return torch.cat(responses, 0), torch.cat(logits)


def get_prototypes(model, dataset):
    prototypes = list()
    all_x = torch.cat([data.x for data in dataset])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    possible_x = torch.unique(all_x, sorted=False, dim=0).to(device)

    for ker_layer in model.ker_layers:
        if ker_layer.hidden_dim:
            encoded_possible_x = torch.nn.ReLU()(ker_layer.fc_in(possible_x))
        x = ker_layer.features_hidden.permute(0, 2, 1)
        x_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        sim = encoded_possible_x @ x.T
        x = possible_x[sim.argmax(0)]
        x = x.reshape(x_shape[0], x_shape[1], -1)
        adj_hidden_norm = torch.zeros(ker_layer.size_graph_filter, ker_layer.size_graph_filter, ker_layer.out_dim).to(
            device
        )
        idx = torch.triu_indices(ker_layer.size_graph_filter, ker_layer.size_graph_filter, 1)
        adj_hidden_norm[idx[0], idx[1], :] = ker_layer.relu(ker_layer.adj_hidden)
        adj = adj_hidden_norm + torch.transpose(adj_hidden_norm, 0, 1)
        adj[adj < 0] = 0
        for i in range(adj.shape[0]):
            edge_index = dense_to_sparse(adj[i] > 0.0)[0]
            edge_attr = torch.zeros(edge_index.shape[1]).to(edge_index.device)
            for j in range(edge_index.shape[1]):
                edge_attr[j] = float(min(adj[i, edge_index[0, j], edge_index[1, j]], 1) > 0)
            prototypes.append(Data(edge_index=edge_index, x=x[i], edge_attr=edge_attr))
    return prototypes


def perturb_model(model, dataset, p_x=0.2, p_edge=0.2):
    model_copy = copy.deepcopy(model)
    model_copy.eval()
    all_x = torch.cat([data.x for data in dataset])

    for ker_layer in model_copy.ker_layers:
        x = ker_layer.features_hidden.detach()
        adj = ker_layer.adj_hidden.detach()
        if p_x > 0:
            x = x.permute((0, 2, 1))
            x_shape = x.shape
            x = x.reshape(-1, x.shape[-1])
            keys = torch.tensor(np.random.choice([0, 1], p=[1 - p_x, p_x], size=len(x))).bool()
            size = keys.sum().item()
            if size > 0:
                x[keys] = torch.nn.ReLU()(
                    ker_layer.fc_in(all_x[np.random.randint(low=0, high=len(all_x), size=size)].to(x.device))
                )

            ker_layer.features_hidden.data = x.reshape(x_shape[0], x_shape[1], -1).permute((0, 2, 1))
        if p_edge > 0:
            adj_shape = adj.shape
            adj = adj.reshape(-1)
            keys = torch.tensor(np.random.choice([0, 1], p=[1 - p_edge, p_edge], size=len(adj))).bool().to(adj.device)
            adj[keys] = (adj[keys] < 0).float()
            ker_layer.adj_hidden.data = adj.reshape(adj_shape)
    return model_copy


def load_model(path, dataset_name):
    params = torch.load(path, map_location=torch.device("cpu"))
    if "args" in params:
        loaded_args = params["args"]
    else:
        loaded_args = {
            "hidden_dims": [16, 32],
            "kernel": "wl",
            "max_stop": 1,
            "num_mlp_layers": 1,
            "mlp_hidden_dim": 16,
            "dropout_rate": 0.4,
            "size_graph_filter": [6],
            "size_subgraph": 10,
            "no_norm": dataset_name in ["MUTAG", "PROTEINS", "NCI1", "PROTEINS_full"],
        }

    loaded_args = argparse.Namespace(**loaded_args)
    features_dim = {
        "MUTAG": 7,
        "PROTEINS": 4,
        "IMDB-BINARY": 1,
        "IMDB-MULTI": 1,
        "BA-2motif": 10,
        "BAMultiShapes": 10,
        "PROTEINS_full": 10,
    }[dataset_name]
    n_classes = {
        "MUTAG": 2,
        "PROTEINS": 2,
        "IMDB-BINARY": 2,
        "IMDB-MULTI": 3,
        "BA-2motif": 2,
        "BAMultiShapes": 2,
        "PROTEINS_full": 2,
    }[dataset_name]
    # Create models
    model = kergnn(
        features_dim,
        n_classes,
        hidden_dims=loaded_args.hidden_dims,
        kernel=loaded_args.kernel,
        max_step=loaded_args.max_step,
        num_mlp_layers=loaded_args.num_mlp_layers,
        mlp_hidden_dim=loaded_args.mlp_hidden_dim,
        dropout_rate=loaded_args.dropout_rate,
        size_graph_filter=loaded_args.size_graph_filter,
        size_subgraph=loaded_args.size_subgraph,
        no_norm=loaded_args.no_norm,
    )
    model.load_state_dict(params["state_dict"])
    model.eval()
    return model, loaded_args


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
    nidx = [torch.nn.functional.pad(n, pad=(0, subgraph_size - len(n)), value=-1) for n in nidx]
    nidx = torch.stack(nidx)
    return x, adj, nidx


def collate_fn(batch):
    graph_indicator = torch.cat(
        tuple(i * torch.ones(len(b.x), dtype=torch.long, device=device) for i, b in enumerate(batch))
    )
    x = torch.cat([b.x for b in batch]).to(device)

    gt_expl_node_mask = (
        torch.cat([b.gt_expl_node_mask for b in batch]).to(device) if hasattr(batch[0], "gt_expl_node_mask") else None
    )
    adj = torch.cat([b.adj for b in batch]).to(device)
    nidx = torch.cat([b.nidx for b in batch]).to(device)
    y = torch.cat([b.y for b in batch]).to(device)

    _, counts = torch.unique(graph_indicator, return_counts=True)
    counts_sum = torch.cumsum(counts, dim=0)[:-1].long()
    nidx_diff = torch.zeros_like(graph_indicator, device=device)
    nidx_diff[counts_sum.long()] = counts[:-1]
    nidx_diff = torch.cumsum(nidx_diff, dim=0)
    nidx[nidx >= 0] = (nidx + nidx_diff.unsqueeze(-1))[nidx >= 0]

    edge_index = list()
    num_nodes = 0
    for b in batch:
        edge_index.append(b.edge_index + num_nodes)
        num_nodes += len(b.x)
    edge_index = torch.cat(edge_index, dim=-1)
    # x = torch.cat([x, torch.zeros_like(x[0:1])], dim=0)
    return Data(
        x=x, edge_index=edge_index, adj=adj, nidx=nidx, batch=graph_indicator, y=y, gt_expl_node_mask=gt_expl_node_mask,
    )


def get_explanations_shap(model, explainer, dataloader, forward_fn):
    explanations = list()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for data in tqdm(dataloader, total=len(dataloader)):
        data = data.to(device)
        num_features = data.x.shape[-1]
        with torch.no_grad():
            logits, all_responses, response = forward_fn(model, data)
        pred = logits.argmax(dim=-1)
        importance = run_shap(explainer, response).to(device)
        importance = torch.stack([importance[pred[i]][i] for i in range(len(pred))], dim=0).to(device)
        response[response == 0] = 1e-36
        gt_expl_node_mask = data.gt_expl_node_mask if hasattr(data, "gt_expl_node_mask") else None
        init_adj, init_nidx = data.adj, data.nidx

        num_nodes = 0
        for b in tqdm(range(len(data.y)), total=len(data.y)):
            mask = data.batch == b
            im, r, ar = importance[b], response[b], all_responses[mask]
            im = (im / r) * ar
            im_features = im[:, :num_features]
            im = im[:, num_features:]

            nidx = data.nidx[mask] - num_nodes
            x = data.x[mask]

            edge_index = data.edge_index[:, data.batch[data.edge_index[0]] == b] - num_nodes
            num_nodes += len(x)
            node_mask = torch.zeros(len(x), im.shape[-1] + im_features.shape[-1], device=device)
            node_mask[:, num_features:] = im
            node_mask[:, :num_features] += im_features
            nidx[nidx == len(data.x)] = -1
            e = Data(
                x=x.cpu(),
                edge_index=edge_index.cpu(),
                y=data.y[b : b + 1].cpu(),
                y_pred=logits[b : b + 1].cpu(),
                node_mask=node_mask.cpu().sum(-1),
                edge_mask=None,  # edge_mask.cpu(),
                gt_expl_node_mask=gt_expl_node_mask[mask] if gt_expl_node_mask is not None else None,
                adj=init_adj[mask],
                nidx=nidx,
            )
            explanations.append(e)
    return explanations


def main():
    args = args_parser()
    print(args)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, loaded_args = load_model(args.model_path, args.dataset)
    args.num_layers = 0
    model = model.to(device)
    model.eval()
    degree_attr = model.ker_layers[0].fc_in.in_features == 1

    dataset = get_dataset(args.dataset, degree_attr=degree_attr)
    classifier = Classifier(model).to(device)
    classifier.eval()
    model = (model, classifier)

    gt_fn = get_gt_fn(args.dataset)

    def preprocess_graph(batch, add_gt_expl=True):
        if add_gt_expl:
            if gt_fn is not None:
                batch = gt_fn(batch)
        x, adj, nidx = extract_subgraphs(batch.edge_index, batch.x, loaded_args.k, loaded_args.size_subgraph)
        y = torch.tensor([batch.y.item()])
        data = Data(
            x=x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            adj=adj,
            nidx=nidx,
            y=y,
            gt_expl_node_mask=batch.gt_expl_node_mask if hasattr(batch, "gt_expl_node_mask") else None,
        )
        return data

    def preprocess_dataset(dataset, add_gt_expl=True):
        return list(map(lambda x: preprocess_graph(x, add_gt_expl), dataset))

    train_idxs, val_idxs, test_idxs = get_splits(args.dataset, size=len(dataset), seed=args.seed, split=args.split)
    dataset_train, dataset_val, dataset_test = (
        Subset(dataset, train_idxs),
        Subset(dataset, val_idxs),
        Subset(dataset, test_idxs),
    )

    dataset_test = preprocess_dataset(dataset_test)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    forward_fn = lambda model, x: forward(model[0], model[1], x)[0]

    results = {}

    if args.explainer == "SHAP":
        get_explanations = get_explanations_shap
        wrapp_model = model
    else:
        wrapp_model = WrapperModel(model, forward_fn)
        wrapp_model.eval()
        get_explanations = lambda m, e, d, f: explainer_get_explanations(m, d, args, model_type="KerGNN")

    if "A" in args.metrics:
        results["A"] = get_acc(model, dataloader_test, forward_fn)
        print_results(results)

    if args.metrics == ["A"]:
        print("Final")
        print_results(results)
        exit(0)

    def logits_and_responses_fn(model, data):
        model, classifier = model
        logits, _, _, _, responses, _ = forward(model, classifier, data)
        return logits, responses

    def logits_and_all_responses_fn(model, data):
        model, classifier = model
        logits, _, _, all_responses, responses, _ = forward(model, classifier, data)
        return logits, all_responses, responses

    dataloader = DataLoader(
        preprocess_dataset(dataset), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
    )
    responses, logits = get_responses_and_logits(model, dataloader, logits_and_responses_fn)

    shap_values, explainer = get_shap(classifier, responses)

    explanations = get_explanations(wrapp_model, explainer, dataloader_test, logits_and_all_responses_fn)
    node_mask_fn = get_fn(args.node_mask_fn)
    edge_mask_fn = get_fn(args.edge_mask_fn)

    truncated_explanations = get_truncated_explanations(explanations, node_mask_fn, edge_mask_fn)
    prototypes = get_prototypes(model[0], dataset)

    if "A1" in args.metrics:
        distances = compare_with_gt_instance(truncated_explanations, iou_distance)
        if len(distances) == 0:
            results["A1"] = None
        else:
            results["A1"] = np.mean(distances)
        print_results(results)

    if "A2" in args.metrics:
        ground_truths = get_gt_explanations_model(args.dataset)
        if len(ground_truths) > 0:
            distances = compare_with_gt_model(prototypes, ground_truths, weighted_ged_distance)
            results["A2"] = 1 - np.mean(distances)
        else:
            results["A2"] = None
        print_results(results)

    if bool({"I1", "I2"} & set(args.metrics)):
        with_explanations = with_without_explanation_perturb(truncated_explanations, keep_expl=True, num_samples=10,)
        without_explanations = with_without_explanation_perturb(truncated_explanations, keep_expl=False, num_samples=10)
        with_explanations = [data for data in with_explanations if data is not None]
        without_explanations = [data for data in without_explanations if data is not None]
        with_explanations = preprocess_dataset(with_explanations, add_gt_expl=False)
        without_explanations = preprocess_dataset(without_explanations, add_gt_expl=False)
        if (len(with_explanations) < 20) or (len(without_explanations) < 20):
            results["I1"] = None
            results["I2"] = None
        else:
            if "I1" in args.metrics:
                dataloader_with = DataLoader(
                    with_explanations, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
                )
                results["I1"] = get_acc(model, dataloader_with, forward_fn)
                print_results(results)
            if "I2" in args.metrics:
                dataloader_without = DataLoader(
                    without_explanations, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
                )
                results["I2"] = 1 - get_acc(model, dataloader_without, forward_fn)
                print_results(results)

    if "I3" in args.metrics:
        original, dataset_noisy_node = add_noise_perturb(
            truncated_explanations,
            forward_fn,
            model,
            p_x=0.1,
            p_edges_add=0.0,
            p_edges_del=0.0,
            preprocess_graph=lambda x: preprocess_graph(x, add_gt_expl=False),
        )
        if len(original) < 20:
            print(f"Only {len(original)} samples")
            results["I3"] = None
        else:
            # dataset_noisy_node = preprocess_dataset(dataset_noisy_node)

            dataloader_new = DataLoader(
                dataset_noisy_node, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
            )
            new_explanations = get_explanations(wrapp_model, explainer, dataloader_new, logits_and_all_responses_fn)
            new_truncated_explanations = get_truncated_explanations(new_explanations, node_mask_fn, edge_mask_fn)
            distances = compare_lists(original, new_truncated_explanations, iou_distance)
            results["I3"] = np.mean(distances)
        print_results(results)

    if "I4" in args.metrics:
        original, dataset_noisy_edge = add_noise_perturb(
            truncated_explanations,
            forward_fn,
            model,
            p_x=0.0,
            p_edges_add=0.1,
            p_edges_del=0.1,
            preprocess_graph=lambda x: preprocess_graph(x, add_gt_expl=False),
        )
        if len(original) < 20:
            print(f"Only {len(original)} samples")
            results["I4"] = None
        else:
            dataloader_new = DataLoader(
                dataset_noisy_edge, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
            )
            new_explanations = get_explanations(wrapp_model, explainer, dataloader_new, logits_and_all_responses_fn)
            new_truncated_explanations = get_truncated_explanations(new_explanations, node_mask_fn, edge_mask_fn)
            distances = compare_lists(original, new_truncated_explanations, iou_distance)
            results["I4"] = np.mean(distances)
        print_results(results)

    if "I5" in args.metrics:

        _, new_explainer = get_shap(classifier, responses)
        new_explanations = get_explanations(wrapp_model, new_explainer, dataloader_test, logits_and_all_responses_fn)
        new_truncated_explanations = get_truncated_explanations(new_explanations, node_mask_fn, edge_mask_fn)
        distances = compare_lists(truncated_explanations, new_truncated_explanations, iou_distance)
        results["I5"] = np.mean(distances)
        print_results(results)

    if "M1" in args.metrics:
        set_seed(args.seed)
        # try:
        perturbed_model = perturb_model(model[0], dataset, p_x=0.5, p_edge=0)

        if args.explainer != "SHAP":
            new_wrapp_model = WrapperModel((perturbed_model, model[1]), forward_fn)
            perturbed_explainer = None
        else:
            perturbed_responses, perturbed_logits = get_responses_and_logits(
                (perturbed_model, model[1]), dataloader, logits_and_responses_fn
            )
            _, perturbed_explainer = get_shap(model[1], perturbed_responses, check_additivity=False)
            new_wrapp_model = (perturbed_model, model[1])
        perturbed_explanations = get_explanations(
            new_wrapp_model, perturbed_explainer, dataloader_test, logits_and_all_responses_fn,
        )
        perturbed_explanations = get_truncated_explanations(perturbed_explanations, node_mask_fn, edge_mask_fn)
        distances = compare_lists(truncated_explanations, perturbed_explanations, iou_distance)
        results["M1"] = 1 - np.mean(distances)
        # except:
        #     results["M1"] = None
        print_results(results)

    if "M2" in args.metrics:
        set_seed(args.seed)
        # try:
        perturbed_model = perturb_model(model[0], dataset, p_x=0, p_edge=0.5)

        if args.explainer != "SHAP":
            new_wrapp_model = WrapperModel((perturbed_model, model[1]), forward_fn)
            perturbed_explainer = None
        else:
            perturbed_responses, perturbed_logits = get_responses_and_logits(
                (perturbed_model, model[1]), dataloader, logits_and_responses_fn
            )
            _, perturbed_explainer = get_shap(model[1], perturbed_responses, check_additivity=False)
            new_wrapp_model = (perturbed_model, model[1])
        perturbed_explanations = get_explanations(
            new_wrapp_model, perturbed_explainer, dataloader_test, logits_and_all_responses_fn,
        )
        perturbed_explanations = get_truncated_explanations(perturbed_explanations, node_mask_fn, edge_mask_fn)
        distances = compare_lists(truncated_explanations, perturbed_explanations, iou_distance)
        results["M2"] = 1 - np.mean(distances)
        # except:
        #     results["M2"] = None
        print_results(results)

    if "M3" in args.metrics:
        results["M3"] = 1 - pairwise_list_corr(responses.cpu().detach().numpy(), to_abs=True)
        print_results(results)

    print("Final")
    print_results(results)


if __name__ == "__main__":
    main()
