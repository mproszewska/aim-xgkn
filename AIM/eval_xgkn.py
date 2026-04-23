import argparse
from tqdm import tqdm

import sys

sys.path.insert(1, f"../XGKN")

import os
import importlib.util

spec1 = importlib.util.spec_from_file_location("gkn_utils", f"../XGKN/utils.py")
gkn_utils = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(gkn_utils)
from models import GKNetwork

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse

from distances import iou_distance, weighted_ged_distance
from eval_explainer import get_explanations as explainer_get_explanations
from utils import *

GKN_PATH = f"../XGKN"
METRICS_NAMES = list(METRICS.keys())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--split", type=int, default=-1, help="Split index")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument(
        "--explainer", type=str, required=True, default="SHAP", choices=EXPLAINERS + ["SHAP"], help="Type of explainer",
    )
    parser.add_argument(
        "--metrics", type=str, nargs="+", choices=METRICS_NAMES, default=METRICS_NAMES, help="Metrics to evaluate",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--node_mask_fn", type=str, default="elbow_softmax:0", help="Thresholding of nodes",
    )
    parser.add_argument("--edge_mask_fn", type=str, default="none", help="Thresholding for edges")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()
    return args


class WrapperModel(nn.Module):
    def __init__(self, model, forward_fn, preprocess_graph):
        super().__init__()
        self.model = model
        self.forward_fn = forward_fn
        self.preprocess_graph = preprocess_graph

    def forward(self, x, edge_index, batch):
        data = Data(x=x, edge_index=edge_index)
        data = self.preprocess_graph(data)
        data = Batch.from_data_list([data])
        return self.forward_fn(self.model, data)


def get_responses_and_logits(model, dataloader, forward_fn):
    logits, responses = list(), list()
    for data in dataloader:
        data = data.to(device)
        l, r = forward_fn(model, data)
        responses.append(r), logits.append(l)
    return torch.cat(responses, 0), torch.cat(logits)


def get_prototypes(model, dataset):
    prototypes = list()
    all_x = torch.cat([data.x.clone().cpu() for data in dataset])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    possible_x = torch.unique(all_x, sorted=False, dim=0).to(device)

    for ker_layer in model.ker_layers:
        encoded_possible_x = ker_layer.encoder(possible_x)
        x = ker_layer.x_hidden()
        x_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        sim = encoded_possible_x @ x.T
        x = possible_x[sim.argmax(0)]
        x = x.reshape(x_shape[0], x_shape[1], -1)
        adj = ker_layer.adj_hidden()
        for i in range(adj.shape[0]):
            edge_index = dense_to_sparse(adj[i, :, :] > 0.0)[0]
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
        x = ker_layer._x_hidden.detach()
        adj = ker_layer._adj_hidden.detach()
        if p_x > 0:
            x = x.permute((0, 2, 1))
            x_shape = x.shape
            x = x.reshape(-1, x.shape[-1])
            keys = torch.tensor(np.random.choice([0, 1], p=[1 - p_x, p_x], size=len(x))).bool()
            size = keys.sum().item()
            if size > 0:
                x[keys] = ker_layer.encoder(all_x[np.random.randint(low=0, high=len(all_x), size=size)].to(x.device))
            ker_layer._x_hidden.data = x.reshape(x_shape[0], x_shape[1], -1).permute((0, 2, 1))
        if p_edge > 0:
            adj_shape = adj.shape
            adj = adj.reshape(-1)
            keys = torch.tensor(np.random.choice([0, 1], p=[1 - p_edge, p_edge], size=len(adj))).bool().to(adj.device)
            adj[keys] = (adj[keys] < 0).float()
            ker_layer._adj_hidden.data = adj.reshape(adj_shape)
    return model_copy


def get_explanations_shap(model, explainer, dataloader, forward_fn):
    explanations = list()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for data in tqdm(dataloader, total=len(dataloader)):
        data = data.to(device)
        with torch.no_grad():
            logits, all_responses, response = forward_fn(model, data)
        pred = logits.argmax(dim=-1)
        importance = run_shap(explainer, response).to(device)
        importance = torch.stack([importance[pred[i]][i] for i in range(len(pred))], dim=0).to(device)
        response[response == 0] = 1e-36
        gt_expl_node_mask = data.gt_expl_node_mask if hasattr(data, "gt_expl_node_mask") else None

        num_nodes = 0
        for b in tqdm(range(len(data.y)), total=len(data.y)):
            mask = data.batch == b
            im, r, ar = importance[b], response[b], all_responses[mask]
            im = (im / r) * ar
            x = data.x[mask]
            edge_index = data.edge_index[:, data.batch[data.edge_index[0]] == b] - num_nodes
            nidx = data.subgraphs_nidx[mask]  # - num_nodes
            num_nodes += len(x)
            node_mask = im
            e = Data(
                x=x.cpu(),
                edge_index=edge_index.cpu(),
                y=data.y[b : b + 1].cpu(),
                y_pred=logits[b : b + 1].cpu(),
                node_mask=node_mask.sum(-1).cpu(),
                edge_mask=None,  # edge_mask.cpu(),
                gt_expl_node_mask=gt_expl_node_mask[mask].cpu() if gt_expl_node_mask is not None else None,
            )
            explanations.append(e)
    return explanations


def main():
    args = args_parser()
    print(args)

    params = torch.load(args.model_path, map_location=torch.device("cpu"))
    model = GKNetwork(**params["model_args"]).to(device)
    model.load_state_dict(params["state_dict"], strict=False)
    model.eval()
    degree_attr = params["model_args"]["in_features"] == 1
    dataset = get_dataset(args.dataset, degree_attr=degree_attr)
    k, subgraph_size = params["args"]["k"], params["args"]["subgraph_size"]

    gt_fn = get_gt_fn(args.dataset)

    def preprocess_graph(data, add_gt_expl=True):
        if add_gt_expl and (gt_fn is not None):
            data = gt_fn(data)
            device = data.gt_expl_node_mask.device
            data.gt_expl_node_mask = torch.cat(
                [data.gt_expl_node_mask, torch.zeros(1, dtype=torch.bool, device=device)], dim=-1
            )
        if not add_gt_expl:
            data.gt_expl_node_mask = None
        data = gkn_utils.transform(data, k, subgraph_size, node_degree_label=False)

        return data

    def preprocess_dataset(dataset, add_gt_expl=True):
        return list(map(lambda x: preprocess_graph(x, add_gt_expl), dataset))

    train_idxs, val_idxs, test_idxs = get_splits(args.dataset, size=len(dataset), seed=args.seed, split=args.split)
    dataset_train, dataset_val, init_dataset_test = (
        Subset(dataset, train_idxs),
        Subset(dataset, val_idxs),
        Subset(dataset, test_idxs),
    )
    preprocessed_dataset_test = preprocess_dataset(init_dataset_test)
    dataset_test = preprocessed_dataset_test if args.explainer == "SHAP" else init_dataset_test

    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)
    preprocessed_dataloader_test = DataLoader(preprocessed_dataset_test, batch_size=args.batch_size, shuffle=False)

    def forward_fn(model, data):
        return model(data)[0]

    results = {}

    if args.explainer == "SHAP":
        get_explanations = get_explanations_shap
        wrapp_model = model
    else:
        wrapp_model = WrapperModel(model, forward_fn, preprocess_graph)
        wrapp_model.eval()
        get_explanations = lambda m, e, d, f: explainer_get_explanations(m, d, args, model_type="XGKN")

    if "A" in args.metrics:
        results["A"] = get_acc(model, preprocessed_dataloader_test, forward_fn)
        print_results(results)

    def logits_and_responses_fn(model, data):
        logits, _, responses, _ = model(data)
        return logits, responses

    def logits_and_all_responses_fn(model, data):
        logits, _, responses, all_responses = model(data)
        return logits, all_responses[0][0], responses

    dataloader = DataLoader(preprocess_dataset(dataset), batch_size=args.batch_size, shuffle=False)

    responses, logits = get_responses_and_logits(model, dataloader, logits_and_responses_fn)

    shap_values, explainer = get_shap(model.mlp, responses)

    explanations = get_explanations(wrapp_model, explainer, dataloader_test, logits_and_all_responses_fn)

    node_mask_fn = get_fn(args.node_mask_fn)
    edge_mask_fn = get_fn(args.edge_mask_fn)

    truncated_explanations = get_truncated_explanations(explanations, node_mask_fn, edge_mask_fn, skip_last=True,)
    prototypes = get_prototypes(model, dataset)

    if "A1" in args.metrics:
        if args.dataset in ["MUTAG", "BA-2motif", "BAMultiShapes"]:
            distances = compare_with_gt_instance(truncated_explanations, iou_distance)
            results["A1"] = np.mean(distances)
        else:
            results["A1"] = None
        print_results(results)

    if "A2" in args.metrics:
        if args.dataset in ["MUTAG", "BA-2motif", "BAMultiShapes"]:
            ground_truths = get_gt_explanations_model(args.dataset)
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
                dataloader_with = DataLoader(with_explanations, batch_size=args.batch_size, shuffle=False)
                results["I1"] = get_acc(model, dataloader_with, forward_fn)
                print_results(results)
            if "I2" in args.metrics:
                dataloader_without = DataLoader(without_explanations, batch_size=args.batch_size, shuffle=False)
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
            preprocess_graph=lambda x: preprocess_graph(x, False),
            skip_last=True,
        )
        if len(original) < 20:
            print(f"Only {len(original)} samples")
            results["I3"] = None
        else:
            dataloader_new = DataLoader(dataset_noisy_node, batch_size=args.batch_size, shuffle=False)
            new_explanations = get_explanations(wrapp_model, explainer, dataloader_new, logits_and_all_responses_fn)
            new_truncated_explanations = get_truncated_explanations(
                new_explanations, node_mask_fn, edge_mask_fn, skip_last=True,
            )
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
            preprocess_graph=lambda x: preprocess_graph(x, False),
            skip_last=True,
        )
        if len(original) < 20:
            print(f"Only {len(original)} samples")
            results["I4"] = None
        else:
            dataloader_new = DataLoader(dataset_noisy_edge, batch_size=args.batch_size, shuffle=False)
            new_explanations = get_explanations(wrapp_model, explainer, dataloader_new, logits_and_all_responses_fn)
            new_truncated_explanations = get_truncated_explanations(
                new_explanations, node_mask_fn, edge_mask_fn, skip_last=True,
            )
            distances = compare_lists(original, new_truncated_explanations, iou_distance)

            results["I4"] = np.mean(distances)
        print_results(results)

    if "I5" in args.metrics:
        new_explanations = get_explanations(wrapp_model, explainer, dataloader_test, logits_and_all_responses_fn)
        new_truncated_explanations = get_truncated_explanations(
            new_explanations, node_mask_fn, edge_mask_fn, skip_last=True,
        )
        distances = compare_lists(truncated_explanations, new_truncated_explanations, iou_distance)
        results["I5"] = np.mean(distances)
        print_results(results)

    if "M1" in args.metrics:
        perturbed_model = perturb_model(model, dataset, p_x=0.5, p_edge=0)
        perturbed_responses, perturbed_logits = get_responses_and_logits(
            perturbed_model, dataloader, logits_and_responses_fn
        )
        _, perturbed_explainer = get_shap(perturbed_model.mlp, perturbed_responses)

        perturbed_explanations = get_explanations(
            WrapperModel(perturbed_model, forward_fn, preprocess_graph)
            if args.explainer != "SHAP"
            else perturbed_model,
            perturbed_explainer,
            dataloader_test,
            logits_and_all_responses_fn,
        )
        perturbed_explanations = get_truncated_explanations(
            perturbed_explanations, node_mask_fn, edge_mask_fn, skip_last=True,
        )
        distances = compare_lists(truncated_explanations, perturbed_explanations, iou_distance)
        results["M1"] = 1 - np.mean(distances)
        print_results(results)

    if "M2" in args.metrics:
        perturbed_model = perturb_model(model, dataset, p_x=0, p_edge=0.5)
        perturbed_responses, perturbed_logits = get_responses_and_logits(
            perturbed_model, dataloader, logits_and_responses_fn
        )
        _, perturbed_explainer = get_shap(perturbed_model.mlp, perturbed_responses)
        perturbed_explanations = get_explanations(
            WrapperModel(perturbed_model, forward_fn, preprocess_graph)
            if args.explainer != "SHAP"
            else perturbed_model,
            perturbed_explainer,
            dataloader_test,
            logits_and_all_responses_fn,
        )
        perturbed_explanations = get_truncated_explanations(
            perturbed_explanations, node_mask_fn, edge_mask_fn, skip_last=True,
        )
        distances = compare_lists(truncated_explanations, perturbed_explanations, iou_distance)
        results["M2"] = 1 - np.mean(distances)
        print_results(results)

    if "M3" in args.metrics:
        results["M3"] = 1 - pairwise_list_corr(responses.cpu().detach().numpy(), to_abs=True)
        print_results(results)

    print("Final")
    print_results(results)


if __name__ == "__main__":
    main()
