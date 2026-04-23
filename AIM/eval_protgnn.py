import argparse
from tqdm import tqdm

import sys

sys.path.insert(2, f"../../ProtGNN")
import os
import load_dataset
from models import GnnNets
from my_mcts import mcts

import numpy as np
from torch import nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn.conv import MessagePassing

from distances import iou_distance, weighted_ged_distance
from utils import *

PROTGNN_PATH = f"../../ProtGNN"
METRICS_NAMES = list(METRICS.keys())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--split", type=int, default=-1, help="Split index")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument(
        "--explainer", type=str, default="SHAP", choices=EXPLAINERS + ["SHAP"], help="Type of explainer",
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


def fix_model_args(model_args, params):
    model_args.model_name = "gin"
    model_args.latent_dim = [params[f"model.gnn_layers.{i}.nn.0.weight"].shape[0] for i in range(3)]

    model_args.num_prototypes_per_class = (
        params["model.prototype_vectors"].shape[0] // params["model.last_layer.weight"].shape[0]
    )
    return model_args


def fix_data_args(data_args, args):
    data_args.dataset_name = args.dataset
    data_args.random_split = args.dataset == "MUTAG"
    data_args.seed = args.seed
    return data_args


def get_explanations_shap(model, explainer, dataloader, forward_fn, seed):
    explanations = list()
    set_seed(seed)
    for data in tqdm(dataloader, total=len(dataloader)):
        data = data.to(device)
        with torch.no_grad():
            logits, response = forward_fn(model, data)
        pred = logits.argmax(dim=-1)
        importance = run_shap(explainer, response).to(device)
        importance = torch.stack([importance[pred[i]][i] for i in range(len(pred))], dim=0)
        num_nodes = 0
        for b in tqdm(range(len(data.y)), total=len(data.y)):
            im = importance[b]
            mask = data.batch == b
            edge_index = data.edge_index[:, data.batch[data.edge_index[0]] == b] - num_nodes
            d = Data(x=data.x[mask], edge_index=edge_index, y=data.y[b]).to(device)
            num_nodes += mask.sum()
            node_mask = torch.zeros(len(d.x), len(im), device=device)
            for i in range(model.model.last_layer.in_features):
                coalition, similarity, prot = mcts(d, model, model.model.prototype_vectors[i])
                norm = len(coalition)
                if norm == 0:
                    continue
                node_mask[coalition, i] += im[i] / norm

            x, edge_index, y, pred = (
                d.x.clone(),
                d.edge_index.clone(),
                d.y.clone(),
                logits[b].clone(),
            )
            e = Data(
                x=x.cpu(),
                edge_index=edge_index.cpu(),
                y=y.cpu(),
                y_pred=pred.cpu(),
                node_mask=node_mask.sum(-1).cpu(),
                edge_mask=None,  # edge_mask.sum(-1),
                gt_expl_node_mask=data.gt_expl_node_mask[mask].cpu() if hasattr(data, "gt_expl_node_mask") else None,
            )
            explanations.append(e)
    return explanations


def get_prototypes(model, dataset, num_prototypes_per_class):
    prototypes = list()
    for i in range(model.model.last_layer.in_features):
        count = 0
        best_similarity = 0
        label = i // num_prototypes_per_class
        prototypes.append(None)
        for data in tqdm(dataset, total=len(dataset)):
            data = data.to(device)
            if data.y == label:
                count += 1
                coalition, similarity, prot = mcts(data, model, model.model.prototype_vectors[i])
                if similarity > best_similarity:
                    best_similarity = similarity
                    edge_index, _ = subgraph(coalition, data.edge_index, relabel_nodes=True, num_nodes=len(data.x),)
                    prototypes[i] = Data(x=data.x[coalition], edge_index=edge_index, y=data.y)
    return prototypes


def perturb_model(model, prototypes, dataset, p_x=0.2, p_edge=0.2):
    model_copy = copy.deepcopy(model)
    model_copy.eval()
    all_x = torch.cat([data.x.cpu() for data in dataset])
    prototype_vectors = model_copy.model.prototype_vectors.detach()
    for i in range(prototype_vectors.shape[0]):
        x, edge_index = prototypes[i].x, prototypes[i].edge_index
        if p_x > 0:
            keys = torch.tensor(np.random.choice([0, 1], p=[1 - p_x, p_x], size=len(x))).bool().to(x.device)
            size = keys.sum().item()
            if size > 0:
                x[keys] = all_x[np.random.randint(low=0, high=len(all_x), size=size)].to(x.device)
        if p_edge > 0:
            adj = to_dense_adj(edge_index)[0]
            idx = torch.triu_indices(adj.shape[0], adj.shape[0], 1)
            flatten_adj = adj[idx[0], idx[1]]
            keys = (
                torch.tensor(np.random.choice([0, 1], p=[1 - p_edge, p_edge], size=len(flatten_adj)))
                .bool()
                .to(flatten_adj.device)
            )
            flatten_adj[keys] = 1 - flatten_adj[keys]
            adj = torch.zeros_like(adj)
            adj[idx[0], idx[1]] = flatten_adj
            adj = adj + adj.T
            edge_index = dense_to_sparse(adj)[0]
        data = Data(x=x, edge_index=edge_index, batch=torch.zeros(len(x), dtype=torch.long)).to(x.device)
        _, _, _, emb, _ = model(data)
        prototype_vectors[i] = emb
    model_copy.model.prototype_vectors.data = prototype_vectors
    return model_copy


def main():
    from Configures import data_args, train_args, model_args

    args = args_parser()
    set_seed(args.seed)
    print(args)

    dataset = load_dataset.get_dataset(f"datasets/", args.dataset, task=None)
    num_node_features, num_classes = dataset.num_node_features, dataset.num_classes
    if args.dataset.startswith("IMDB"):
        num_node_features = 1

    dataset_name = {"BA_2Motifs": "BA-2motif"}
    if args.dataset in dataset_name:
        dataset_name = dataset_name[args.dataset]
    else:
        dataset_name = args.dataset
    gt_fn = get_gt_fn(dataset_name)

    def preprocess_graph(batch, add_gt_expl=True):
        if args.dataset.startswith("IMDB"):
            batch.x = to_dense_adj(batch.edge_index)[0].sum(-1).unsqueeze(-1)
        if add_gt_expl and (gt_fn is not None):
            batch = gt_fn(batch)
        return batch

    def preprocess_dataset(dataset):
        return list(map(lambda x: preprocess_graph(x), dataset))

    dataset = preprocess_dataset(dataset)

    train_idxs, val_idxs, test_idxs = get_splits(args.dataset, size=len(dataset), seed=args.seed, split=args.split)
    dataset_train, dataset_val, dataset_test = (
        Subset(dataset, train_idxs),
        Subset(dataset, val_idxs),
        Subset(dataset, test_idxs),
    )

    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    params = torch.load(args.model_path)
    params = params["net"]
    model_args = fix_model_args(model_args, params)
    model = GnnNets(num_node_features, num_classes, model_args).to(device)
    model.update_state_dict(params)
    model.eval()
    forward_fn = lambda model, data: model(data)[0]
    results = {}

    for module in model.modules():
        if isinstance(module, MessagePassing):
            if not hasattr(module, "in_channels"):
                module.in_channels = module.nn[0].in_features
                module.out_channels = module.nn[3].out_features

    args.num_layers = 3

    class WrapperModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x, edge_index, batch):
            data = Data(x=x, edge_index=edge_index, batch=batch)
            return self.model(data)[0]

    if args.explainer == "SHAP":
        get_explanations = get_explanations_shap
        wrapp_model = model
    else:
        from eval_explainer import get_explanations as explainer_get_explanations

        get_explanations = lambda m, e, d, f, s: explainer_get_explanations(m, d, args, model_type="ProtGNN")
        wrapp_model = WrapperModel(model)
        wrapp_model.eval()

    if "A" in args.metrics:
        results["A"] = get_acc(model, dataloader_test, forward_fn)
        print_results(results)

    def logits_and_responses_fn(model, data):
        logits, _, _, _, min_distances = model(data)
        prototype_activations = torch.log((min_distances + 1) / (min_distances + model.model.epsilon))
        return logits, prototype_activations

    dataloader = DataLoader(dataset_train + dataset_val + dataset_test, batch_size=args.batch_size, shuffle=False,)

    def get_shap(classifier, inputs):
        torch.set_grad_enabled(True)
        explainer = shap.DeepExplainer(classifier, Variable(inputs))
        shap_values = explainer.shap_values(Variable(inputs))
        shap_values = np.stack(shap_values, axis=0)
        return torch.from_numpy(shap_values), explainer

    responses, logits = get_responses_and_logits(model, dataloader, logits_and_responses_fn)
    shap_values, explainer = get_shap(model.model.last_layer, responses)

    explanations = get_explanations(wrapp_model, explainer, dataloader_test, logits_and_responses_fn, args.seed)

    node_mask_fn = get_fn(args.node_mask_fn)
    edge_mask_fn = get_fn(args.edge_mask_fn)

    truncated_explanations = get_truncated_explanations(explanations, node_mask_fn, edge_mask_fn)
    prototypes = get_prototypes(model, dataset_test, model_args.num_prototypes_per_class)

    if "A1" in args.metrics:
        distances = compare_with_gt_instance(truncated_explanations, iou_distance,)
        if len(distances) == 0:
            results["A1"] = None
        else:
            results["A1"] = np.mean(distances)
        print_results(results)

    if "A2" in args.metrics:
        ground_truths = get_gt_explanations_model(args.dataset if args.dataset != "BA_2Motifs" else "BA-2motif")
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
            preprocess_graph=lambda batch: preprocess_graph(batch, add_gt_expl=False),
        )
        if len(original) < 20:
            print(f"Only {len(original)} samples")
            results["I3"] = None
        else:
            dataloader_new = DataLoader(dataset_noisy_node, batch_size=args.batch_size, shuffle=False)
            new_explanations = get_explanations(
                wrapp_model, explainer, dataloader_new, logits_and_responses_fn, args.seed,
            )
            new_truncated_explanations = get_truncated_explanations(new_explanations, node_mask_fn, edge_mask_fn)
            distances = compare_lists(original, new_truncated_explanations, iou_distance)
            results["I3"] = np.mean(distances)
        print_results(results)

    if "I4" in args.metrics:
        original, dataset_noisy_edge = add_noise_perturb(
            truncated_explanations,
            forward_fn,
            model,
            p_x=0,
            p_edges_add=0.1,
            p_edges_del=0.1,
            preprocess_graph=lambda batch: preprocess_graph(batch, add_gt_expl=False),
        )
        if len(original) < 20:
            print(f"Only {len(original)} samples")
            results["I4"] = None
        else:
            dataloader_new = DataLoader(dataset_noisy_edge, batch_size=args.batch_size, shuffle=False)
            new_explanations = get_explanations(
                wrapp_model, explainer, dataloader_new, logits_and_responses_fn, args.seed,
            )
            new_truncated_explanations = get_truncated_explanations(new_explanations, node_mask_fn, edge_mask_fn)
            distances = compare_lists(original, new_truncated_explanations, iou_distance)
            results["I4"] = np.mean(distances)
        print_results(results)

    if "I5" in args.metrics:
        new_explanations = get_explanations(
            wrapp_model, explainer, dataloader_test, logits_and_responses_fn, args.seed + 1
        )
        new_truncated_explanations = get_truncated_explanations(new_explanations, node_mask_fn, edge_mask_fn)
        distances = compare_lists(truncated_explanations, new_truncated_explanations, iou_distance)
        results["I5"] = np.mean(distances)
        print_results(results)

    if "M1" in args.metrics:
        # try:
        perturbed_model = perturb_model(model, prototypes, dataset, p_x=0.5, p_edge=0).to(device)
        if args.explainer != "SHAP":
            wrapp_perturbed_model = WrapperModel(perturbed_model)
        else:
            wrapp_perturbed_model = perturbed_model

        for data in dataset_train + dataset_val + dataset_test:
            data = data.cpu()
        dataloader = DataLoader(dataset_train + dataset_val + dataset_test, batch_size=args.batch_size, shuffle=False,)
        perturbed_responses, perturbed_logits = get_responses_and_logits(
            perturbed_model, dataloader, logits_and_responses_fn
        )
        _, perturbed_explainer = get_shap(model.model.last_layer, perturbed_responses)
        perturbed_explanations = get_explanations(
            wrapp_perturbed_model, perturbed_explainer, dataloader_test, logits_and_responses_fn, args.seed,
        )
        perturbed_explanations = get_truncated_explanations(perturbed_explanations, node_mask_fn, edge_mask_fn)
        distances = compare_lists(truncated_explanations, perturbed_explanations, iou_distance)
        results["M1"] = 1 - np.mean(distances)
        print_results(results)

    if "M2" in args.metrics:
        try:
            perturbed_model = perturb_model(model, prototypes, dataset, p_x=0, p_edge=0.5)
            if args.explainer != "SHAP":
                wrapp_perturbed_model = WrapperModel(perturbed_model).to(device)
            else:
                wrapp_perturbed_model = perturbed_model

            for data in dataset_train + dataset_val + dataset_test:
                data = data.cpu()
            dataloader = DataLoader(
                dataset_train + dataset_val + dataset_test, batch_size=args.batch_size, shuffle=False,
            )
            perturbed_responses, perturbed_logits = get_responses_and_logits(
                perturbed_model, dataloader, logits_and_responses_fn
            )
            _, perturbed_explainer = get_shap(model.model.last_layer, perturbed_responses)
            perturbed_explanations = get_explanations(
                wrapp_perturbed_model, perturbed_explainer, dataloader_test, logits_and_responses_fn, args.seed,
            )
            perturbed_explanations = get_truncated_explanations(perturbed_explanations, node_mask_fn, edge_mask_fn)
            distances = compare_lists(truncated_explanations, perturbed_explanations, iou_distance)
            results["M2"] = 1 - np.mean(distances)
            print_results(results)
        except Exception as e:
            print(e)
            results["M2"] = None

    if "M3" in args.metrics:
        results["M3"] = 1 - pairwise_list_corr(responses.cpu().detach().numpy(), to_abs=True)
        print_results(results)

    print("Final")
    print_results(results)


if __name__ == "__main__":
    main()
