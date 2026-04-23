import argparse
import copy

import numpy as np
from torch.utils.data import Subset
from torch_geometric.explain import (
    Explainer,
    GNNExplainer,
    PGExplainer,
    CaptumExplainer,
    AttentionExplainer,
    GraphMaskExplainer,
)
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import MessagePassing

from distances import iou_distance
from utils import *
from train_gnns import get_model

METRICS_NAMES = list(METRICS.keys())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--split", type=int, default=-1, help="Split index")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument(
        "--explainer", type=str, required=True, choices=EXPLAINERS, help="Type of explainer",
    )
    parser.add_argument(
        "--metrics", type=str, nargs="+", choices=METRICS_NAMES, default=METRICS_NAMES, help="Metrics to evaluate",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--node_mask_fn", type=str, default="none", help="Thresholding of nodes")
    parser.add_argument("--edge_mask_fn", type=str, default="none", help="Thresholding of edges")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()
    return args


def get_explanations(model, dataloader, args, model_type=None):
    explanations = list()
    if args.explainer == "PGExplainer":
        explainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=args.epochs, lr=args.lr).to(device),  # lr 0.003
            explanation_type="phenomenon",
            edge_mask_type="object",
            model_config=dict(mode="multiclass_classification", task_level="graph", return_type="raw",),
        )
        for epoch in range(args.epochs):
            for data in dataloader:
                data = data.to(device)
                pred = model(data.x, data.edge_index, data.batch)
                explainer.algorithm.train(
                    epoch, model, data.x, data.edge_index, batch=data.batch, target=pred.argmax(dim=-1),
                )
    elif args.explainer == "GNNExplainer":
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=args.epochs, lr=args.lr),
            explanation_type="model",
            node_mask_type="object",
            edge_mask_type=None,
            model_config=dict(mode="multiclass_classification", task_level="graph", return_type="raw",),
        )
    elif args.explainer == "IntegratedGradients":
        explainer = Explainer(
            model=model,  # It is assumed that model outputs a single tensor.
            algorithm=CaptumExplainer("IntegratedGradients"),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type=None,  # "object",
            model_config=dict(
                mode="multiclass_classification", task_level="graph", return_type="raw",  # Model returns probabilities.
            ),
        )
    elif args.explainer == "ShapleyValueSampling":
        explainer = Explainer(
            model=model,  # It is assumed that model outputs a single tensor.
            algorithm=CaptumExplainer(
                "ShapleyValueSampling", show_progress=True, n_samples=25 if args.dataset == "MUTAG" else 10
            ),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type=None,  # "object",
            model_config=dict(
                mode="multiclass_classification", task_level="graph", return_type="raw",  # Model returns probabilities.
            ),
        )
    elif args.explainer == "AttentionExplainer":
        explainer = Explainer(
            model=model,
            algorithm=AttentionExplainer(reduce="max"),
            explanation_type="model",
            # node_mask_type='object',
            edge_mask_type="object",
            model_config=dict(mode="multiclass_classification", task_level="graph", return_type="raw",),
        )

    elif args.explainer == "GraphMaskExplainer":
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        explainer = Explainer(
            model=model_copy,
            algorithm=GraphMaskExplainer(num_layers=args.num_layers, epochs=args.epochs, lr=args.lr),
            explanation_type="model",
            node_mask_type="object",
            edge_mask_type=None,
            model_config=dict(mode="multiclass_classification", task_level="graph", return_type="raw",),
        )
    else:
        assert False, args.explainer

    for data in tqdm(dataloader, total=len(dataloader)):
        data = data.to(device)
        target = data.y if args.explainer in ["PGExplainer"] else None
        if model_type == "KerGNN":
            explanation = explainer(data.x, data.edge_index, batch=data.batch, target=target)
            explanation.x = data.x
            explanation.adj = data.adj
            explanation.nidx = data.nidx
            explanation.nidx[explanation.nidx == len(explanation.x)] = -1
            if not hasattr(explanation, "prediction"):
                explanation.prediction = model(x, data.edge_index, batch=data.batch)
        else:
            explanation = explainer(data.x, data.edge_index, batch=data.batch, target=target)
            if not hasattr(explanation, "prediction"):
                explanation.prediction = model(data.x, data.edge_index, data.batch)

        explanation.y = explanation.target
        num_nodes = 0
        gt_expl_node_mask = data.gt_expl_node_mask if hasattr(data, "gt_expl_node_mask") else None
        for b in range(explanation.batch.max() + 1):
            mask = explanation.batch == b
            data = Data(
                x=explanation.x[mask],
                edge_index=explanation.edge_index[:, explanation.batch[explanation.edge_index[0]] == b] - num_nodes,
                y=explanation.y[b],
                y_pred=explanation.prediction[b],
                node_mask=explanation.node_mask[mask].sum(-1) if hasattr(explanation, "node_mask") else None,
                edge_mask=(
                    explanation.edge_mask[explanation.batch[explanation.edge_index[0]] == b]
                    if hasattr(explanation, "edge_mask")
                    else None
                ),
                mask=explanation.mask[mask] if hasattr(explanation, "mask") else None,
                gt_expl_node_mask=gt_expl_node_mask[mask] if gt_expl_node_mask is not None else None,
                nidx=explanation.nidx[mask] if hasattr(explanation, "nidx") else None,
                adj=explanation.adj[mask] if hasattr(explanation, "adj") else None,
            )
            explanations.append(data)
            num_nodes += mask.sum()
    return explanations


def main():
    args = args_parser()
    print(args)
    set_seed(args.seed)
    params = torch.load(args.model_path, map_location=torch.device("cpu"))
    model = get_model(**params["args"]).to(device)
    for module in model.modules():
        if isinstance(module, MessagePassing):
            if not hasattr(module, "in_channels"):
                channel_list = module.nn.channel_list
                module.in_channels = channel_list[0]
                module.out_channels = channel_list[-1]

    # model.in_channels = params["args"]["num_node_features"]
    args.num_layers = params["args"]["num_layers"]
    model.load_state_dict(params["state_dict"])
    model.eval()

    degree_attr = params["args"]["num_node_features"] == 1
    dataset = get_dataset(args.dataset, degree_attr=degree_attr)
    train_idxs, val_idxs, test_idxs = get_splits(args.dataset, size=len(dataset), seed=args.seed, split=args.split)
    dataset_train, dataset_val, dataset_test = (
        Subset(dataset, train_idxs),
        Subset(dataset, val_idxs),
        Subset(dataset, test_idxs),
    )

    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    forward_fn = lambda model, data: model(data.x, data.edge_index, batch=data.batch)
    results = {}

    explanations = get_explanations(model, dataloader_test, args)

    node_mask_fn = get_fn(args.node_mask_fn)
    edge_mask_fn = get_fn(args.edge_mask_fn)

    if "A" in args.metrics:
        results["A"] = get_acc(model, dataloader_test, forward_fn)
        print_results(results)

    explanations = get_explanations(model, dataloader_test, args)
    truncated_explanations = get_truncated_explanations(explanations, node_mask_fn, edge_mask_fn)

    if "A1" in args.metrics:
        distances = compare_with_gt_instance(truncated_explanations, iou_distance)
        if len(distances) == 0:
            results["A1"] = None
        else:
            results["A1"] = np.mean(distances)
        print_results(results)

    if "A2" in args.metrics:
        results["A2"] = None
        print_results(results)

    if bool({"I1", "I2"} & set(args.metrics)):
        with_explanations = with_without_explanation_perturb(truncated_explanations, keep_expl=True, num_samples=10)
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
            truncated_explanations, forward_fn, model, p_x=0.1, p_edges_add=0.0, p_edges_del=0.0,
        )
        if len(original) < 20:
            print(f"Only {len(original)} samples")
            results["I3"] = None
        else:
            dataloader_new = DataLoader(dataset_noisy_node, batch_size=args.batch_size, shuffle=False)
            new_explanations = get_explanations(model, dataloader_new, args)
            new_truncated_explanations = get_truncated_explanations(new_explanations, node_mask_fn, edge_mask_fn)
            distances = compare_lists(original, new_truncated_explanations, iou_distance)
            results["I3"] = np.mean(distances)
        print_results(results)

    if "I4" in args.metrics:
        original, dataset_noisy_edge = add_noise_perturb(
            truncated_explanations, forward_fn, model, p_x=0.0, p_edges_add=0.1, p_edges_del=0.1,
        )
        if len(original) < 20:
            print(f"Only {len(original)} samples")
            results["I4"] = None
        else:
            dataloader_new = DataLoader(dataset_noisy_edge, batch_size=args.batch_size, shuffle=False)
            new_explanations = get_explanations(model, dataloader_new, args)
            new_truncated_explanations = get_truncated_explanations(new_explanations, node_mask_fn, edge_mask_fn)
            distances = compare_lists(original, new_truncated_explanations, iou_distance)
            results["I4"] = np.mean(distances)
        print_results(results)

    if "I5" in args.metrics:
        new_explanations = get_explanations(model, dataloader_test, args)
        new_truncated_explanations = get_truncated_explanations(new_explanations, node_mask_fn, edge_mask_fn)
        distances = compare_lists(truncated_explanations, new_truncated_explanations, iou_distance)
        results["I5"] = np.mean(distances)
        print_results(results)

    if "M1" in args.metrics:
        results["M1"] = None
    if "M2" in args.metrics:
        results["M2"] = None
    if "M3" in args.metrics:
        results["M3"] = None

    print("Final")
    print_results(results)


if __name__ == "__main__":
    main()
