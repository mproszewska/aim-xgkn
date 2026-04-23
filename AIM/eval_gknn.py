import argparse
from tqdm import tqdm

import sys

sys.path.insert(1, f"../../GKNN_submitted/src")
from dataset import TUDataset as GKNNDataset
from model import Model

import numpy as np
from torch import nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse

from distances import iou_distance, weighted_ged_distance
from eval_explainer import get_explanations as explainer_get_explanations
from utils import *

GKNN_PATH = f"../..//GKNN_submitted"
METRICS_NAMES = list(METRICS.keys())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--split", type=int, default=-1, help="Split index")
    parser.add_argument("--seed", type=int, default=0, help="seed")
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


def get_responses_and_logits(model, dataloader, forward_fn):
    logits, responses = list(), list()
    for data in dataloader:
        data = data.to(device)
        l, r = forward_fn(model, data)
        responses.append(r), logits.append(l)
    return torch.cat(responses, 0), torch.cat(logits)


def get_prototypes(model):
    prototypes = list()
    for conv_layer in model.conv_layers:
        x = conv_layer.X
        adj = conv_layer.P
        for i in range(adj.shape[0]):
            edge_index = dense_to_sparse(adj[i])[0]
            prototypes.append(Data(edge_index=edge_index, x=x[i]))
    return prototypes


def perturb_model(model, dataset, p_x=0.2, p_edge=0.2):
    model_copy = copy.deepcopy(model)
    model_copy.eval()
    all_x = torch.cat([data.x for data in dataset])
    for ker_layer in model_copy.conv_layers:
        x = ker_layer.X.detach()
        adj = ker_layer.P.detach()
        if p_x > 0:
            x_shape = x.shape
            x = x.reshape(-1, x.shape[-1])
            keys = torch.from_numpy(np.random.choice([0, 1], p=[1 - p_x, p_x], size=len(x))).bool()
            size = keys.sum().item()
            if size > 0:
                values = all_x[np.random.randint(low=0, high=len(all_x), size=size)].to(x.device)
                if values.shape[-1] != x_shape[-1]:
                    values = x[np.random.randint(low=0, high=len(x), size=size)].to(x.device)
                x[keys] = values
            ker_layer.X.data = x.reshape(x_shape)
        if p_edge > 0:
            adj_shape = adj.shape
            adj = adj.reshape(-1)
            keys = (
                torch.from_numpy(np.random.choice([0, 1], p=[1 - p_edge, p_edge], size=len(adj))).bool().to(adj.device)
            )
            adj[keys] = 1 - adj[keys]
            ker_layer.P.data = adj.reshape(adj_shape)
    return model_copy


def get_explanations_shap(model, explainer, dataloader, forward_fn, seed):
    set_seed(seed)
    explanations = list()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            logits, all_responses, response = forward_fn(model, data)
        pred = logits.argmax(dim=-1)
        importance = run_shap(explainer, response).to(device)
        importance = torch.stack([importance[pred[i]][i] for i in range(len(pred))], dim=0).to(device)
        response[response == 0] = 1e-36

        num_nodes = 0
        gt_expl_node_mask = data.gt_expl_node_mask if hasattr(data, "gt_expl_node_mask") else None
        for b in tqdm(range(len(data.y)), total=len(data.y)):
            mask = data.batch == b
            im, r, ar = importance[b], response[b], all_responses[mask]
            im = (im / r) * ar

            nidx = data.nidx[mask]
            x = data.x[mask]
            edge_index = data.edge_index[:, data.batch[data.edge_index[0]] == b] - num_nodes
            num_nodes += len(x)
            node_mask = torch.zeros(len(x), im.shape[-1], device=device)
            for i in range(len(nidx)):
                nidx_i = nidx[i : i + 1]
                # norm = len(nidx_i)
                # if norm == 0:
                #    continue
                node_mask[nidx_i] += im[i]  # / norm
            e = Data(
                x=x.cpu(),
                edge_index=edge_index.cpu(),
                y=data.y[b : b + 1].cpu(),
                y_pred=logits[b : b + 1].cpu(),
                node_mask=node_mask.cpu().sum(-1),
                edge_mask=None,
                gt_expl_node_mask=gt_expl_node_mask[mask] if gt_expl_node_mask is not None else None,
            )
            explanations.append(e)
    return explanations


class WrapperModel(nn.Module):
    def __init__(self, model, forward_fn):
        super().__init__()
        self.model = model
        self.forward_fn = forward_fn

    def forward(self, x, edge_index, batch):
        data = Data(x=x, edge_index=edge_index, batch=batch)
        out = self.forward_fn(self.model, data)
        return out


def main():
    args = args_parser()
    print(args)
    set_seed(args.seed)

    model = Model.load_from_checkpoint(checkpoint_path=args.model_path, map_location=None).to(device)
    model.eval()
    args.num_layers = 1  # len(model.conv_layers)

    dataset = GKNNDataset(f"{GKNN_PATH}/data/", model.hparams.dataset)

    gt_fn = get_gt_fn(model.hparams.dataset)

    def preprocess_graph(data, add_gt_expl=True):
        if add_gt_expl and (gt_fn is not None):
            data = gt_fn(data)
        data.nidx = torch.arange(len(data.x))
        return data

    def preprocess_dataset(dataset):
        return list(map(preprocess_graph, dataset))

    train_idxs, val_idxs, test_idxs = get_splits(args.dataset, size=len(dataset), seed=args.seed, split=args.split)
    dataset_train, dataset_val, dataset_test = (
        Subset(dataset, train_idxs),
        Subset(dataset, val_idxs),
        Subset(dataset, test_idxs),
    )

    dataset_test = preprocess_dataset(dataset_test)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    forward_fn = lambda model, x: model(x)[0]

    results = {}

    if "A" in args.metrics:
        results["A"] = get_acc(model, dataloader_test, forward_fn)
        print_results(results)

    def logits_and_responses_fn(model, data):
        logits, _, _, responses = model(data)
        return logits, responses

    def logits_and_all_responses_fn(model, data):
        logits, all_responses, _, responses = model(data)
        return logits, all_responses[0], responses

    dataloader = DataLoader(preprocess_dataset(dataset), batch_size=args.batch_size, shuffle=False)

    responses, logits = get_responses_and_logits(model, dataloader, logits_and_responses_fn)

    shap_values, explainer = get_shap(model.fc, responses)

    if args.explainer == "SHAP":
        get_explanations = get_explanations_shap
        wrap_model = model
    else:
        wrap_model = WrapperModel(model, forward_fn)
        wrap_model.eval()
        get_explanations = lambda m, e, d, f, s: explainer_get_explanations(m, d, args, model_type="GKNN")

    explanations = get_explanations(wrap_model, explainer, dataloader_test, logits_and_all_responses_fn, args.seed)

    node_mask_fn = get_fn(args.node_mask_fn)
    edge_mask_fn = get_fn(args.edge_mask_fn)

    truncated_explanations = get_truncated_explanations(explanations, node_mask_fn, edge_mask_fn)

    if bool({"M1", "M2", "M3", "A3"} & set(args.metrics)):
        prototypes = get_prototypes(model)
    else:
        prototypes = None

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
        with_explanations = with_without_explanation_perturb(truncated_explanations, keep_expl=True, num_samples=100)
        without_explanations = with_without_explanation_perturb(
            truncated_explanations, keep_expl=False, num_samples=100
        )
        with_explanations = [data for data in with_explanations if data is not None]
        without_explanations = [data for data in without_explanations if data is not None]
        if (len(with_explanations) < 20) or (len(without_explanations) < 20):
            results["I1"] = None
            results["I2"] = None
        else:
            if "I1" in args.metrics:
                dataloader_with = DataLoader(with_explanations, batch_size=args.batch_size, shuffle=False)
                f = (
                    forward_fn
                    if args.explainer == "SHAP"
                    else lambda model, data: model(data.x, data.edge_index, data.batch)
                )
                results["I1"] = get_acc(wrap_model, dataloader_with, f)
                print_results(results)
            if "I2" in args.metrics:
                dataloader_without = DataLoader(without_explanations, batch_size=args.batch_size, shuffle=False)
                f = (
                    forward_fn
                    if args.explainer == "SHAP"
                    else lambda model, data: model(data.x, data.edge_index, data.batch)
                )
                results["I2"] = 1 - get_acc(wrap_model, dataloader_without, f)
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
            dataset_noisy_node = preprocess_dataset(dataset_noisy_node)

            dataloader_new = DataLoader(dataset_noisy_node, batch_size=args.batch_size, shuffle=False)
            new_explanations = get_explanations(
                wrap_model, explainer, dataloader_new, logits_and_all_responses_fn, args.seed
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
            p_x=0.0,
            p_edges_add=0.1,
            p_edges_del=0.1,
            preprocess_graph=lambda x: preprocess_graph(x, add_gt_expl=False),
        )
        if len(original) < 20:
            print(f"Only {len(original)} samples")
            results["I4"] = None
        else:
            dataset_noisy_edge = preprocess_dataset(dataset_noisy_edge)
            dataloader_new = DataLoader(dataset_noisy_edge, batch_size=args.batch_size, shuffle=False)
            new_explanations = get_explanations(
                wrap_model, explainer, dataloader_new, logits_and_all_responses_fn, args.seed
            )
            new_truncated_explanations = get_truncated_explanations(new_explanations, node_mask_fn, edge_mask_fn)
            distances = compare_lists(original, new_truncated_explanations, iou_distance)
            results["I4"] = np.mean(distances)
        print_results(results)

    if "I5" in args.metrics:
        new_explanations = get_explanations(
            wrap_model, explainer, dataloader_test, logits_and_all_responses_fn, args.seed
        )
        new_truncated_explanations = get_truncated_explanations(new_explanations, node_mask_fn, edge_mask_fn)
        distances = compare_lists(truncated_explanations, new_truncated_explanations, iou_distance)
        results["I5"] = np.mean(distances)
        print_results(results)

    if "M1" in args.metrics:
        perturbed_model = perturb_model(model, dataset, p_x=0.5, p_edge=0)
        perturbed_responses, perturbed_logits = get_responses_and_logits(
            perturbed_model, dataloader, logits_and_responses_fn
        )

        _, perturbed_explainer = get_shap(perturbed_model.fc, perturbed_responses)

        new_wrap_model = perturbed_model if args.explainer == "SHAP" else WrapperModel(perturbed_model, forward_fn)
        new_wrap_model.eval()
        perturbed_explanations = get_explanations(
            new_wrap_model, perturbed_explainer, dataloader_test, logits_and_all_responses_fn, args.seed,
        )
        perturbed_explanations = get_truncated_explanations(perturbed_explanations, node_mask_fn, edge_mask_fn)
        distances = compare_lists(truncated_explanations, perturbed_explanations, iou_distance)
        results["M1"] = 1 - np.mean(distances)
        print_results(results)

    if "M2" in args.metrics:
        perturbed_model = perturb_model(model, dataset, p_x=0, p_edge=0.5)
        new_wrap_model = perturbed_model if args.explainer == "SHAP" else WrapperModel(perturbed_model, forward_fn)
        new_wrap_model.eval()
        perturbed_responses, perturbed_logits = get_responses_and_logits(
            perturbed_model, dataloader, logits_and_responses_fn
        )
        _, perturbed_explainer = get_shap(perturbed_model.fc, perturbed_responses)
        perturbed_explanations = get_explanations(
            new_wrap_model, perturbed_explainer, dataloader_test, logits_and_all_responses_fn, args.seed,
        )
        perturbed_explanations = get_truncated_explanations(perturbed_explanations, node_mask_fn, edge_mask_fn)
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
