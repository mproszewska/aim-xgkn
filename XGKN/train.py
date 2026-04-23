import argparse
import copy
import logging
import time
from tqdm import tqdm

from models import GKNetwork
from utils import *

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from torch_geometric.datasets import TUDataset, BA2MotifDataset, BAMultiShapesDataset
from torch_geometric.loader import DataLoader as GDataLoader
from torch_geometric.nn import global_add_pool


SAVE_DIR = "datasets/saved"
SPLITS_DIR = "datasets/splits"
LOG_DIR = "logs"


def args_parser():
    parser = argparse.ArgumentParser(description="graph-kernel-networks")
    parser.add_argument("--save_path", type=str, default=None, help="Model save path")
    parser.add_argument("--load_path", type=str, default=None, help="Model load path")
    parser.add_argument("--dataset", type=str, default="MUTAG", help="dataset")
    parser.add_argument("--k", type=int, default=2, help="Number of hops in k-hop")
    parser.add_argument("--subgraph_size", type=int, default=10, help="subgraph size")
    parser.add_argument("--split", type=int, default=-1, help="split index")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument(
        "--early_stop", type=int, default=40, help="number of evals before early stopping",
    )
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[16], help="hidden layers sizes")
    parser.add_argument(
        "--mlp_hidden_dims", type=int, nargs="+", default=[], help="hidden MLP layers sizes",
    )
    parser.add_argument("--mlp_dropout_rate", type=float, default=0.4, help="dropout rate")
    parser.add_argument("--mlp_activation", default="relu", help="activation function for MLP")
    parser.add_argument(
        "--mlp_start_with_batch_norm", action="store_true", default=True, help="Start MLP with BN",
    )
    parser.add_argument("--filters_sizes", nargs="+", type=int, default=[6], help="nodes in filters")
    parser.add_argument("--kernels", nargs="+", type=str, default=["rws"], help="types of kernels")
    parser.add_argument("--max_step", type=int, default=6, help="max length of random walks")
    parser.add_argument("--encoder_dim", type=int, default=16, help="dim of features encoder")
    parser.add_argument("--rw_dropout_rate", type=float, default=0.4, help="dropout rate")
    parser.add_argument("--eval_freq", type=int, default=10, help="frequency of test evaluation")
    parser.add_argument("--seed", type=int, default=0, help="seed for splitting the dataset")
    parser.add_argument("--rw_norm", action="store_true", default=True, help="normalize x")
    parser.add_argument(
        "--node_degree_label", action="store_true", default=False, help="node degree label",
    )
    args = parser.parse_args()
    return args


def get_logger(log_filename):
    log_filename = os.path.join(LOG_DIR, log_filename)
    print(log_filename)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def train(model, optimizer, data):
    optimizer.zero_grad()
    output, responses, mlp_input, mlp_inputs = model(data)
    loss = F.cross_entropy(output, data.y)
    acc = accuracy(output, data.y)
    loss.backward()
    optimizer.step()
    return loss, acc


def test(model, data):
    output, responses, mlp_input, mlp_inputs = model(data)
    loss = F.cross_entropy(output, data.y)
    acc = accuracy(output, data.y)
    return loss, acc


def main():
    args = args_parser()
    if len(args.filters_sizes) == 1 and len(args.hidden_dims) > 1:
        args.filters_sizes = args.filters_sizes * len(args.hidden_dims)
    if args.mlp_hidden_dims == [0]:
        args.mlp_hidden_dims = []

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    root = f"../datasets/saved/ss{args.subgraph_size}_k{args.k}"
    if args.node_degree_label:
        root = f"{root}_ndl"
    if args.dataset == "BA-2motif":
        dataset = BA2MotifDataset(
            f"{root}/BA-2motif",
            pre_transform=lambda data: transform(data, args.k, args.subgraph_size, args.node_degree_label),
        )
    elif args.dataset == "BAMultiShapes":
        dataset = BAMultiShapesDataset(
            f"{root}/BAMultiShapes",
            pre_transform=lambda data: transform(data, args.k, args.subgraph_size, args.node_degree_label),
        )
    else:
        dataset = TUDataset(
            root,
            args.dataset,
            use_node_attr=True,
            pre_transform=lambda data: transform(data, args.k, args.subgraph_size, args.node_degree_label,),
        )

    num_classes = dataset.num_classes
    x_dim = dataset.x.shape[-1]

    kernel_kwargs = {
        "kernels": args.kernels,
        "max_step": args.max_step,
        "encoder_dim": args.encoder_dim,
        "dropout_rate": args.rw_dropout_rate,
        "rw_norm": args.rw_norm,
    }

    model_args = {
        "in_features": x_dim,
        "out_features": num_classes,
        "hidden_dims": args.hidden_dims,
        "kernel_kwargs": kernel_kwargs,
        "filters_sizes": args.filters_sizes,
        "mlp_hidden_dims": args.mlp_hidden_dims,
        "mlp_activation": args.mlp_activation,
        "mlp_dropout_rate": args.mlp_dropout_rate,
        "mlp_start_with_batch_norm": args.mlp_start_with_batch_norm,
    }
    model = GKNetwork(**model_args).to(device)

    train_idxs, val_idxs, test_idxs = split_data(
        dataset, args.dataset, seed=args.seed, split=args.split, splits_dir=SPLITS_DIR,
    )
    dataset_train = torch.utils.data.Subset(dataset, train_idxs)
    dataset_val = torch.utils.data.Subset(dataset, val_idxs)
    dataset_test = torch.utils.data.Subset(dataset, test_idxs)

    dataloader_train = GDataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    dataloader_val = GDataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)
    dataloader_test = GDataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    if args.load_path is not None:
        params = torch.load(args.load_path, map_location=device)
        model = GKNetwork(**params["model_args"])
        model.load_state_dict(params["state_dict"])
        model.eval()
        model = model.to(device)

    optimizer, scheduler = get_optimizer(model, args)
    early_stopper = EarlyStopper(patience=args.early_stop, min_delta=0.00)

    time_str = time.strftime("%Y%m%d_%H%M%S")
    logger = get_logger(f"{time_str}.log")
    logger.info(args)
    logger.info(model)

    pbar = tqdm(range(1, args.epochs + 1), total=args.epochs, bar_format="{l_bar}{bar:500}{r_bar}{bar:-10b}",)
    loss_ce, acc = (Metric(), Metric())

    def add_metrics(batch_loss, batch_acc, split):
        loss_ce.add(batch_loss, split)
        acc.add(batch_acc, split)

    bast_state_dict = copy.deepcopy(model.state_dict())
    torch.autograd.set_detect_anomaly(True)
    warm_up_epochs = -1
    for epoch in pbar:
        description_str = f"epoch={epoch},"

        model.train()
        for batch in dataloader_train:
            batch = batch.to(device)
            batch_loss, batch_acc = train(model, optimizer, batch)
            if epoch >= warm_up_epochs:
                add_metrics(batch_loss, batch_acc, "train")
        description_str += f"train_loss={loss_ce.get('train'):.4f},train_acc={acc.get('train'):.4f},"

        if epoch % args.eval_freq == 0:
            model.eval()
            with torch.no_grad():
                for batch in dataloader_val:
                    batch = batch.to(device)
                    batch_loss, batch_acc = test(model, batch)
                    if epoch >= warm_up_epochs:
                        add_metrics(batch_loss, batch_acc, "val")
                description_str += f"val_loss={loss_ce.get('val'):.4f},val_acc={acc.get('val'):.4f},"

                for batch in dataloader_test:
                    batch = batch.to(device)
                    batch_loss, batch_acc = test(model, batch)
                    if epoch >= warm_up_epochs:
                        add_metrics(batch_loss, batch_acc, "test")
                description_str += f"test_loss={loss_ce.get('test'):.4f},test_acc={acc.get('test'):.4f}"
                pbar.set_description_str(description_str, refresh=True)
                if early_stopper.early_stop(loss_ce.get("val")):
                    break

        loss_ce.restart()
        if acc.save_higher():
            if epoch >= warm_up_epochs:
                bast_state_dict = copy.deepcopy(model.state_dict())
        if scheduler is not None:
            scheduler.step()

        logger.info(description_str)
    final_str = f"Final:train_acc={acc.get_best('train'):.4f},val_acc={acc.get_best('val'):.4f},test_acc={acc.get_best('test'):.4f}"
    logger.info(final_str)
    print(final_str)
    model.load_state_dict(bast_state_dict)

    if args.save_path is not None:
        save_model(args.save_path, bast_state_dict, model_args, vars(args))


if __name__ == "__main__":
    main()
