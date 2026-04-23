import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_max_pool
from torch_geometric.nn import Sequential, GAT, GCN, GIN, GATConv

from utils import *


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name")
    parser.add_argument("--degree_attr", action="store_true", help="Degree as attribute")
    parser.add_argument("--split", type=int, default=-1, help="Split index")
    parser.add_argument("--seed", type=int, default=123, help="Seed")
    parser.add_argument(
        "--model_type", type=str, default=None, choices=["GCN", "GAT", "GIN"], help="Type of model",
    )
    parser.add_argument("--path", type=str, default=None, help="Save path")
    parser.add_argument("--hd", type=int, default=32, help="Hidden dim")
    parser.add_argument("--nl", type=int, default=3, help="Number of layers")
    parser.add_argument("--ld", type=int, default=32, help="Linear dim")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--bs", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.0001, help="Weight decay")
    args = parser.parse_args()
    return args


def get_model(
    model_type, num_node_features, num_classes, hidden_dim=32, num_layers=3, linear_dim=32,
):
    if model_type == "GCN":
        model = Sequential(
            "x, edge_index, batch",
            [
                (
                    GCN(
                        num_node_features,
                        hidden_dim,
                        num_layers=num_layers,
                        out_channels=linear_dim,
                        dropout=0.4,
                        norm="batch_norm",
                    ),
                    "x, edge_index -> x",
                ),
                (global_add_pool, "x, batch -> x"),
                Linear(linear_dim, num_classes),
            ],
        )

    elif model_type == "GAT":
        model = Sequential(
            "x, edge_index, batch",
            [
                (
                    GAT(
                        num_node_features,
                        hidden_dim,
                        num_layers=num_layers,
                        heads=2,
                        v2=True,
                        out_channels=linear_dim,
                        act="elu",
                        dropout=0.6,
                        norm="batch_norm",
                        # jk='max',
                    ),
                    "x, edge_index -> x",
                ),
                (global_add_pool, "x, batch -> x"),
                Linear(linear_dim, num_classes),
            ],
        )
    elif model_type == "GIN":
        model = Sequential(
            "x, edge_index, batch",
            [
                (
                    GIN(
                        num_node_features,
                        hidden_dim,
                        num_layers=num_layers,
                        out_channels=linear_dim,
                        norm="batch_norm",
                        dropout=0.4,
                    ),
                    "x, edge_index -> x",
                ),
                (global_add_pool, "x, batch -> x"),
                Linear(linear_dim, num_classes),
            ],
        )
    else:
        assert False
    return model


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, batch=data.batch)
    loss = F.cross_entropy(out, data.y)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index, batch=data.batch)
    pred = out.argmax(dim=-1)
    loss = F.cross_entropy(out, data.y)
    acc = (pred == data.y).float().mean()
    return loss, acc


def train_model(
    model, dataloader_train, dataloader_val, dataloader_test, epochs=200, lr=0.001, weight_decay=0.005,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    pbar = tqdm(range(1, epochs + 1))
    max_acc_train, max_acc_val, max_acc_test = 0, 0, 0
    bast_state_dict = copy.deepcopy(model.state_dict())

    for epoch in pbar:
        model.train()
        loss_train, acc_train, loss_val, acc_val, loss_test, acc_test = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for data in dataloader_train:
            train(model, optimizer, data.to(device))

        if epoch % 10 == 0:
            for data in dataloader_train:
                loss, acc = test(model, data.to(device))
                loss_train.append(loss.item()), acc_train.append(acc.item())
            for data in dataloader_val:
                loss, acc = test(model, data.to(device))
                loss_val.append(loss.item()), acc_val.append(acc.item())
            for data in dataloader_test:
                loss, acc = test(model, data.to(device))
                loss_test.append(loss.item()), acc_test.append(acc.item())
            stats = [loss_train, acc_train, loss_val, acc_val, loss_test, acc_test]
            loss_train, acc_train, loss_val, acc_val, loss_test, acc_test = [np.mean(s) for s in stats]
            if max_acc_val < acc_val:
                max_acc_train, max_acc_val, max_acc_test = acc_train, acc_val, acc_test
                bast_state_dict = copy.deepcopy(model.state_dict())

            pbar.set_description(
                f"Train l:{loss_train:.4f} a:{acc_train:.4f} | Val l:{loss_val:.4f} a:{acc_val:.4f} | Test l:{loss_test:.4f} a:{acc_test:.4f}"
            )
    print(f"Train a:{max_acc_train:.4f} | Val a:{max_acc_val:.4f} | Test a:{max_acc_test:.4f}")
    pbar.close()
    model.load_state_dict(bast_state_dict)
    model.eval()

    return model


def main():
    args = args_parser()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(args.dataset, args.degree_attr)
    train_idxs, val_idxs, test_idxs = get_splits(args.dataset, size=len(dataset), seed=args.seed, split=args.split)

    dataset_train, dataset_val, dataset_test = (
        Subset(dataset, train_idxs),
        Subset(dataset, val_idxs),
        Subset(dataset, test_idxs),
    )

    dataloader_train = DataLoader(dataset_train, batch_size=args.bs, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=args.bs, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=args.bs, shuffle=False)

    model_args = {
        "model_type": args.model_type,
        "num_node_features": dataset_test[0].num_node_features,
        "num_classes": dataset.num_classes,
        "hidden_dim": args.hd,
        "num_layers": args.nl,
        "linear_dim": args.ld,
    }

    model = get_model(**model_args).to(device)
    model = train_model(
        model, dataloader_train, dataloader_val, dataloader_test, epochs=args.epochs, lr=args.lr, weight_decay=args.wd,
    )
    torch.save({"state_dict": model.state_dict(), "args": model_args, "train_args": vars(args)}, args.path)


if __name__ == "__main__":
    main()
