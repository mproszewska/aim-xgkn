import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_dims,
        dropout_rate=0.5,
        activation="relu",
        bias=True,
        mlp_start_with_batch_norm=False,
    ):
        super(MLP, self).__init__()

        self.dropout_rate = dropout_rate
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dims = hidden_dims
        self.activation = activation
        dims = [in_features] + hidden_dims
        self.layers = list()
        if mlp_start_with_batch_norm:
            self.layers += [nn.BatchNorm1d(dims[0])]
        for i in range(len(dims) - 1):
            self.layers += [
                nn.Linear(dims[i], dims[i + 1], bias=bias),
                {"relu": nn.ReLU(), "sigmoid": nn.Sigmoid()}[activation],
            ]
            self.layers += [nn.Dropout(p=dropout_rate)]
        self.layers += [nn.Linear(dims[-1], out_features, bias=bias)]
        self.layers = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


def rw_kernel(x_a, adj_a, x_b, adj_b, max_step, dropout, norm_rw=False, from_one_node=False):
    device = x_a.device
    filters_size = x_b.shape[0]
    if norm_rw:
        x_a = F.normalize(x_a, dim=-1)
        x_b = F.normalize(x_b, dim=1)

    xx = torch.einsum("mcn,abc->ambn", (x_b, x_a))  # (#G, #Nodes_filter, #Nodes_sub, D_out)
    out = []
    for i in range(max_step):
        if i == 0:
            eye = torch.eye(filters_size, device=device)
            o = torch.einsum("ab,bcd->acd", (eye, x_b))
            t = torch.einsum("mcn,abc->ambn", (o, x_a))
        else:
            x_a = torch.einsum("abc,acd->abd", (adj_a, x_a))
            x_b = torch.einsum("abd,bcd->acd", (adj_b, x_b))  # adj_hidden_norm: (Nhid,Nhid,Dout)
            if from_one_node:
                t = torch.einsum("mcn,abc->ambn", (x_b, x_a[:, 0:1, :]))
            else:
                t = torch.einsum("mcn,abc->ambn", (x_b, x_a))
        t = dropout(t)
        t = torch.mul(xx, t)  # (#G, #Nodes_filter, #Nodes_sub, D_out)
        t = rw_agg_fn(t)
        out += [t]

    return sum(out) / len(out)


def rw_agg_fn(t):
    return torch.mean(t, dim=[1, 2])


class DiffGKLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        filters_size,
        dropout_rate,
        max_step=1,
        encoder_dim=16,
        rw_norm=False,
        kernels=None,
        sd_k=5,
        sd_take_last=True,
        batch_norm=False,
    ):
        super(DiffGKLayer, self).__init__()
        self.kernels = kernels
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(p=dropout_rate)
        self.filters_size = filters_size
        self.max_step = max_step
        self.rw_norm = rw_norm
        self.sd_k = sd_k
        self.sd_take_last = sd_take_last
        self.kernel_codes = kernels

        class Encoder(nn.Module):
            def __init__(self, in_features, encoder_dim):
                super(Encoder, self).__init__()
                self.layers = nn.ModuleList([nn.Linear(in_features, encoder_dim, bias=True), nn.ReLU()])

            def forward(self, x):
                for l in self.layers:
                    x = l(x)
                return x

        self.encoder = Encoder(in_features, encoder_dim)

        self._x_hidden = Parameter(torch.FloatTensor(filters_size, encoder_dim, out_features))
        self._adj_hidden = Parameter(torch.FloatTensor((filters_size * (filters_size - 1)) // 2, out_features))

        self.init_weights()

        self.kernels_fn = list()
        for kernel in self.kernels:
            if kernel in ["rw", "rws"]:
                kernel_fn = lambda x, adj, x_hidden, adj_hidden: rw_kernel(
                    x, adj, x_hidden, adj_hidden, self.max_step, self.dropout, self.rw_norm, kernel == "rws",
                )
            else:
                raise ValueError(f"Invalid kernel {kernel}")
            self.kernels_fn += [kernel_fn]

    def init_weights(self):
        self._adj_hidden.data.uniform_(-1, 1)
        self._x_hidden.data.uniform_(0, 1)

    def adj_hidden(self, permuted=False):
        device = self._adj_hidden.device
        adj_hidden_norm = torch.zeros(self.filters_size, self.filters_size, self.out_features).to(device)
        indices = torch.triu_indices(self.filters_size, self.filters_size, 1).to(device)
        adj_hidden_norm[indices[0], indices[1], :] = F.relu(self._adj_hidden)
        adj_hidden_norm = adj_hidden_norm + torch.transpose(adj_hidden_norm, 0, 1)
        return adj_hidden_norm if permuted else torch.permute(adj_hidden_norm, (2, 0, 1))

    def x_hidden(self, permuted=False):
        x = F.relu(self._x_hidden)
        return x if permuted else torch.permute(x, (2, 0, 1))

    def forward(self, x, adj, nidx):
        device = adj.device
        x_hidden = self.x_hidden(permuted=True)
        x = self.encoder(x.to_dense())  # (#G, D_hid)
        adj_hidden = self.adj_hidden(permuted=True).to(device)
        adj = adj.to_dense()
        x = x[nidx]  # (#G, #Nodes_sub, D_hid)
        kernel_responses = list()

        for kernel_fn in self.kernels_fn:
            kernel_responses += [kernel_fn(x, adj, x_hidden, adj_hidden)]
        kernel_responses = torch.stack(kernel_responses, dim=-1)
        responses = torch.sum(kernel_responses, dim=-1)
        return responses


def _apply(func, x):
    out = [func(i) for i in torch.unbind(x, dim=0)]
    return torch.stack(out, dim=0)


class GKNetwork(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_dims=None,
        ker_activation=None,
        kernel_kwargs=None,
        filters_sizes=None,
        mlp_hidden_dims=None,
        mlp_activation="relu",
        mlp_dropout_rate=0.4,
        mlp_start_with_batch_norm=False,
    ):
        super(GKNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ker_activation = ker_activation

        self.ker_layers = nn.ModuleList()

        self.ker_layers += [
            DiffGKLayer(
                in_features=in_features, out_features=hidden_dims[0], filters_size=filters_sizes[0], **kernel_kwargs,
            )
        ]

        mlp_dim = sum(hidden_dims)

        dims = [in_features] + hidden_dims
        for i in range(2, len(dims)):
            self.ker_layers += [
                DiffGKLayer(
                    in_features=dims[i - 1], out_features=dims[i], filters_size=filters_sizes[i - 1], **kernel_kwargs,
                )
            ]

        self.mlp = MLP(
            mlp_dim,
            out_features,
            mlp_hidden_dims,
            mlp_dropout_rate,
            mlp_activation,
            mlp_start_with_batch_norm=mlp_start_with_batch_norm,
        )

    def forward(self, data):
        x, adj, nidx, batch = (
            data.subgraphs_x,
            data.subgraphs_adj,
            data.subgraphs_nidx,
            data.batch,
        )

        device = x.device

        unique, counts = torch.unique(batch, return_counts=True)
        n_graphs = unique.max() + 1

        responses = []
        mlp_inputs = []
        h = x

        def pool(x, pool_fn):
            pooled_lst = list()
            for r, g in x:
                if r.shape[0] != g.shape[0]:
                    r = r[:-1]
                pooled = pool_fn(r, g)
                pooled_lst += [pooled]
            return torch.cat(pooled_lst, -1)

        for layer_idx in range(len(self.ker_layers)):
            h = self.ker_layers[layer_idx](h, adj, nidx)
            responses += [(h, batch)]
            assert torch.all(h >= 0), h[h < 0]
            norm = global_add_pool(h + 1e-36, batch).sum(-1).unsqueeze(-1)
            norm = norm[batch]
            h = h / norm
            mlp_inputs += [(h, batch)]

        mlp_inputs = [(r * torch.log(r + 1e-36), g) for r, g in mlp_inputs]
        mlp_input = pool(mlp_inputs, global_add_pool)

        outputs = self.mlp(mlp_input)
        return outputs, responses, mlp_input, mlp_inputs
