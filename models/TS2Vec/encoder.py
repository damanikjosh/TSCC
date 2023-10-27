import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .dilated_conv import DilatedConvEncoder
from .inception import InceptionTime
import math


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)

    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)

    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T - l + 1)
            res[i, t:t + l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)
    # return torch.full((B, T), True, dtype=torch.bool).bernoulli_(p)

class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, n_cluster, hidden_dims=64, depths=[5, 5], mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.n_cluster = n_cluster
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)

        self.repr_dropout = nn.Dropout(p=0.1)
        self.repr_projection = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims)
        )
        self.cluster_projection = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, n_cluster),
            nn.Softmax(dim=-1)
        )
        self.layers = []
        for i, depth in enumerate(depths):
            self.layers.append(DilatedConvEncoder(
                hidden_dims,
                [hidden_dims] * depth,
                kernel_size=3
            ))
        self.layers = nn.ModuleList(self.layers)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x, pos_start=0, mask=None):  # x: B x T x input_dims
        # print(f'x.shape: {x.shape}')
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask = (~mask) | (~nan_mask)
        x[mask] = 0

        x = x.transpose(1, 2)  # B x Ch x T
        for i in range(len(self.layers)):
            x = self.layers[i](x) + x
            x = self.repr_dropout(x)
        x = x.transpose(1, 2)  # B x T x Ch

        r = self.repr_projection(x) # B x T x output_dims

        z = nn.MaxPool1d(kernel_size=x.size(1))(x.transpose(1, 2)).squeeze(-1)
        p = self.cluster_projection(z) # B x T x n_cluster

        return r, p

    def hierarchical_loss(self, z1, z2, p1, p2):
        loss = torch.tensor(0., device=z1.device)

        d = 0
        while z1.size(1) > 1:
            loss += self.instance_contrastive_loss(z1, z2)
            loss += self.temporal_contrastive_loss(z1, z2)

            z1 = self.max_pool(z1.transpose(1, 2)).transpose(1, 2)
            z2 = self.max_pool(z2.transpose(1, 2)).transpose(1, 2)

        loss += self.instance_contrastive_loss(z1, z2)
        loss += self.cluster_single_contrastive_loss(p1, p2)
        d += 1
        return loss / d

    @staticmethod
    def instance_contrastive_loss(z1, z2, ti=1.0):
        B, T = z1.size(0), z1.size(1)
        if B == 1:
            return z1.new_tensor(0.)
        z = torch.cat([z1, z2], dim=0)  # 2B x T x Ch
        z = z.transpose(0, 1)  # T x 2B x Ch

        sim = torch.matmul(z, z.transpose(1, 2)) / ti  # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss

    @staticmethod
    def temporal_contrastive_loss(z1, z2, tt=1.0):
        B, T = z1.size(0), z1.size(1)
        if T == 1:
            return z1.new_tensor(0.)
        z = torch.cat([z1, z2], dim=1) / tt  # B x 2T x C
        sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        t = torch.arange(T, device=z1.device)
        loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
        return loss

    @staticmethod
    def cluster_single_contrastive_loss(y1, y2, tc=1.0):
        B, C = y1.size(0), y1.size(1)
        if C == 1:
            return y1.new_tensor(0.)

        y = torch.cat([y1, y2], dim=1)  # B x 2C
        y = y.transpose(0, 1)  # 2C x B

        sim = torch.matmul(y, y.transpose(0, 1)) / tc  # 2C x 2C
        logits = torch.tril(sim, diagonal=-1)[:, :-1]  # (2C-1) x 2C
        logits += torch.triu(sim, diagonal=1)[:, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        i = torch.arange(C, device=y1.device)
        loss = (logits[i, C + i - 1].mean() + logits[C + i, i].mean()) / 2

        P1 = torch.mean(y1, dim=0)
        P2 = torch.mean(y2, dim=0)
        entropy = torch.sum(P1 * torch.log(P1) + P2 * torch.log(P2))

        return loss + entropy

