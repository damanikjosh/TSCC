import torch
from torch import nn
import torch.nn.functional as F


def hierarchical_contrastive_loss(r1, r2, z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=r1.device)
    d = 0
    while r1.size(1) > 1:
        loss += instance_contrastive_loss(r1, r2)
        loss += temporal_contrastive_loss(r1, r2)
        # loss += cluster_contrastive_loss(z1, z2)
        # loss += temporal_contrastive_loss(z1, z2)
        d += 1

        # print(f'z1.shape: {z1.shape}')
        # print(f'z2.shape: {z2.shape}')
        r1 = F.max_pool1d(r1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        r2 = F.max_pool1d(r2.transpose(1, 2), kernel_size=2).transpose(1, 2)

        # z1 = F.avg_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        # z2 = F.avg_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if r1.size(1) == 1:
        z1 = nn.AvgPool1d(kernel_size=z1.size(1))(z1.transpose(1, 2)).transpose(1, 2)
        z2 = nn.AvgPool1d(kernel_size=z2.size(1))(z2.transpose(1, 2)).transpose(1, 2)
        loss += instance_contrastive_loss(r1, r2)
        loss += cluster_contrastive_loss(z1, z2)
    return loss / d

def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss

def cluster_contrastive_loss(y1, y2):
    B, T, C = y1.size(0), y1.size(1), y1.size(2)
    if C == 1:
        return y1.new_tensor(0.)

    y = torch.cat([y1, y2], dim=2)  # B x T x 2C
    y = y.transpose(0, 1) # T x B x 2C
    y = y.transpose(1, 2) # T x 2C x B

    y = y / torch.norm(y, dim=-1, keepdim=True) # T x 2C x B
    sim = torch.matmul(y, y.transpose(1, 2)) # T x 2C x 2C
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1] # T x 2C x (2C-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(C, device=y1.device)
    loss = (logits[:, i, C + i - 1].mean() + logits[:, C + i, i].mean()) / 2

    P1 = torch.mean(y1, dim=0)
    P2 = torch.mean(y2, dim=0)
    entropy = torch.mean(P1 * torch.log(P1) + P2 * torch.log(P2))

    return loss + entropy