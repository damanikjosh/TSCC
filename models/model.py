import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score, adjusted_rand_score, normalized_mutual_info_score
from torch.optim.swa_utils import AveragedModel

from .TS2Vec import TSEncoder, hierarchical_contrastive_loss

import logging

logger = logging.getLogger('TSCC')

def take_per_row(A, indx, num_elem):
    # all_indx = indx[:,None] + np.arange(num_elem)
    all_indx = indx[:, None] + torch.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]


class TSCC:
    def __init__(self,
                 input_dims,
                 n_cluster,
                 n_samples,
                 latent_dims=8,
                 hidden_dims=64,
                 depth=10,
                 mask_mode='binomial',
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        self.input_dims = input_dims
        self.n_cluster = n_cluster
        self.latent_dims = latent_dims

        self._model = TSEncoder(input_dims, latent_dims, n_cluster, hidden_dims, depth, mask_mode).to(device)
        # self._model = Encoder(input_dims, max_length, n_cluster, latent_dims, hidden_dims, depth, mask_mode).to(device)
        self.model = AveragedModel(self._model)
        self.model.update_parameters(self._model)
        # self.positional_encoding = PositionalEncoding(d_model=1, max_len=max_length)
        self.device = device

    def train(self, train_dataloader, lr, n_epochs, epoch_callback=None, end_callback=None, crop=True):
        logger.info('Training...')

        optimizer = Adam(self._model.parameters(), lr=lr, weight_decay=0.)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

        for epoch in range(n_epochs):
            train_loss = 0.

            for idx_batch, x_batch, y_batch in train_dataloader:
                if crop:
                    ts_l = x_batch.size(1)

                    crop_l = torch.randint(low=2 ** (0 + 1), high=ts_l + 1, size=(1,)).item()
                    crop_left = torch.randint(low=0, high=ts_l - crop_l + 1, size=(1,)).item()
                    crop_right = crop_left + crop_l
                    crop_eleft = torch.randint(low=0, high=crop_left + 1, size=(1,)).item()
                    crop_eright = torch.randint(low=crop_right, high=ts_l + 1, size=(1,)).item()
                    crop_offset = torch.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=(x_batch.size(0),))

                    x_aug1_batch = take_per_row(x_batch, crop_offset + crop_eleft, crop_right - crop_eleft).to(
                        self.device)
                    x_aug2_batch = take_per_row(x_batch, crop_offset + crop_left, crop_eright - crop_left).to(
                        self.device)

                    r1_batch, z1_batch = self._model(x_aug1_batch)
                    r2_batch, z2_batch = self._model(x_aug2_batch)

                    r1_batch = r1_batch[:, -crop_l:]
                    r2_batch = r2_batch[:, :crop_l]
                else:
                    x_aug1_batch = x_batch.to(self.device)
                    x_aug2_batch = x_batch.to(self.device)

                    r1_batch, z1_batch = self._model(x_aug1_batch, 0)
                    r2_batch, z2_batch = self._model(x_aug2_batch, 0)

                loss = self._model.hierarchical_loss(r1_batch, r2_batch, z1_batch, z2_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_dataloader)
            self.model.update_parameters(self._model)

            if (epoch + 1) % 10 == 0:
                # torch.optim.swa_utils.update_bn(dataloader, self.model, device=self.device)
                idx = []
                x = []
                y = []
                pred = []
                z = []
                with torch.no_grad():
                    for idx_batch, x_batch, y_batch in train_dataloader:
                        z_batch, pred_batch = self.model(x_batch.to(self.device))
                        z_batch = nn.MaxPool1d(z_batch.size(1))(z_batch.transpose(1, 2)).squeeze(-1)
                        # pred_batch = nn.AvgPool1d(pred_batch.size(1))(pred_batch.transpose(1, 2)).squeeze(-1)

                        idx.append(idx_batch.cpu())
                        x.append(x_batch.cpu())
                        y.append(y_batch.cpu())
                        z.append(z_batch.cpu())
                        pred.append(pred_batch.cpu())

                idx = torch.cat(idx, dim=0)
                x = torch.cat(x, dim=0)
                y = torch.cat(y, dim=0)
                z = torch.cat(z, dim=0)
                pred = torch.cat(pred, dim=0)

                # kmeans = KMeans(n_clusters=self.n_cluster, n_init=20)
                # predicted = kmeans.fit_predict(pred.numpy())
                predicted = torch.argmax(pred, dim=1).numpy()
                ri = rand_score(y.numpy(), predicted)
                ari = adjusted_rand_score(y.numpy(), predicted)
                nmi = normalized_mutual_info_score(y.numpy(), predicted)

                logger.info(f'Epoch: {epoch + 1}, Loss: {train_loss:.4f}, RI: {ri:.4f}, ARI: {ari:.4f}, NMI: {nmi:.4f}')
                if epoch_callback is not None:
                    epoch_callback(epoch, (train_loss, ri, ari, nmi, optimizer.param_groups[0]['lr']), (x, z, y, predicted))
            # scheduler.step()

        if end_callback is not None:
            end_callback(epoch, train_loss)
