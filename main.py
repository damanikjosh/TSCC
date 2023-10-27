import torch
from torch.utils.data import TensorDataset, DataLoader

import wandb
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from models import TSCC
from datautils import load_Flights, load_UCR

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Coffee')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--no_wandb', action='store_true')
args = parser.parse_args()

SEED = args.seed
DATASET = args.dataset
SAVE_PATH = Path('results', DATASET + '_' + datetime.now().strftime('%Y%m%d_%H%M%S')) if args.save_path is None else Path(args.save_path)

# Wandb
if not args.no_wandb:
    wandb.init(project='tscc', entity='joshuad', name=f'{DATASET}_{SEED}', config=args)

# Reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
g = torch.Generator()
g.manual_seed(SEED)

# train_data, train_labels, test_data, test_labels = load_Flights(DATASET)
train_data, train_labels, test_data, test_labels = load_UCR(DATASET)
data = np.concatenate((train_data, test_data), axis=0)
labels = np.concatenate((train_labels, test_labels), axis=0)

n_samples = len(data)
data_length = data.shape[1]
input_dims = data.shape[-1]
n_cluster = len(np.unique(labels))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 512 if n_samples > 4096 else 64 if n_samples > 512 else 8
# batch_size = 64
print(f'batch_size: {batch_size}')
dataset = TensorDataset(torch.arange(len(data)).long(), torch.from_numpy(data).float(), torch.from_numpy(labels).long())
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=g, worker_init_fn=seed_worker)
val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

ts2vecdec = TSCC(
    input_dims=input_dims,
    n_cluster=n_cluster,
    n_samples=len(dataset),
    latent_dims=64,
    hidden_dims=64,
    depth=[10],
    mask_mode='binomial',
    device=device
)


def plot_latent(epoch, y, pred, z):

    if z.shape[1] > 2:
        z_embedded = TSNE(n_components=2).fit_transform(z)
    else:
        z_embedded = z
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10))
    for label in np.unique(y):
        ax0.scatter(z_embedded[y == label, 0], z_embedded[y == label, 1], s=2, label=label)
    ax0.set_title(f'Ground truth. Epoch {epoch+1}')
    for label in np.unique(pred):
        ax1.scatter(z_embedded[pred == label, 0], z_embedded[pred == label, 1], s=2, label=label)
    ax1.set_title(f'Prediction. Epoch {epoch+1}')
    ax1.legend()
    fig.tight_layout()
    Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
    plt.savefig(SAVE_PATH / f'latent_{epoch+1}.png')
    plt.close()



# best_ari = 0.
# no_best_count = 0
def epoch_callback(epoch, metrics, outputs):
    loss, ri, ari, nmi, lr = metrics
    x, z, y, pred = outputs
    if not args.no_wandb:
        wandb_log(epoch, loss, ri, ari, nmi, lr)
    # plot_trajectory(epoch, x, pred)
    # plot_latent(epoch, y, pred, z)


def plot_trajectory(epoch, x, y):

    unique_labels = np.unique(y)
    num_labels = len(unique_labels)

    num_rows = np.round(np.sqrt(num_labels)).astype(int)
    num_cols = (num_labels // num_rows) + (num_labels % num_rows > 0)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows), sharey='all', sharex='all')
    for i, label in enumerate(unique_labels):
        ax = axes.flatten()[i]
        idx = np.where(y == label)[0]
        for id in idx:
            if input_dims == 1:
                ax.plot(x[id], c='k', linewidth=1, alpha=0.1)
            else:
                ax.plot(x[id, :, 0], x[id, :, 1], c='k', linewidth=1, alpha=0.1)
        ax.set_title(f'Label {label}')
    fig.tight_layout()
    Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
    plt.savefig(SAVE_PATH / f'trajectory_{epoch+1}.png')
    plt.close()


def wandb_log(epoch, loss, ri, ari, nmi, lr):
    wandb.log({
        'loss': loss,
        'ri': ri,
        'ari': ari,
        'nmi': nmi,
        'lr': lr,
    }, step=(epoch+1))


ts2vecdec.train(train_dataloader, 1e-3, n_epochs=1000, epoch_callback=epoch_callback)
