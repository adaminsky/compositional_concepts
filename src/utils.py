from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass

import numpy as np
import PIL
import PIL.Image
import torch
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class Patch:
    image: PIL.Image
    bbox: tuple
    # patch: PIL.Image


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, processed):
        self.input_ids = processed["input_ids"]
        self.attention_mask = processed["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self

    def __getitem__(self, idx):
        return self.input_ids, self.attention_mask


def load(filename):
    try:
        with open(filename, "rb") as f:
            obj = pickle.load(f)
    except:
        raise Exception("File not found")
    return obj


def save(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def normalize(x):
    if type(x) == torch.Tensor:
        return x / (x.norm(dim=1)[:, None] + 1e-6)
    else:
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)


def whiten(X, return_fit=False):
    """
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Input data matrix with data examples along the first dimension
    """
    # X = X.reshape((-1, torch.prod(X.shape[1:])))
    X_centered = X - torch.mean(X, dim=0)
    Sigma = torch.mm(X_centered.T, X_centered) / (X_centered.shape[0] - 1)

    U, S, _ = torch.linalg.svd(Sigma)
    s = torch.sqrt(S.clip(1e-5))
    s_inv = torch.diag(1.0 / s)
    s = torch.diag(s)
    whiten = torch.mm(torch.mm(U, s_inv), U.T)
    dewhiten = torch.mm(torch.mm(U, s), U.T)

    if return_fit:
        return torch.mm(X_centered, whiten.T), torch.mean(X, dim=0), whiten, dewhiten
    else:
        return torch.mm(X_centered, whiten.T)


def apply_whiten(X, mean, whiten):
    X_centered = X - mean
    return torch.mm(X_centered, whiten.T)


def unwhiten(X, mean, dewhiten):
    return torch.mm(X, dewhiten) + mean


def centering(x, center=None, std=None):
    if type(x) == torch.Tensor:
        # x = x / (x.norm(dim=1)[:, None] + 1e-6)
        if center is not None and std is not None:
            x = (x - center) / std
        else:
            x = (x - x.mean(dim=0).detach()) / x.std(dim=0).detach()
    elif center is not None and std is not None:
        x = (x - center) / std
    else:
        x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    return x


def preprocessing_data(data):
    data = centering(data)
    # data = normalize(data)
    return data


def cosim(x, y):
    if len(x.shape) == 1 and len(y.shape) == 1:
        return torch.dot(x, y) / (torch.norm(x) * torch.norm(y))
    return torch.mm(x, y.T) / (
        torch.norm(x, dim=1)[:, None] * torch.norm(y, dim=1)[None, :]
    )


def clean_small_values(data, threshold=1e-4):
    data[torch.logical_and(data > 0, data < threshold)] = threshold
    data[torch.logical_and(data < 0, data > -threshold)] = -threshold
    return data


def compute_cosine_similarity_matrix(data1, data2):
    data1_c = torch.where(
        torch.logical_and(data1 > 0, data1 < 1e-4), torch.full_like(data1, 1e-4), data1
    )
    data2_c = torch.where(
        torch.logical_and(data2 < 0, data2 > -1e-4),
        torch.full_like(data2, -1e-4),
        data2,
    )
    # data1_c = clean_small_values(data1)
    # data2_c = clean_small_values(data2)
    sim_matrix = torch.mm(data1_c, data2_c.T) / (
        torch.norm(data1_c, dim=1).unsqueeze(1) * torch.norm(data2_c, dim=1) + 1e-4
    )
    return sim_matrix


def kpp(data: torch.Tensor, k: int, sample_size: int = -1, cosine=False, dot=False):
    """Picks k points in the data based on the kmeans++ method.

    Parameters
    ----------
    data : torch.Tensor
        Expect a rank 1 or 2 array. Rank 1 is assumed to describe 1-D
        data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    sample_size : int
        sample data to avoid memory overflow during calculation

    Returns
    -------
    init : ndarray
        A 'k' by 'N' containing the initial centroids.

    References
    ----------
    .. [1] D. Arthur and S. Vassilvitskii, "k-means++: the advantages of
       careful seeding", Proceedings of the Eighteenth Annual ACM-SIAM Symposium
       on Discrete Algorithms, 2007.
    .. [2] scipy/cluster/vq.py: _kpp
    """
    if sample_size > 0:
        data = data[
            torch.randint(
                0, int(data.shape[0]), [min(100000, data.shape[0])], device=data.device
            )
        ]
    dims = data.shape[1] if len(data.shape) > 1 else 1
    init = torch.zeros((k, dims)).to(data.device)

    r = torch.distributions.uniform.Uniform(0, 1)
    for i in range(k):
        if i == 0:
            init[i, :] = data[torch.randint(data.shape[0], [1])]

        else:
            if cosine:
                D2 = (1 - compute_cosine_similarity_matrix(init[:i, :], data)).amin(
                    dim=0
                )
            elif dot:
                D2 = -torch.mm(init[:i, :], data.T).max(dim=0)[0]
            else:
                D2 = torch.cdist(init[:i, :][None, :], data[None, :], p=2)[0].amin(
                    dim=0
                )
            probs = D2 / torch.sum(D2)
            cumprobs = torch.cumsum(probs, dim=0)
            init[i, :] = data[
                torch.searchsorted(cumprobs, r.sample([1]).to(data.device))
            ]
    return init


def ward_init(data, k, cosine=False):
    if cosine:
        distances = 1 - compute_cosine_similarity_matrix(data, data)
    else:
        distances = torch.cdist(data[None, :], data[None, :])[0]
    clustering = AgglomerativeClustering(
        n_clusters=k, linkage="complete", metric="precomputed"
    ).fit(distances.detach().cpu().numpy())
    labels = clustering.labels_

    init = torch.zeros((k, data.shape[1])).to(data.device)
    for i in range(k):
        init[i] = torch.mean(data[labels == i], axis=0)
    return init


def kmeans_clustering(
    data, k, max_iters=10000, tol=1e-4, init=None, repeats=1, cosine=False, dot=False
):
    # Randomly initialize centroids
    if init is None:
        centroids = kpp(data, k, cosine=cosine, dot=dot)
    else:
        centroids = init
    # centroids = ward_init(data, k, cosine=cosine)
    # centroids = data[torch.randperm(len(data))[:k]]
    # centroids = torch.rand(k, data.shape[1]).to(data.device)
    if cosine or dot:
        centroids = normalize(centroids)
    prev_centroids = centroids.clone()

    for _ in range(max_iters):
        # Assign each data point to the nearest centroid
        # sim_matrix = torch.cosine_similarity(data.unsqueeze(0), centroids.unsqueeze(1))
        if cosine:
            # sim_matrix = torch.mm(data, centroids.T)/(torch.norm(data, dim=1).unsqueeze(1)*torch.norm(centroids, dim=1))
            sim_matrix = compute_cosine_similarity_matrix(data, centroids)
        elif dot:
            sim_matrix = torch.mm(data, centroids.T)
        else:
            sim_matrix = -torch.cdist(data, centroids)
        assignments = torch.argmax(sim_matrix, axis=1)
        # assignments = torch.argmin(np.array([[cosine_similarity(data[i], centroid) for centroid in centroids] for i in range(len(data))]), axis=1)

        # Update centroids
        for i in range(k):
            cluster_points = data[assignments == i]
            if len(cluster_points) > 0:
                if not cosine and not dot:
                    centroids[i] = torch.mean(cluster_points, axis=0, keepdim=True)
                else:
                    centroids[i] = normalize(
                        torch.mean(normalize(cluster_points), axis=0, keepdim=True)
                    )[0]

        # Check convergence
        if torch.norm(centroids - prev_centroids) < tol:
            break

        prev_centroids = centroids.clone()

    return assignments, centroids


def kmeans_assign(data, centroids, cosine=False):
    if cosine:
        sim_matrix = compute_cosine_similarity_matrix(data, centroids)
    else:
        sim_matrix = -torch.cdist(data, centroids)
    assignments = torch.argmax(sim_matrix, axis=1)
    return assignments


def torch_to_numpy(tensor):
    try:
        return tensor.detach().cpu().numpy()
    except:
        return np.array(tensor)


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, list):
        return [_to_device(xi, device) for xi in x]
    elif isinstance(x, TextDataset):
        return x.to(device)


def hashfn(x: list):
    if type(x[0]) == PIL.Image.Image:
        samples_hash = hashlib.sha1(
            np.stack([img.resize((32, 32)) for img in tqdm(x)])
        ).hexdigest()
    else:
        samples_hash = hashlib.sha1(np.array(x)).hexdigest()
    return samples_hash


def gini(x: torch.Tensor):
    x = x.view(-1)
    n = x.size(0)
    x = x.sort()[0]
    index = torch.arange(1, n + 1).float()
    pos = x[x > 0].sum()
    neg = x[x <= 0].abs().sum()
    norm = abs(pos - neg) / (pos + neg)
    return norm * (x * ((2 * n) + 1 - (2 * index))).sum() / (n * x.sum())


def hoyer(x: torch.Tensor):
    x = x.view(-1)
    n = torch.tensor(x.size(0))
    return (torch.sqrt(n) - (torch.norm(x, p=1) / torch.norm(x, p=2))) / (
        torch.sqrt(n) - 1
    )


def l2l1(x: torch.Tensor):
    x = x.view(-1)
    return torch.norm(x, p=2) / torch.norm(x, p=1)


def _batch_inference(
    model, dataset: Dataset, batch_size=128, resize=None, processor=None, device="cuda"
) -> torch.Tensor:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True
    )

    results = []
    with torch.no_grad():
        for batch in tqdm(loader):
            # for i in tqdm(range(0, len(dataset), batch_size)):
            # batch = dataset[i:i+batch_size]
            if processor is not None:
                batch = processor(batch)
            x = _to_device(batch.squeeze(), device)

            if resize:
                x = torch.nn.functional.interpolate(
                    x, size=resize, mode="bilinear", align_corners=False
                )

            results.append(model(x).cpu())

    results = torch.cat(results)
    return results


def _batch_patch_emb(
    model, patches, batch_size=128, resize=None, processor=None, device="cuda"
) -> torch.Tensor:
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    results = []
    with torch.no_grad():
        # for batch in tqdm(loader):
        for i in tqdm(range(0, len(patches), batch_size)):
            batch = patches[i : i + batch_size]
            if type(batch[0]) == Patch:
                batch = [patch.image.crop(patch.bbox) for patch in batch]
            if processor is not None:
                batch = processor(batch)
            x = _to_device(batch, device)

            if resize:
                x = torch.nn.functional.interpolate(
                    x, size=resize, mode="bilinear", align_corners=False
                )

            results.append(model(x).cpu())

    results = torch.cat(results)
    return results
