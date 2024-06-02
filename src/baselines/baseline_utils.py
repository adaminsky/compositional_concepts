from __future__ import annotations

import torch
from sklearn.cluster import KMeans
from tqdm import tqdm


def learn_seminmf_2(X, k):
    n, d = X.shape
    F = torch.randn(n, k)
    F = F / torch.norm(F, dim=0, keepdim=True)
    G = torch.randn(k, d).T

    kmeans = KMeans(n_clusters=k).fit(X.T)
    G = torch.zeros_like(G.data)
    print(kmeans.labels_.shape)
    onehot = torch.eye(k)[kmeans.labels_.reshape(-1)]
    print(onehot.shape)
    G = onehot
    G = G + 0.2

    def get_pos(A):
        return torch.nn.functional.relu(A)

    def get_neg(A):
        return torch.nn.functional.relu(-A)

    error = 0
    iterator = tqdm(range(5000), desc="Processing", unit="error")
    for i in iterator:
        F = X @ G @ torch.inverse(G.T @ G)
        G = G * torch.sqrt(
            (get_pos(X.T @ F) + (G @ get_neg(F.T @ F)))
            / (get_neg(X.T @ F) + (G @ get_pos(F.T @ F)))
        )

        error = torch.norm(X - F @ G.T, p="fro")

        iterator.set_description(f"Reconstruction error {error.item()!s}")
        # tqdm.refresh() # to show immediately the update

    return F / torch.norm(F, dim=0, keepdim=True), G
