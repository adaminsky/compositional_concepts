from __future__ import annotations

import torch
from sklearn.cluster import KMeans


def learn_seminmf(X, k):
    # Learn X = WH where H is non-negative
    # X: (n, d)
    # W: (n, k)
    # H: (k, d)
    n, d = X.shape
    H_params = torch.nn.Parameter(torch.randn(k, d), requires_grad=True)
    H_params.data.clamp_(min=0)
    W = torch.nn.Parameter(torch.randn(n, k), requires_grad=True)
    kmeans = KMeans(n_clusters=k).fit(X.T)
    W.data = torch.tensor(kmeans.cluster_centers_.T).float()
    W.data = W.data / torch.norm(W.data, dim=0, keepdim=True)
    onehot = torch.eye(k)[kmeans.labels_.reshape(-1)]
    H_params.data = onehot.T + 0.2
    # H_params.data = (W.T @ X).clamp_(min=0)

    optimizer_w = torch.optim.Adam([W], lr=0.01)
    optimizer_h = torch.optim.Adam([H_params], lr=0.01)
    for i in range(4000):
        optimizer_w.zero_grad()
        optimizer_h.zero_grad()
        WH = torch.mm(W, H_params)

        # H_opt = torch.nn.functional.relu(torch.inverse(W.T @ W) @ W.T @ X)
        WH = torch.mm(W, H_params)

        loss = torch.norm(X - WH, p="fro")  # + 0.01 * torch.norm(H_params, p=1)
        loss.backward()
        optimizer_w.step()
        optimizer_h.step()

        if i % 100 == 0:
            print(loss.item())

        with torch.no_grad():
            W.data = W.data / torch.norm(W.data, dim=0, keepdim=True)
            H_params.data = H_params.data.clamp_(min=0)

    return W.detach(), H_params.detach()


def learn_seminmf_2(X, k, seed=0):
    torch.manual_seed(seed)
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

    for i in range(5000):
        F = X @ G @ torch.inverse(G.T @ G)
        G = G * torch.sqrt(
            (get_pos(X.T @ F) + (G @ get_neg(F.T @ F)))
            / (1e-5 + get_neg(X.T @ F) + (G @ get_pos(F.T @ F)))
        )

        print(torch.norm(X - F @ G.T, p="fro"))

    return F / torch.norm(F, dim=0, keepdim=True), G
