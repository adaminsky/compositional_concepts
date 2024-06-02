from __future__ import annotations

import itertools
import os

import numpy as np
import PIL
import torch
from fast_pytorch_kmeans import KMeans
from scipy.stats import pearsonr
from skimage.measure import label, regionprops
from skimage.segmentation import slic
from sklearn.decomposition import PCA, MiniBatchDictionaryLearning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import pipeline

import utils
from nmf import learn_seminmf_2
from silhouette import Silhouette
from utils import Patch


def concept_learning_test(concepts):
    c = torch.tensor(concepts[:1]).double()
    # c = torch.randn(50, concepts.shape[1]).double()
    c = c / c.norm(dim=1)[:, None]
    new_concepts = Parameter(c, requires_grad=True)
    new_concepts.requires_grad = True
    optimizer = torch.optim.SGD([new_concepts], lr=0.00001)
    for i in tqdm(range(10000)):
        new_concepts.data = new_concepts.data / new_concepts.data.norm(dim=1)[:, None]
        optimizer.zero_grad()
        loss = (
            -3 * torch.maximum(concepts @ new_concepts.T - 0.4, torch.tensor(0.0)).sum()
            + (torch.minimum(concepts @ new_concepts.T - 0.4, torch.tensor(0.0)) + 0.4)
            .square()
            .sum()
            + (new_concepts @ new_concepts.T - torch.eye(50)).square().sum()
        )
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(loss.item())
    return new_concepts


def concept_learning_test2(concepts):
    c = torch.randn(10, concepts.shape[1]).double()
    c = c / c.norm(dim=1)[:, None]
    new_concepts = Parameter(c, requires_grad=True)
    new_concepts.requires_grad = True
    optimizer = torch.optim.SGD([new_concepts], lr=0.0000001)
    N = concepts.shape[0]
    for i in tqdm(range(10000)):
        new_concepts.data = new_concepts.data / new_concepts.data.norm(dim=1)[:, None]
        optimizer.zero_grad()
        with torch.no_grad():
            dot = concepts @ new_concepts.T
            argsort_max = torch.argsort(-dot, dim=0)[: N // 8, :]
            argsort_min = torch.argsort(-dot, dim=0)[N // 8 :, :]
        # loss = -10 * (concepts @ new_concepts.T)[argsort_max].sum() \
        #     + (concepts @ new_concepts.T)[argsort_min].square().sum() \
        #     + 5000 * (new_concepts @ new_concepts.T - torch.eye(10)).square().sum()
        loss = (concepts @ new_concepts.T)[argsort_min].square().sum() / (
            concepts @ new_concepts.T
        )[argsort_max].sum()
        # + 0.001 * (new_concepts @ new_concepts.T - torch.eye(10)).square().sum()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(loss.item())
    return new_concepts


def learn_sup_split_concept(activations, concept_labels, lr=0.1):
    linear_layer = torch.nn.Linear(
        activations.shape[1], concept_labels.shape[1], bias=False
    )
    torch.nn.init.orthogonal_(linear_layer.weight)
    split_concept = linear_layer
    # split_concept = torch.nn.utils.parametrizations.orthogonal(
    #     linear_layer)
    # with torch.no_grad():
    #     print(split_concept.weight @ split_concept.weight.T)

    data = TensorDataset(activations, concept_labels)
    loader = DataLoader(data, batch_size=2000, shuffle=True)
    optimizer = torch.optim.SGD(split_concept.parameters(), lr=lr)
    for _ in range(5000):
        for i, batch in enumerate(loader):
            batch, labels = batch
            optimizer.zero_grad()
            with torch.no_grad():
                split_concept.weight.data = (
                    split_concept.weight.data
                    / split_concept.weight.data.norm(dim=1)[:, None]
                )
            project = split_concept(batch)

            with torch.no_grad():
                top_mask = labels == 1
                bottom_mask = labels == 0

            loss = (
                torch.sum(
                    project[bottom_mask].abs().mean(dim=0)
                    - project[top_mask].mean(dim=0)
                )
                + 10
                * (
                    split_concept.weight @ split_concept.weight.T
                    - torch.eye(concept_labels.shape[1])
                )
                .square()
                .sum()
            )
            # loss = torch.sum(- project[top_mask].mean(dim=0)) \
            #     + 1 * (split_concept.weight @ split_concept.weight.T - torch.eye(concept_labels.shape[1])).square().sum()

            loss.backward()
            optimizer.step()
            # if i % 500 == 0:
        print(loss.item())
        # print(activations[:10] @ activations[:10].T)
        # print(split_concept.weight @ split_concept.weight.T)
    return split_concept.weight.detach()  # , split_concept.bias.detach()


def learn_split_concept(activations, n_concepts=1, lr=0.1, device="cpu"):
    split_concept = torch.nn.Linear(activations.shape[1], n_concepts, bias=False).to(
        device
    )
    # torch.nn.init.orthogonal_(split_concept.weight.data)
    # split_concept = torch.nn.utils.parametrizations.orthogonal(split_concept)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(activations)
    # ortho_clusters = torch.linalg.qr(torch.tensor(kmeans.cluster_centers_[:n_concepts].T))[0].T[:n_concepts]
    # split_concept.weight.data = torch.tensor(ortho_clusters).float().to(device)
    split_concept.weight.data = kmeans.centroids[:n_concepts].float().to(device)
    split_concept.weight.data /= torch.linalg.norm(split_concept.weight.data, dim=1)[
        :, None
    ]

    print(activations.shape)
    data = TensorDataset(activations)
    loader = DataLoader(data, batch_size=5000, shuffle=True, pin_memory=True)
    optimizer = torch.optim.SGD(split_concept.parameters(), lr=lr)
    for _ in range(400):
        for i, batch in enumerate(loader):
            # normalize the concepts to unit norm (PGD)
            with torch.no_grad():
                split_concept.weight.data = (
                    split_concept.weight.data
                    / split_concept.weight.data.norm(dim=1)[:, None]
                )

            batch = batch[0].to(device)
            optimizer.zero_grad()
            project = split_concept(batch)

            with torch.no_grad():
                top_mask = project >= torch.quantile(
                    project, 0.9, dim=0
                )  # This is the "fitness" objective
                bottom_mask = project < torch.quantile(
                    project, 0.1, dim=0
                )  # This is a sparsity constraint
                mid_mask = (project < torch.quantile(project, 0.9, dim=0)) & (
                    project >= torch.quantile(project, 0.1, dim=0)
                )

            loss = torch.sum(
                project[bottom_mask].abs().mean(dim=0) - project[top_mask].mean(dim=0)
            )
            loss.backward()
            optimizer.step()
        print(loss.item())
    return normalize(split_concept.weight.detach().cpu())


def learn_split_subspace(
    activations,
    n_components=3,
    n_clusters=-1,
    lr=0.1,
    device="cpu",
    cosine=False,
    cross_entropy=False,
    no_reg=False,
    epochs=300,
):
    split_space = torch.nn.Linear(activations.shape[1], n_components, bias=False).to(
        device
    )
    if cross_entropy:
        classifier = torch.nn.Linear(n_components, n_clusters).to(device)
    torch.nn.init.orthogonal_(split_space.weight.data)
    # pca_comp = normalize(torch.tensor(PCA(n_components=n_components).fit(activations.numpy()).components_))
    # split_space.weight.data = pca_comp.float().to(device)
    split_space = torch.nn.utils.parametrizations.orthogonal(split_space)

    if n_clusters == -1:
        with torch.no_grad():
            train_act, val_act = train_test_split(
                activations @ split_space.weight.data.detach().cpu().T, test_size=0.5
            )
        scores = []
        for nc in range(1, 10):
            score = []
            for _ in range(3):
                _, centroids = utils.kmeans_clustering(train_act, nc, cosine=cosine)
                labels = utils.kmeans_assign(val_act, centroids, cosine=cosine)
                # kmeans = KMeans(n_clusters=nc, init_method="kmeans++", minibatch=len(train_act))
                # kmeans.fit(train_act)
                # labels = kmeans.predict(val_act)
                score.append(
                    Silhouette().score(val_act, labels, fast=False, cosine=cosine)
                )
            scores.append(np.mean(score))
        print(scores)
        n_clusters = np.argmax(scores) + 1
        print("Optimal number of components:", n_clusters)

    print(activations.shape)
    data = TensorDataset(activations)
    loader = DataLoader(data, batch_size=5000, shuffle=True, pin_memory=True)
    if cross_entropy:
        optimizer = torch.optim.Adam(
            list(split_space.parameters()) + list(classifier.parameters()), lr=lr
        )
    else:
        optimizer = torch.optim.Adam(split_space.parameters(), lr=lr)
        silhouette = Silhouette()

    best_loss = torch.inf
    best_space = None
    mult = 0.1
    init = None
    for j in tqdm(range(epochs)):
        total_sil = 0
        for i, batch in enumerate(loader):
            batch = batch[0].to(device)
            optimizer.zero_grad()
            project = split_space(batch)
            project_orig = project @ split_space.weight

            with torch.no_grad():
                clabels, centroids = utils.kmeans_clustering(
                    project, n_clusters, init=init, cosine=cosine
                )
                # init = centroids

            if cross_entropy:
                sil = torch.nn.functional.cross_entropy(project, clabels)
            else:
                sil = silhouette.score(
                    project, clabels, loss=True, cosine=cosine, fast=False
                )
            print("Silhouette:", sil.item())
            if not no_reg:
                for i in range(n_clusters):
                    if torch.sum(clabels == i) == 0:
                        continue

                    centroid = batch[clabels == i].mean(dim=0, keepdims=True)
                    proj_centroid = project_orig[clabels == i].mean(
                        dim=0, keepdims=True
                    )
                    with torch.no_grad():
                        print(
                            "cosine diff:",
                            utils.cosim(centroid, proj_centroid)[0, 0].item(),
                        )
                    sil -= (1 / (mult * n_clusters)) * utils.cosim(
                        centroid, proj_centroid
                    )[0, 0]

            sil.backward()
            total_sil += sil.item()
            optimizer.step()
        print(total_sil / len(loader))

        if total_sil < best_loss:
            best_loss = sil.item()
            best_space = split_space.weight.detach().cpu()

    _, centroids = utils.kmeans_clustering(
        activations @ best_space.T, n_clusters, cosine=cosine
    )
    return best_space, centroids.detach().cpu() @ best_space
    # orig_centroids = []
    # for i in range(n_clusters):
    #     orig_centroids.append(activations[clabels.cpu() == i].mean(dim=0, keepdims=True))
    # orig_centroids = normalize(torch.cat(orig_centroids, dim=0))
    # subspace = torch.linalg.qr(orig_centroids.T)[0].T.detach().cpu()
    # # subspace = best_space
    # return subspace, orig_centroids.detach().cpu()


def learn_split_subspace2(
    activations, n_components=3, n_clusters=2, lr=0.1, device="cpu"
):
    subspace = []
    for _ in range(n_components):
        split_space = torch.nn.Linear(activations.shape[1], 1, bias=False).to(device)
        torch.nn.init.orthogonal_(split_space.weight.data)
        split_space.weight.data /= torch.linalg.norm(split_space.weight.data, dim=1)[
            :, None
        ]

        data = TensorDataset(activations)
        loader = DataLoader(data, batch_size=5000, shuffle=True, pin_memory=True)
        optimizer = torch.optim.Adam(split_space.parameters(), lr=lr)
        for _ in range(400):
            for i, batch in enumerate(loader):
                # normalize the concepts to unit norm (PGD)
                with torch.no_grad():
                    split_space.weight.data = (
                        split_space.weight.data
                        / split_space.weight.data.norm(dim=1)[:, None]
                    )

                batch = batch[0].to(device)
                optimizer.zero_grad()
                project = split_space(batch)

                with torch.no_grad():
                    top_mask = project >= torch.quantile(
                        project, 0.9, dim=0
                    )  # This is the "fitness" objective
                    bottom_mask = project < torch.quantile(
                        project, 0.1, dim=0
                    )  # This is a sparsity constraint

                loss = torch.sum(
                    project[bottom_mask].abs().mean(dim=0)
                    - project[top_mask].mean(dim=0)
                )
                if len(subspace) > 0:
                    with torch.no_grad():
                        overal = torch.nn.functional.log_softmax(
                            (batch @ torch.cat(subspace, dim=0).T).sum(dim=1), 0
                        )
                    loss += torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(project, 1),
                        overal,
                        log_target=True,
                    )

                loss.backward()
                optimizer.step()
            print(loss.item())
        subspace.append(split_space.weight.detach().cpu())
    return normalize(torch.cat(subspace, dim=0))


def learn_split_concept_ae(activations, lr=0.1):
    best_loss = torch.inf
    best_concept = None  # split_concept.weight.detach()
    for _ in range(1):
        split_concept = torch.nn.utils.parametrizations.orthogonal(
            torch.nn.Linear(activations.shape[1], 3, bias=False)
        )
        optimizer = torch.optim.SGD(split_concept.parameters(), lr=lr)
        for i in range(5000):
            optimizer.zero_grad()
            project = split_concept(activations)

            recreate = project @ split_concept.weight
            loss = (
                1 + torch.nn.functional.mse_loss(recreate, activations)
            ) / torch.nn.functional.mse_loss(project, torch.zeros_like(project))
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(loss.item())
                # print(activations[:10] @ activations[:10].T)
                # print(split_concept.weight @ split_concept.weight.T)
        if loss.item() < best_loss:
            print(loss.item())
            best_loss = loss.item()
            best_concept = split_concept.weight.detach()
    return best_concept


def learn_linear_op(input_emb, output_emb, lr=0.1):
    best_loss = torch.inf
    best_concept = None  # split_concept.weight.detach()
    for _ in range(1):
        op = torch.nn.utils.parametrizations.orthogonal(
            torch.nn.Linear(input_emb.shape[1], input_emb.shape[1], bias=False)
        )
        optimizer = torch.optim.Adam(op.parameters(), lr=lr)
        for i in range(100):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(op(input_emb), output_emb)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(loss.item())
        if loss.item() < best_loss:
            print(loss.item())
            best_loss = loss.item()
            best_concept = op.weight.detach()
    return best_concept


def ablate_concepts(activations, concepts):
    if isinstance(activations, torch.Tensor):
        assert isinstance(concepts, torch.Tensor)
        orth_space = concepts / concepts.norm(dim=1)[:, None]
        project = (
            orth_space.T
            @ torch.linalg.inv((orth_space @ orth_space.T).float()).to(orth_space.dtype)
            @ orth_space
        )
        reject = (
            torch.eye(project.shape[0], device=project.device, dtype=project.dtype)
            - project
        )
    elif isinstance(activations, np.ndarray):
        assert isinstance(concepts, np.ndarray)
        orth_space = concepts / np.linalg.norm(concepts, axis=1, keepdims=True)
        project = orth_space.T @ np.linalg.inv(orth_space @ orth_space.T) @ orth_space
        reject = np.eye(project.shape[0]) - project
    # np.testing.assert_allclose(project @ project, project)
    # np.testing.assert_allclose(project.T, project)

    return activations @ reject


def proj_concepts(activations, concepts):
    orth_space = concepts
    project = orth_space.T @ np.linalg.inv(orth_space @ orth_space.T) @ orth_space
    return activations @ project.T


def concept_learning_test3(activations, init="kmeans", clustering="kmeans"):
    """Learn concepts by a simple hierarchical EM algorithm"""
    if init == "kmeans":
        kmeans = KMeans(n_clusters=20)
        kmeans.fit(activations)
        concepts = kmeans.centroids

        subconcepts1 = concepts[:10]
        subconcepts2 = concepts[10:]
        concepts1 = np.mean(subconcepts1, axis=0, keepdims=True)
        concepts2 = np.mean(subconcepts2, axis=0, keepdims=True)
        concepts1 = concepts1 / np.linalg.norm(concepts1)
        concepts2 = concepts2 / np.linalg.norm(concepts2)
        concepts2 = concepts1 - (concepts1 @ concepts2.T @ concepts2)
    elif init == "pca":
        concepts = PCA(n_components=4).fit(activations).components_
        subconcepts1 = concepts[:2]
        subconcepts2 = concepts[:2]
        concepts1 = np.mean(subconcepts1, axis=0, keepdims=True)
        concepts2 = np.mean(subconcepts2, axis=0, keepdims=True)
        concepts1 = concepts1 / np.linalg.norm(concepts1)
        concepts2 = concepts2 / np.linalg.norm(concepts2)
    elif init == "random":
        subconcepts1 = np.random.randn(2, activations.shape[1])
        subconcepts2 = np.random.randn(2, activations.shape[1])
        concepts1 = np.mean(subconcepts1, axis=0, keepdims=True)
        concepts2 = np.mean(subconcepts2, axis=0, keepdims=True)
        concepts1 = concepts1 / np.linalg.norm(concepts1)
        concepts2 = concepts2 / np.linalg.norm(concepts2)
    else:
        raise ValueError("init must be either 'kmeans' or 'pca' or 'random'.")

    if clustering == "kmeans":
        cluster = lambda x: KMeans(n_clusters=10, n_init="auto").fit(x).centroids
    elif clustering == "pca":
        cluster = lambda x: PCA(n_components=10).fit(x).components_
    else:
        raise ValueError("clustering must be either 'kmeans' or 'pca'.")

    for i in range(100):
        activations1 = ablate_concepts(activations, subconcepts2)
        subconcepts1 = cluster(activations1)
        subconcepts1 = subconcepts1 / np.linalg.norm(
            subconcepts1, axis=1, keepdims=True
        )

        activations2 = ablate_concepts(activations, subconcepts1)
        subconcepts2 = cluster(activations2)
        subconcepts2 = subconcepts2 / np.linalg.norm(
            subconcepts2, axis=1, keepdims=True
        )

        concepts1_n = np.mean(subconcepts1, axis=0, keepdims=True)
        concepts2_n = np.mean(subconcepts2, axis=0, keepdims=True)
        concepts1_n = concepts1_n / np.linalg.norm(concepts1_n)
        concepts2_n = concepts2_n / np.linalg.norm(concepts2_n)

        print(
            np.linalg.norm(concepts1 - concepts1_n),
            np.linalg.norm(concepts2 - concepts2_n),
        )
        concepts1 = concepts1_n
        concepts2 = concepts2_n

    return concepts1, concepts2, subconcepts1, subconcepts2


def concept_learning_test4(activations):
    child1_space = torch.randn(2, activations.shape[1]).double()
    child1_space = child1_space / child1_space.norm(dim=1)[:, None]

    child2_space = torch.randn(2, activations.shape[1]).double()
    child2_space = child2_space / child2_space.norm(dim=1)[:, None]

    for i in tqdm(range(10)):
        project = proj_concepts(activations, child1_space)
        kmeans = KMeans(n_clusters=30)
        kmeans.fit(project)
        concepts_child1 = kmeans.centroids
        new_child2_space = learn_split_concept(torch.tensor(concepts_child1))
        new_child2_space = new_child2_space / new_child2_space.norm(dim=1)[:, None]

        project = proj_concepts(activations, child2_space)
        kmeans = KMeans(n_clusters=30)
        kmeans.fit(project)
        concepts_child2 = kmeans.centroids
        new_child1_space = learn_split_concept(torch.tensor(concepts_child2))
        new_child1_space = new_child1_space / new_child1_space.norm(dim=1)[:, None]

        print(
            np.linalg.norm(child1_space - new_child1_space),
            np.linalg.norm(child2_space - new_child2_space),
        )
        child1_space = new_child1_space
        child2_space = new_child2_space
    return child1_space, child2_space


def cartesian_kmeans(activations):
    n = activations.shape[1]
    p = activations.shape[0]
    m = 10

    mu = np.mean(activations, axis=1, keepdims=True)
    X_mu = activations - mu

    # initialize R with random rotation of m dim
    C = np.cov(X_mu.T, rowvar=False)
    R = np.linalg.svd(np.random.randn(m, m))[0]
    if p > m:
        pc = np.linalg.eig(C)[1][:, :m]
        R = pc @ R  # p x m

    print(R.shape)

    for i in range(1000):
        RX = R.T @ X_mu
        B = np.sign(RX)
        d = np.mean(np.abs(RX), axis=1, keepdims=True)
        DB = d * B

        U, S, V = np.linalg.svd(X_mu @ DB.T, full_matrices=False)
        R = U @ V.T

        RDB = R @ DB
        mu = np.mean(activations - RDB, axis=1, keepdims=True)
        X_mu = activations - mu

        print(np.linalg.norm(activations - mu - RDB))

    return d, B, R, mu


def normalize(x):
    if type(x) == torch.Tensor:
        return x / (x.norm(dim=1)[:, None] + 1e-4)
    else:
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-4)


""" Mask to bounding boxes """


def masks_to_bboxes(masks):
    all_bboxes = []

    widths = []
    heights = []
    for img_mask in masks:
        bboxes = []
        props = regionprops(img_mask)
        for prop in props:
            x1 = prop.bbox[1]
            y1 = prop.bbox[0]

            x2 = prop.bbox[3]
            y2 = prop.bbox[2]

            bboxes.append([x1, y1, x2, y2])
            widths.append(x2 - x1)
            heights.append(y2 - y1)
        all_bboxes.append(bboxes)

    # Remove outlier bboxes
    # all_boxes_standard = []
    # max_width = np.quantile(widths, 0.9)
    # max_height = np.quantile(heights, 0.9)
    # min_width = np.quantile(widths, 0.1)
    # min_height = np.quantile(heights, 0.1)
    # for bboxes in all_bboxes:
    #     boxes_standard = []
    #     for bbox in bboxes:
    #         x1, y1, x2, y2 = bbox
    #         if x2 - x1 > min_width and y2 - y1 > min_height and x2 - x1 < max_width and y2 - y1 < max_height:
    #             boxes_standard.append(bbox)
    #     all_boxes_standard.append(boxes_standard)

    return all_bboxes


def get_sam_segments(images):
    generator = pipeline("mask-generation", model="facebook/sam-vit-large", device=0)
    all_labels = []
    for out in tqdm(generator(ListDataset(images), points_per_batch=32)):
        labels = np.zeros(out["masks"][0].shape)
        total_labels = -1
        for mask in out["masks"]:
            mask = mask.astype(int)
            cur_label = label(1 + mask) + total_labels
            total_labels = cur_label.max() - 1
            labels[mask == 1] = cur_label[mask == 1]
        all_labels.append(labels.astype(int))
    return all_labels


def get_slic_segments(images, n_segments=32):
    all_labels = []
    for image in tqdm(images):
        segments = slic(
            np.array(image),
            n_segments=n_segments,
            compactness=10,
            sigma=1,
            start_label=1,
        )
        all_labels.append(segments)
    return all_labels


def get_patches_from_bboxes(images, all_bboxes, sub_bboxes=None, image_size=(224, 224)):
    if sub_bboxes == None:
        sub_bboxes = [[[bbox] for bbox in img_bboxes] for img_bboxes in all_bboxes]

    all_patches = []
    img_labels = []
    for i, (bboxes, visible, image) in enumerate(zip(all_bboxes, sub_bboxes, images)):
        patches = []
        for bbox, viz in zip(bboxes, visible):
            curr_patch = PIL.Image.new(
                "RGB", image.size
            )  # image.copy().filter(ImageFilter.GaussianBlur(radius=10))
            for viz_box in viz:
                # add the visible patches from the original image to curr_patch
                curr_patch.paste(image.copy().crop(viz_box), box=viz_box)
            curr_patch = PIL.ImageOps.pad(curr_patch.crop(bbox), image_size)
            patches.append(curr_patch)
            img_labels.append(i)
        all_patches += patches
    return all_patches, img_labels


class ListDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return len(self.images)


class Concepts:
    def __init__(self, patches, patch_activations, concepts):
        self.patches = patches
        self.patch_activations = patch_activations
        self.concepts = concepts

    def __getitem__(self, index):
        if type(self.patches) == list:
            top = np.argsort(-self.patch_activations[index] @ self.concepts[index])
            bot = np.argsort(self.patch_activations[index] @ self.concepts[index])
            return (
                self.patches[index][top],
                self.patches[index][bot],
                self.concepts[index],
            )

        top = np.argsort(-self.patch_activations @ self.concepts[index])
        bot = np.argsort(self.patch_activations @ self.concepts[index])
        return self.patches[top], self.patches[bot], self.concepts[index]


def merge_boxes(boxes, x_val, y_val):
    size = len(boxes)
    if size < 2:
        return boxes

    if size == 2:
        if boxes_mergeable(boxes[0], boxes[1], x_val, y_val):
            boxes[0] = union(boxes[0], boxes[1])
            del boxes[1]
        return boxes

    boxes_sort = sorted(boxes, key=lambda x: x[0])
    merged = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes_mergeable(boxes_sort[i], boxes_sort[j], x_val, y_val):
                merged.append(((i, j), union(boxes_sort[i], boxes_sort[j])))
            else:
                break
    return merged


def boxes_mergeable(box1, box2, x_val, y_val):
    (x1, y1, x2, y2) = box1
    (x3, y3, x4, y4) = box2
    w1 = x2 - x1
    h1 = y2 - y1
    w2 = x4 - x3
    h2 = y4 - y3
    return (
        max(x1, x3) - min(x1, x3) - minx_w(x1, w1, x3, w2) < x_val
        and max(y1, y3) - min(y1, y3) - miny_h(y1, h1, y3, h2) < y_val
    )


def minx_w(x1, w1, x2, w2):
    return w1 if x1 <= x2 else w2


def miny_h(y1, h1, y2, h2):
    return h1 if y1 <= y2 else h2


def union(a, b):
    assert a[0] <= a[2] and a[1] <= a[3] and b[0] <= b[2] and b[1] <= b[3]
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    x2 = max(a[2], b[2])
    y2 = max(a[3], b[3])
    return x, y, x2, y2


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def get_patches_from_bboxes2(bboxes_for_img, images) -> list[list[Patch]]:
    patches_for_imgs = []
    for i, bboxes in enumerate(bboxes_for_img):
        patches = []
        for bbox in bboxes:
            # TODO: modify a Patch to only contain a reference to the image and
            # a bbox. Then, we will construct the patch when getting the
            # embedding.
            patches.append(Patch(images[i], bbox))
        patches_for_imgs.append(patches)
    return patches_for_imgs


def compose_patches(patches: list[Patch], compose_method="img") -> Patch:
    assert all([patches[0].image == patch.image for patch in patches])

    if compose_method == "img":
        merged_patch = PIL.Image.new("RGB", patches[0].image.size)
        union_box = patches[0].bbox
        for patch in patches:
            merged_patch.paste(patch.patch, box=patch.bbox)
            union_box = union(union_box, patch.bbox)
        merged_patch = merged_patch.crop(union_box)
    elif callable(compose_method):
        return compose_method(patches)
    else:
        raise NotImplementedError

    return Patch(patches[0].image, union_box, merged_patch)


def compose_all_patches(
    patches: list[list[list[Patch]]], compose_method="img"
) -> list[list[Patch]]:
    return [
        [compose_patches(patches, compose_method) for patches in img_patches]
        for img_patches in patches
    ]


class ConceptLearner:
    def __init__(
        self,
        samples: list[PIL.Image],
        input_to_latent,
        input_processor,
        device: str = "cpu",
        batch_size: int = 128,
    ):
        self.samples = samples
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.input_to_latent = input_to_latent
        self.image_size = 224 if type(samples[0]) == PIL.Image.Image else None
        self.input_processor = input_processor

    def patches(
        self, images=None, patch_method="sam", output_dir="output/"
    ) -> list[list[Patch]]:
        if images is None:
            images = self.samples

        samples_hash = utils.hashfn(images)
        if os.path.exists(
            f"{output_dir}/saved_patches_{patch_method}_{samples_hash}.pkl"
        ):
            print("Loading cached patches")
            print(f"{output_dir}/saved_patches_{patch_method}_{samples_hash}.pkl")
            patches = utils.load(
                f"{output_dir}/saved_patches_{patch_method}_{samples_hash}.pkl"
            )
            return patches

        if patch_method == "sam":
            masks = get_sam_segments(images)
            bboxes_for_imgs = masks_to_bboxes(masks)
            patches_for_imgs = get_patches_from_bboxes2(bboxes_for_imgs, images)
        elif patch_method == "slic":
            masks = get_slic_segments(images, n_segments=8 * 8)
            bboxes_for_imgs = masks_to_bboxes(masks)
            patches_for_imgs = get_patches_from_bboxes2(bboxes_for_imgs, images)
        elif patch_method == "none":
            patches_for_imgs = [
                [Patch(img, (0, 0, img.size[0], img.size[1]))] for img in images
            ]
        elif callable(patch_method):
            patches_for_imgs = patch_method(images)
        else:
            raise ValueError(f"Patch method {patch_method} not supported.")

        utils.save(
            patches_for_imgs,
            f"{output_dir}/saved_patches_{patch_method}_{samples_hash}.pkl",
        )
        return patches_for_imgs

    def get_patches(
        self, n_patches, images=None, method="slic", merge=False, output_dir="output/"
    ):
        """Get patches from images using different segmentation methods."""
        # if images is None:
        #     images = self.samples

        # samples_hash = utils.hashfn(images)
        # if os.path.exists(f"output/saved_patches_{method}_{n_patches}_{samples_hash}{'_merge' if merge else ''}.pkl"):
        #     print("Loading cached patches")
        #     print(samples_hash)
        #     patches, patch_activations, bboxes, img_for_patch = utils.load(f"output/saved_patches_{method}_{n_patches}_{samples_hash}{'_merge' if merge else ''}.pkl")
        #     return patches, patch_activations, bboxes, img_for_patch

        # if method == "slic":
        #     masks = get_slic_segments(images, n_segments=n_patches * n_patches)
        #     bboxes = masks_to_bboxes(masks)
        #     patches, img_for_patch = get_patches_from_bboxes(images, bboxes)
        #     patches = self.input_processor(patches)
        # elif method == "sam":
        #     masks = get_sam_segments(images)
        #     bboxes = masks_to_bboxes(masks)
        #     # Merge close boxes to create relation patches
        #     if merge:
        #         bboxes = [merge_boxes(boxes, 8, 8) for boxes in bboxes]
        #     patches, img_for_patch = get_patches_from_bboxes(images, bboxes)
        #     patches = self.input_processor(patches)
        # elif method == "window":
        #     patch_size = int(self.image_size // n_patches)
        #     strides = int(patch_size)
        #     samples = self.input_processor(images)
        #     patches = torch.nn.functional.unfold(samples, kernel_size=patch_size, stride=strides)
        #     patches = patches.transpose(1, 2).contiguous().view(-1, 3, patch_size, patch_size)
        #     # TODO: add the bbox definition
        #     bboxes = None
        #     img_for_patch = None
        # elif callable(method):
        #     patches = method(images, n_patches)
        #     patches = self.input_processor(patches)
        #     # TODO: add the bbox definition
        #     bboxes = None
        #     img_for_patch = None
        # else:
        #     raise ValueError("method must be either 'slic' or 'sam' or 'window'.")

        # print(len(patches))
        # patch_activations = utils._batch_inference(self.input_to_latent, patches, self.batch_size, self.image_size,
        #     device=self.device)

        # # cache the result
        # if not os.path.exists(f"output/"):
        #     os.mkdir(f"output/")
        # utils.save((patches, patch_activations, bboxes, img_for_patch), f"output/saved_patches_{method}_{n_patches}_{samples_hash}{'_merge' if merge else ''}.pkl")
        patches = self.patches(images, method, output_dir=output_dir)
        patch_activations = self.get_patch_embeddings(
            flatten_list(patches), output_dir=output_dir
        )
        return patches, patch_activations

    def get_patch_subsets(
        self, patches: list[list[Patch]], k: int, method="naive"
    ) -> list[list[list[Patch]]]:
        """Get all subsets of patches of size k from the same image."""
        contains_ls = []
        subsets = []
        curr_patches = 0
        if method == "naive":
            for img_patches in patches:
                subset = list(itertools.combinations(img_patches, k))
                new_contains = list(
                    itertools.combinations(
                        range(curr_patches, curr_patches + len(img_patches)), k
                    )
                )
                final_subset = []
                final_contains = []
                dists = [
                    max(
                        [
                            np.linalg.norm(
                                np.array(patch1.bbox) - np.array(patch2.bbox)
                            )
                            for patch2 in combined
                            for patch1 in combined
                        ]
                    )
                    for combined in subset
                ]
                threshold = np.quantile(dists, 0.3)
                if k == 1:
                    final_subset = subset
                    final_contains = new_contains
                for i in range(len(subset)):
                    if dists[i] < threshold:
                        final_subset.append(subset[i])
                        final_contains.append(new_contains[i])

                subsets.append(final_subset)
                # subsets.append(list(itertools.combinations(img_patches, k)))
                # new_contains = list(itertools.combinations(range(curr_patches, curr_patches+len(img_patches)), k))
                curr_patches += len(img_patches)
                contains_ls.append(final_contains)

            num_new = sum([len(subset) for subset in subsets])
            contains = torch.zeros((curr_patches, num_new))
            i = 0
            for img_contains in contains_ls:
                for pids in img_contains:
                    for pid in pids:
                        contains[pid, i] = 1
                    i += 1
        elif method == "vtop":
            all_patches = torch.stack(
                [
                    patch.patch
                    for patches_for_img in patches
                    for patch in patches_for_img
                ]
            )
            processed_patches = self.input_processor(all_patches)
            patch_activations = utils._batch_inference(
                self.input_to_latent,
                processed_patches,
                self.batch_size,
                self.image_size,
                device=self.device,
            )

            kmeans = KMeans(n_clusters=100, init="k-means++", n_init="auto").fit(
                patch_activations
            )
            word_assignment = kmeans.labels_
            documents = []
            total_len = 0
            for img_patches in patches:
                documents.append(
                    [word_assignment[i] for i in range(total_len, len(img_patches))]
                )
                total_len += len(img_patches)

            # Run LDA on the documents

            # Get patch subsets from the topics
        else:
            raise ValueError("method must be either 'naive' or 'vtop'.")

        return subsets, contains

    def merge_patches(self, bboxes, patch_to_img):
        bboxes = [box for boxes in bboxes for box in boxes]
        merged_patches = []
        sub_patches = []
        img_merge = []
        img_sub = []
        for i in range(len(bboxes)):
            if i > 0 and patch_to_img[i] != patch_to_img[i - 1]:
                merged_patches.append(img_merge)
                sub_patches.append(img_sub)
                img_merge = []
                img_sub = []
            for j in range(i + 1, len(bboxes)):
                if patch_to_img[i] == patch_to_img[j] and boxes_mergeable(
                    bboxes[i], bboxes[j], 8, 8
                ):
                    img_merge.append(union(bboxes[i], bboxes[j]))
                    img_sub.append([bboxes[i], bboxes[j]])
                elif patch_to_img[i] != patch_to_img[j]:
                    break

        merged_patches = get_patches_from_bboxes(
            self.samples, merged_patches, sub_patches
        )[0]
        # subset = np.random.choice(len(merged_patches), 10000, replace=False)
        # merged_patches = np.array(merged_patches)[subset]
        print("Merged patches length:", len(merged_patches))
        merged_patches = self.input_processor(merged_patches)
        merged_activations = normalize(
            utils._batch_inference(
                self.input_to_latent,
                merged_patches,
                self.batch_size,
                self.image_size,
                device=self.device,
            )
        )
        return merged_patches, merged_activations

    def get_patches_containing(self, concept1, concept2):
        patches, activations, bboxes, img_for_patch = self.get_patches(8, method="sam")
        # flatten bboxes
        bboxes = [box for boxes in bboxes for box in boxes]
        concepts = np.stack([concept1, concept2], axis=0)
        patch_concepts = (activations @ concepts.T) > torch.quantile(
            activations @ concepts.T, 0.9, dim=0
        )

        # merged_bboxes is the list of unioned bboxes for each image
        # sub_bboxes is the list of bboxes that were unioned for each image
        merged_bboxes = []
        sub_bboxes = []
        prev_i = 0
        img_bboxes_merge = []
        img_sub_bboxes = []
        for i in range(len(patches)):
            if img_for_patch[i] != img_for_patch[prev_i]:
                merged_bboxes.append(img_bboxes_merge)
                sub_bboxes.append(img_sub_bboxes)
                img_bboxes_merge = []
                img_sub_bboxes = []
            for j in range(i + 1, len(patches)):
                if (
                    img_for_patch[i] == img_for_patch[j]
                    and patch_concepts[i, 0]
                    and patch_concepts[j, 1]
                ):
                    img_bboxes_merge.append(union(bboxes[i], bboxes[j]))
                    img_sub_bboxes.append((bboxes[i], bboxes[j]))
                elif img_for_patch[i] != img_for_patch[j]:
                    break
            prev_i = i
        merged_patches = get_patches_from_bboxes(
            self.samples, merged_bboxes, sub_bboxes
        )[0]
        merged_patches = self.input_processor(merged_patches)
        merged_activations = normalize(
            utils._batch_inference(
                self.input_to_latent,
                merged_patches,
                self.batch_size,
                self.image_size,
                device=self.device,
            )
        )
        return merged_patches, merged_activations

    def learn_ace_concepts(self, n_concepts, patch_activations, use_svm=False, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        labels, centroids = utils.kmeans_clustering(
            patch_activations, n_concepts, cosine=False
        )
        concepts = centroids
        if use_svm:
            concepts = self.learn_supervised_concepts(
                patch_activations,
                OneHotEncoder(sparse_output=False).fit_transform(labels.reshape(-1, 1)),
            )
        return normalize(concepts)

    def learn_pca_concepts(self, n_concepts, patch_activations, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        reducer = PCA(n_components=n_concepts).fit(
            utils.torch_to_numpy(patch_activations)
        )
        W = torch.tensor(reducer.components_).float()
        return normalize(W)

    def learn_kmeans_concepts(self, n_concepts, patch_activations, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        cluster = KMeans(n_clusters=n_concepts, init="k-means++", n_init="auto")
        cluster.fit(patch_activations)
        # concepts = []
        # for cid in np.unique(cluster.labels_):
        #     svm = LinearSVC(C=1, fit_intercept=False, dual='auto').fit(patch_activations, cluster.labels_ == cid)
        #     concepts.append(normalize(svm.coef_))
        return normalize(cluster.centroids)

    def learn_dictlearn_concepts(self, n_concepts, patch_activations, seed=0):
        dict_learn = MiniBatchDictionaryLearning(
            n_components=n_concepts,
            alpha=1,
            n_iter=1000,
            batch_size=512,
            positive_code=True,
            fit_algorithm="cd",
            random_state=seed,
        )
        dict_learn.fit(patch_activations)
        return torch.tensor(normalize(dict_learn.components_)).float()

    def learn_supervised_concepts(self, patch_activations, concept_labels):
        concepts = []
        for cid in range(concept_labels.shape[1]):
            cl = concept_labels[:, cid]
            svm = LinearSVC(C=1, fit_intercept=False, dual="auto").fit(
                patch_activations, cl
            )
            concepts.append(normalize(svm.coef_))
        return np.concatenate(concepts)

    def learn_seminmf_concepts(self, n_concepts, patch_activations, seed=0):
        W, H = learn_seminmf_2(patch_activations.T, n_concepts, seed=seed)
        return normalize(W.T)

    def learn_concepts(
        self, n_concepts, samples=None, concept_method="kmeans", patch_method="slic"
    ):
        concepts = []
        patches = self.patches(samples, patch_method=patch_method)
        patch_emb = self.get_patch_embeddings(flatten_list(patches))

        if concept_method == "kmeans":
            concepts = self.learn_kmeans_concepts(n_concepts, patch_emb)
        elif concept_method == "pca":
            concepts = self.learn_pca_concepts(n_concepts, patch_emb)
        elif concept_method == "dictlearn":
            concepts = self.learn_dictlearn_concepts(n_concepts, patch_emb)
        elif concept_method == "ours-subspace":
            concepts, _ = self.learn_attribute_concepts(
                n_concepts, patch_emb, split_method="ours-subspace"
            )
        elif concept_method == "ours":
            concepts, _ = self.learn_attribute_concepts(
                n_concepts, patch_emb, split_method="ours"
            )

        return concepts, patches, patch_emb

    def learn_attribute_concepts(
        self,
        n_attributes,
        patch_activations: torch.Tensor,
        n_concepts=-1,
        subspace_dim=50,
        split_method="ours",
        seed=0,
        cross_entropy=False,
        no_reg=False,
    ) -> tuple[torch.Tensor, list]:
        """Learn concepts given patch embeddings"""

        torch.manual_seed(seed)
        np.random.seed(seed)
        concepts = []
        subspaces = []
        activations_orig = patch_activations.clone()
        activations = activations_orig.clone()
        all_labels = []
        rels = []
        if n_concepts == -1:
            n_concepts = [-1] * n_attributes
        for cid in range(n_attributes):
            if split_method == "ours":
                sc = normalize(
                    learn_split_concept(activations, 1, lr=0.1, device=self.device)
                )
                labels = (activations @ sc.T) >= torch.quantile(activations @ sc.T, 0.9)
                all_labels.append(labels)
            elif split_method == "ours-subspace":
                subspace, centroids = learn_split_subspace(
                    activations,
                    n_components=subspace_dim,
                    n_clusters=n_concepts[cid],
                    lr=0.001,
                    cosine=True,
                    device=self.device,
                    cross_entropy=cross_entropy,
                    no_reg=no_reg,
                )
                sc = centroids
                subspaces.append(subspace)
            elif split_method == "nmf":
                W, H = learn_seminmf_2(activations.T, n_concepts)
                return normalize(W.T), []
            elif split_method == "kmeans":
                labels, centroids = utils.kmeans_clustering(activations, 2, cosine=True)
                sc = normalize(centroids)
            elif split_method == "reg-kmeans":
                kmeans = KMeans(n_clusters=n_concepts, init="k-means++", n_init="auto")
                kmeans.fit(activations)
                concepts = kmeans.centroids
                return normalize(concepts), []
            elif split_method == "itersvr":
                labels, sc = iterSVR(activations)
                labels = torch.tensor(labels)[:, None]
                all_labels.append(labels)
                sc = normalize(sc[np.newaxis, :])
            elif split_method == "dictlearn":
                concepts = self.learn_dictlearn_concepts(n_concepts, activations)
                return concepts
            elif split_method == "pca":
                concepts = torch.tensor(
                    self.learn_pca_concepts(n_concepts, activations)
                ).float()
                return concepts, []

            concepts.append(sc)
            activations = ablate_concepts(activations_orig, torch.cat(subspaces))

        return torch.cat(concepts)

    def learn_attribute_concepts_rec(
        self, n_concepts, patch_activations, split_method="ours"
    ):
        concepts = []
        activations_orig = patch_activations.clone()
        activations = activations_orig.clone()
        if n_concepts > 0 and len(patch_activations) > 10:
            if split_method == "ours":
                sc = normalize(
                    learn_split_concept(
                        torch.tensor(activations).float(), 1, lr=0.005
                    ).numpy()
                )
                labels = (activations @ sc.T) > 0
            elif split_method == "kmeans":
                kmeans = KMeans(n_clusters=2, init="k-means++", n_init="auto")
                kmeans.fit(activations)
                labels = kmeans.predict(activations).flatten()
                svm = LinearSVC(C=0.1, fit_intercept=False, dual="auto").fit(
                    activations, labels
                )
                sc = normalize(svm.coef_)

            # left split
            left_concepts = self.learn_attribute_concepts_rec(
                n_concepts // 2,
                activations[(activations @ sc[0]) <= 0],
                split_method=split_method,
            )
            right_concepts = self.learn_attribute_concepts_rec(
                n_concepts // 2,
                activations[(activations @ sc[0]) > 0],
                split_method=split_method,
            )
            if left_concepts is None and right_concepts is None:
                return sc
            elif left_concepts is None:
                return np.concatenate([sc, right_concepts])
            elif right_concepts is None:
                return np.concatenate([sc, left_concepts])
            else:
                return np.concatenate([sc, left_concepts, right_concepts])
        else:
            return None

    def learn_sup_attribute_concepts(self, concept_labels, patch_activations):
        concepts = []
        activations_orig = patch_activations.clone()
        activations = activations_orig.clone()
        concepts = learn_sup_split_concept(
            torch.tensor(activations).float(), concept_labels, lr=0.001
        ).numpy()
        return concepts

    def get_patch_embeddings(
        self, patches: list[Patch], output_dir="output/"
    ) -> torch.Tensor:
        # scaled_patches = [PIL.ImageOps.pad(patch.patch, (self.image_size, self.image_size)) for patch in patches]
        if type(patches[0]) == Patch:
            scaled_patches = [patch.bbox for patch in patches]
        else:
            scaled_patches = patches
        patch_hash = utils.hashfn(scaled_patches)
        if os.path.exists(f"{output_dir}/saved_patch_embeddings_{patch_hash}.pkl"):
            print("Loading cached patch embeddings")
            print(f"{output_dir}/saved_patch_embeddings_{patch_hash}.pkl")
            patch_activations = utils.load(
                f"{output_dir}/saved_patch_embeddings_{patch_hash}.pkl"
            )
            return patch_activations

        # processed_patches = self.input_processor(scaled_patches)
        print("before inference")
        patch_activations = utils._batch_patch_emb(
            self.input_to_latent,
            patches,
            self.batch_size,
            self.image_size,
            processor=self.input_processor,
            device=self.device,
        )
        utils.save(
            patch_activations, f"{output_dir}/saved_patch_embeddings_{patch_hash}.pkl"
        )
        return patch_activations

    def parse_grammar(self, concepts_p, patches2img, contains):
        img_parses = []
        for img_id in torch.unique(patches2img[0]).tolist():
            grammar_concepts = []
            concepts_img = [
                concepts_p[i][patches2img[i] == img_id] for i in range(len(concepts_p))
            ]
            contains_img = [torch.zeros(1)]
            for i in range(len(contains)):
                cont = contains[i].T[patches2img[i + 1] == img_id]
                cont = cont[:, patches2img[i] == img_id]
                contains_img.append(cont)

            # iterate over each level in the grammar
            for i in range(len(concepts_img)):
                # iterate over each patch of this image in this level
                print(contains_img[i].shape)
                for j in range(len(concepts_img[i])):
                    if torch.sum(concepts_img[i][j]) > 0:
                        contained_patches = []
                        if i > 0 and torch.sum(contains_img[i][j]) > 0:
                            contained_patches = (
                                torch.nonzero(contains_img[i][j]).flatten().tolist()
                            )
                        grammar_concepts.append(
                            (
                                contained_patches,
                                torch.nonzero(concepts_img[i][j]).flatten().tolist(),
                            )
                        )
            img_parses.append(grammar_concepts)
            break
        return img_parses

    def unique_patches(self, embeddings):
        # find the embeddings that are not similar to many other embeddings
        sim = (embeddings @ embeddings.T) / (
            torch.norm(embeddings, dim=1, keepdim=True) ** 2
        )
        sim = sim > 0.9
        unique = torch.sum(sim, dim=1) < len(embeddings) / 3
        return torch.nonzero(unique).flatten()

    def learn_grammar(
        self,
        T,
        N,
        split_method="ours",
        patch_method="window",
        compose_method="img",
        augment_concepts=False,
        center=False,
        normalize=False,
        whiten=False,
    ):
        P0 = self.patches(patch_method=patch_method)
        concepts = []
        concepts_v = torch.tensor([])
        concepts_p = []
        contains = []
        patches = []
        patches2img = []
        embeddings = []
        kept_patches = []

        for i in range(1, T + 1):
            P, contains_i = self.get_patch_subsets(P0, i, method="naive")
            if i == 1:
                assert torch.all(contains_i == torch.eye(sum([len(P0i) for P0i in P0])))

            CP = compose_all_patches(P, compose_method)
            patches.append([patch.patch for patch in flatten_list(CP)])
            patch2img = torch.zeros(len(patches[-1]))
            img_cnt = 0
            for img_id, img_patches in enumerate(CP):
                patch2img[img_cnt : img_cnt + len(img_patches)] = img_id
                img_cnt += len(img_patches)
            patches2img.append(patch2img)

            # contains is an (n_patches_i, n_patches_ip1) matrix where index (i, j) is 1 if patch j contains patch i.
            if i > 1:
                contains.append(contains_i)

            # E = normalize(self.get_patch_embeddings(flatten_list(CP)))
            # E = utils.whiten(self.get_patch_embeddings(flatten_list(CP)))
            E = self.get_patch_embeddings(flatten_list(CP))
            if whiten:
                E, mean, whiten_m, dewhiten_m = utils.whiten(E, return_fit=True)
            else:
                if center:
                    E = E - torch.mean(E, dim=0, keepdim=True)
                if normalize:
                    E = utils.normalize(E)

            keep_patches = self.unique_patches(E)
            print(len(keep_patches))
            E = E[keep_patches]
            patches[-1] = [patches[-1][i] for i in keep_patches]
            patches2img[-1] = patches2img[-1][keep_patches]
            if i > 1:
                contains[-1] = contains[-1][:, keep_patches]
                contains[-1] = contains[-1][kept_patches[-1], :]
            kept_patches.append(keep_patches)
            if whiten:
                embeddings.append(E @ dewhiten_m)
            else:
                embeddings.append(E)

            # learn N new attribute concepts
            if len(concepts_v) > 0:
                # Augment all previous concepts to new patches
                if augment_concepts:
                    for j in range(len(concepts_v)):
                        this_iter_contains = concepts_p[j].T
                        for k in range(j, len(concepts_v)):
                            this_iter_contains = this_iter_contains @ contains[k]
                        augmented_concepts = this_iter_contains.T @ E
                        concepts_v[j] = utils.normalize(
                            (augmented_concepts + concepts_v[j]) / 2
                        )

                # Remove existing concept subspaces from the patch embeddings
                if whiten:
                    E = ablate_concepts(E, concepts_v @ dewhiten_m.T)
                else:
                    E = ablate_concepts(E, concepts_v)

            # new_concepts: (n_concepts, embedding_dim) matrix where row i is the ith concept.
            new_concepts, _ = self.learn_attribute_concepts(
                N, E, split_method=split_method
            )
            if whiten:
                new_concepts = new_concepts @ whiten_m

            # new_concepts_p: (n_patches, n_concepts) matrix where index (i, j) is 1 if patch i contains concept j.
            if whiten:
                new_concepts_p = (E @ dewhiten_m @ new_concepts.T) > torch.quantile(
                    E @ dewhiten_m @ new_concepts.T, 0.7, dim=0
                )
            else:
                new_concepts_p = torch.quantile(E @ new_concepts.T, 0.7, dim=0) < (
                    E @ new_concepts.T
                )
            concepts_p.append(new_concepts_p)
            concepts_v = torch.cat([concepts_v, new_concepts], dim=0)
            concepts.append(new_concepts)
        return (
            concepts_v,
            concepts_p,
            concepts,
            patches,
            patches2img,
            contains,
            embeddings,
        )

    def find_relation_concepts_emb(
        self,
        n_attr_concepts,
        small_patch_size=8,
        large_patch_size=4,
        patch_type="window",
        split_method="ours",
        use_leace=False,
    ):
        """Learn the terminal concepts of the concept grammar"""

        attr_patches, attr_activations, attr_bboxes, patch_to_img = self.get_patches(
            small_patch_size, method=patch_type
        )
        attr_concepts, a_rels = self.learn_attribute_concepts(
            n_attr_concepts,
            attr_activations,
            split_method=split_method,
            use_leace=use_leace,
        )

        merged_patches, merged_activations = self.merge_patches(
            attr_bboxes, patch_to_img
        )

        # Remove subspace of previous patches
        pca = PCA(n_components=0.9).fit(utils.torch_to_numpy(attr_activations))
        rel_activations_ablate = torch.tensor(
            normalize(ablate_concepts(merged_activations, normalize(pca.components_)))
        ).float()

        # Learn concepts on the merged patches
        rel_concepts, r_rels = self.learn_attribute_concepts(
            n_attr_concepts,
            rel_activations_ablate,
            split_method=split_method,
            use_leace=use_leace,
        )

        # Get the concept activations of the attribute and relation concepts on the merged patches
        attr_act = (merged_activations @ attr_concepts.T) > torch.quantile(
            merged_activations @ attr_concepts.T, 0.7, dim=0
        )
        rel_act = (merged_activations @ rel_concepts.T) > torch.quantile(
            merged_activations @ rel_concepts.T, 0.7, dim=0
        )

        # Find the relation concepts which are highly correlated with attribute concepts
        rels = []
        for i in range(len(rel_concepts)):
            res = [
                pearsonr(attr_act[:, j], rel_act[:, i])
                for j in range(len(attr_concepts))
            ]
            sort = np.argsort(-np.abs([r[0] for r in res]))
            corrs = np.array([r[0] for r in res])[sort]
            ps = np.array([r[1] for r in res])[sort]
            ids = np.arange(len(attr_concepts))[sort]
            rels.append(
                (
                    i,
                    ids[np.logical_and(ps < 0.05, corrs > 0)],
                    ids[np.logical_and(ps < 0.05, corrs < 0)],
                )
            )

        return (
            Concepts(attr_patches, attr_activations, attr_concepts),
            Concepts(merged_patches, merged_activations, rel_concepts),
            rels,
        )

    def find_relation_concepts_img(self):
        attr_patches, attr_activations, _, _ = self.get_patches(7, method="window")
        rel_patches, rel_activations, _, _ = self.get_patches(1, method="window")
        attr_concepts, _ = self.learn_attribute_concepts(
            10, attr_activations, split_method="kmeans", use_leace=False
        )

        best_attr_examples = []
        for i in range(len(attr_concepts)):
            pos = attr_patches[torch.argmax(attr_activations @ attr_concepts[i])]
            neg = attr_patches[torch.argmin(attr_activations @ attr_concepts[i])]
            best_attr_examples.append((pos, neg))

        cf_images = []
        for i in range(len(rel_patches)):
            # patches, acts = self.get_patches(32, images=[rel_patches[i]], method="window")
            patches = attr_patches[49 * i : 49 * (i + 1)]
            acts = attr_activations[49 * i : 49 * (i + 1)]

            for i, (patch, act) in enumerate(zip(patches, acts)):
                top_concept = np.argsort(-np.abs(act @ attr_concepts.T))[0]
                # top_concept = np.argsort(-act @ attr_concepts.T)[0]
                pos = (act @ attr_concepts[top_concept]) > 0
                if pos:
                    patches[i] = best_attr_examples[top_concept][0]
                else:
                    patches[i] = best_attr_examples[top_concept][1]

            strides = 32
            print(patches.shape)
            patches = patches.contiguous().permute(1, 2, 3, 0).view(1, -1, 49)
            print(patches.shape)
            sample = torch.nn.functional.fold(
                patches, output_size=(224, 224), kernel_size=32, stride=strides
            )
            cf_images.append(sample)

        return cf_images
