from __future__ import annotations

import os
import sys

from sklearn.decomposition import NMF, PCA, MiniBatchDictionaryLearning

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from baseline_utils import learn_seminmf_2
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from utils import compute_cosine_similarity_matrix, kmeans_clustering, normalize


class unsupervised_cl_baselines:
    def __init__(self, num_concepts):
        self.num_concepts = num_concepts

    def extract_concepts(self, X):
        pass

    def compute_sample_activations(self, X):
        pass

    def return_concepts(self):
        pass


class PCA_learner(unsupervised_cl_baselines):
    def __init__(self, num_concepts, **kwargs):
        super().__init__(num_concepts)
        self.pca = PCA(n_components=num_concepts)

    def extract_concepts(self, X):
        # reducer = PCA(n_components=total_concepts).fit(input_embs)
        #         W = reducer.components_.astype(np.float32)
        #         learned_concepts = normalize(W)
        if type(X) == torch.Tensor:
            X = X.cpu().numpy()
        self.pca.fit(X)

    def compute_sample_activations(self, X):
        sample_activation = self.pca.transform(X)
        sample_activation = torch.from_numpy(sample_activation).float()
        return sample_activation

    def return_concepts(self):
        components_ = self.pca.components_.astype(np.float32)
        return normalize(components_)


class NMF_learner(unsupervised_cl_baselines):
    def __init__(self, num_concepts, **kwargs):
        super().__init__(num_concepts)
        self.nmf = NMF(n_components=num_concepts)

    def extract_concepts(self, X):
        # self.nmf.fit(X)
        self.W, self.H = learn_seminmf_2(X.T, self.num_concepts)

    def compute_sample_activations(self, X):
        # sample_activation = self.nmf.transform(X)
        sample_activation = X @ self.W
        # sample_activation = torch.from_numpy(sample_activation).float()
        return sample_activation

    def return_concepts(self):
        return normalize(self.W.T)


def learn_supervised_concepts(patch_activations, concept_labels):
    concepts = []
    for cid in range(concept_labels.shape[1]):
        cl = concept_labels[:, cid]
        svm = LinearSVC(C=1, fit_intercept=False).fit(patch_activations, cl)
        concepts.append(normalize(svm.coef_))
    return np.concatenate(concepts)


def learn_ace_concepts(n_concepts, patch_activations, use_svm=False):
    labels, centroids = kmeans_clustering(patch_activations, n_concepts, cosine=False)
    concepts = centroids
    if use_svm:
        concepts = learn_supervised_concepts(
            patch_activations,
            OneHotEncoder(sparse_output=False).fit_transform(labels.reshape(-1, 1)),
        )
    return normalize(concepts)


class ACE_learner(unsupervised_cl_baselines):
    def __init__(self, num_concepts, use_svm=False):
        super().__init__(num_concepts)
        self.svm = use_svm

    def extract_concepts(self, X):
        self.concepts = learn_ace_concepts(self.num_concepts, X, use_svm=self.svm)

    def compute_sample_activations(self, X):
        sample_activation = X @ self.concepts.T
        return sample_activation

    def return_concepts(self):
        return self.concepts


def learn_dictlearn_concepts(n_concepts, patch_activations):
    dict_learn = MiniBatchDictionaryLearning(
        n_components=n_concepts,
        alpha=1,
        n_iter=1000,
        batch_size=512,
        positive_code=True,
        fit_algorithm="cd",
    )
    dict_learn.fit(patch_activations)
    return normalize(dict_learn.components_)


class dictionary_learner(unsupervised_cl_baselines):
    def __init__(self, num_concepts, **kwargs):
        super().__init__(num_concepts)

    def extract_concepts(self, X):
        self.concepts = learn_dictlearn_concepts(self.num_concepts, X)

    def compute_sample_activations(self, X):
        sample_activation = X @ self.concepts.T
        return sample_activation

    def return_concepts(self):
        return self.concepts


class kmeans_learner(unsupervised_cl_baselines):
    def __init__(self, num_concepts, cosine=False):
        super().__init__(num_concepts)
        self.cosine = cosine

    def extract_concepts(self, X):
        # assignments, centroids
        self.assignment, self.concepts = kmeans_clustering(
            X, self.num_concepts, cosine=self.cosine
        )

    def compute_sample_activations(self, X):
        if not self.cosine:
            sample_activation = torch.sqrt(
                torch.sum(
                    (X.detach().unsqueeze(1) - self.concepts.unsqueeze(0)) ** 2, dim=-1
                )
            )
        else:
            # cluster_dist = 1 - torch.mm(project.detach(), cluster_centroids.T)/(torch.norm(project.detach(), dim=1).unsqueeze(1)*torch.norm(cluster_centroids, dim=1))
            sample_activation = 1 - compute_cosine_similarity_matrix(
                X.detach(), self.concepts
            )

        return sample_activation

    def return_concepts(self):
        return self.concepts
