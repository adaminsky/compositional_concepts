from __future__ import annotations

import os
import sys

from CUB_utils import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import random
import shutil
import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import nn
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel

from baselines import *
from compositionality_eval import compositional_f1, compositionality_eval
from concept_learning import ablate_concepts, learn_split_subspace, normalize

# from silhouette_with_cosine import Silhouette
from silhouette import Silhouette

# from concept_learning_utils import *
from utils import cosim, hashfn, preprocessing_data


def concept_gt_match(concepts, gt_concepts):
    """Return the max cosine similarity of each GT concept to a learned concept"""
    similarities = cosim(concepts, gt_concepts)
    return torch.max(similarities, dim=0)[0], torch.argmax(similarities, dim=0)


def concept_gt_match_labels(concept_scores, gt_concept_labels, allow_neg=True):
    match_idx = torch.zeros(gt_concept_labels.shape[1]).long()
    auc_match = torch.zeros(gt_concept_labels.shape[1])
    signs = torch.zeros(gt_concept_labels.shape[1])
    for cid in range(gt_concept_labels.shape[1]):
        best_comp = 0
        best_score = 0
        sign = 1
        for i in range(concept_scores.shape[1]):
            if (
                roc_auc_score(gt_concept_labels[:, cid], concept_scores[:, i])
                > best_score
            ):
                best_score = roc_auc_score(
                    gt_concept_labels[:, cid], concept_scores[:, i]
                )
                best_comp = i
                sign = 1
            if (
                allow_neg
                and roc_auc_score(gt_concept_labels[:, cid], -concept_scores[:, i])
                > best_score
            ):
                best_score = roc_auc_score(
                    gt_concept_labels[:, cid], -concept_scores[:, i]
                )
                best_comp = i
                sign = -1
        match_idx[cid] = best_comp
        auc_match[cid] = best_score
        signs[cid] = sign
    return match_idx, auc_match, signs


def create_deep_set_net_for_programs(
    num_granularities, input_size, output_size, topk=5
):
    # encoder = nn.Sequential(nn.Linear(input_size, int(output_size)))
    # # decoder = torch.nn.Linear(latent_size, ATOM_VEC_LENGTH)
    # decoder = torch.nn.Identity(output_size)
    net = AttentionClassifier(num_granularities, input_size, topk, output_size)
    # embedding_dim=768,
    # num_classes=10,
    # num_heads=2,
    # attention_dropout=0.1,
    # projection_dropout=0.1,
    # # n_unsup_concepts=10,
    # # n_concepts=10,
    # n_spatial_concepts=10,
    # net = ConceptTransformer_model(embedding_dim=input_size, num_classes=output_size, num_heads=topk)
    return net


def train_eval_classification_model_by_concepts(
    log_dir,
    sample_concept_activations,
    sample_labels,
    valid_concept_activations_per_samples,
    test_concept_activations_per_samples,
    valid_labels,
    test_labels,
    mod=None,
):
    if mod is None:
        # mod = LogisticRegression()
        # if not full_image:
        #     mod = create_deep_set_net_for_programs(len(sample_concept_activations), sample_concept_activations[0][0].shape[1], len(sample_labels.unique()), 1)

        #     if torch.cuda.is_available():
        #         mod = mod.cuda()

        #     optimizer = torch.optim.Adam(mod.parameters(), lr=0.001)

        #     criterion = torch.nn.CrossEntropyLoss()

        #     # mod.fit(sample_concept_activations.numpy(), sample_labels.numpy())
        #     mod = train_deepsets_models(log_dir, mod, optimizer, criterion, sample_concept_activations, sample_labels, valid_concept_activations_per_samples, valid_labels, epoch=epochs)
        # else:
        mod = LogisticRegression()
        mod.fit(sample_concept_activations.numpy(), sample_labels.numpy())

    # if not full_image:
    #     mod.load_state_dict(torch.load(os.path.join(log_dir, "best_model.pt")))
    #     pred_train_labels, pred_train_probs = pred_deepsets_models(mod, sample_concept_activations)
    #     pred_val_labels, pred_val_probs = pred_deepsets_models(mod, valid_concept_activations_per_samples)
    #     pred_test_labels, pred_test_probs = pred_deepsets_models(mod, test_concept_activations_per_samples)
    # else:
    pred_train_probs = mod.predict_proba(sample_concept_activations)
    pred_train_labels = mod.predict(sample_concept_activations)

    pred_val_probs = mod.predict_proba(valid_concept_activations_per_samples)
    pred_val_labels = mod.predict(valid_concept_activations_per_samples)

    pred_test_probs = mod.predict_proba(test_concept_activations_per_samples)
    pred_test_labels = mod.predict(test_concept_activations_per_samples)

    train_acc = evaluate_performance(sample_labels, pred_train_labels, pred_train_probs)
    val_acc = evaluate_performance(valid_labels, pred_val_labels, pred_val_probs)
    test_acc = evaluate_performance(test_labels, pred_test_labels, pred_test_probs)

    print("final train acc: ", train_acc)
    print("final val acc: ", val_acc)
    print("final test acc: ", test_acc)
    # return mod, pred_labels, pred_probs


class InvariantModel(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module, topk: int):
        super().__init__()
        self.phi = phi
        self.rho = rho
        self.topk = topk

    def forward(self, x):
        # compute the representation for each data point
        x_ls = []
        for k in range(len(x)):
            sub_x = self.phi(x[k])
            # coeff=0
            # if x[k].shape[0] < self.topk:
            #     sub_input = torch.cat([x[k], torch.zeros(self.topk - x[k].shape[0], x[k].shape[1]).to(x[k].device)])
            # else:
            #     sub_input = x[k]

            # if torch.cuda.is_available():
            #     sample_weight = torch.softmax(sub_input[:,-1], dim=-1).view(-1,1).cuda()
            #     topk_indices = torch.topk(sub_input[:,-1], self.topk)[1]
            #     sub_x = self.phi.forward(sub_input[topk_indices,0:-1].view(1,-1).cuda())*sample_weight
            #     coeff += torch.sum(sample_weight)
            # else:
            #     sample_weight = torch.softmax(sub_input[:,-1], dim=-1).view(-1,1)
            #     topk_indices = torch.topk(sub_input[:,-1], self.topk)[1]
            #     sub_x = self.phi.forward(sub_input[topk_indices,0:-1].view(1,-1))*sample_weight
            #     coeff += torch.sum(sample_weight)

            # sum up the representations
            # here I have assumed that x is 2D and the each row is representation of an input, so the following operation
            # will reduce the number of rows to 1, but it will keep the tensor as a 2D tensor.
            x_ls.append(torch.sum(sub_x, dim=0))
            # x_ls.append(sub_x)

        # compute the output
        out = self.rho.forward(torch.stack(x_ls))

        return out


class AttentionClassifier(torch.nn.Module):
    def __init__(self, num_granularities, d_model, num_heads, num_classes):
        super(AttentionClassifier, self).__init__()

        self.num_granularities = num_granularities
        # MultiheadAttention layer
        self.attention_ls = torch.nn.ModuleList(
            [torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)]
            * num_granularities
        )

        # Linear layer for classification
        self.classifier = torch.nn.Linear(d_model * num_granularities, num_classes)

    def forward(self, input_sequence):
        # Assuming query, key, and value are the same in this example

        pooled_output_ls = []
        for idx in range(len(input_sequence)):
            curr_pooled_output_ls = []
            for sid in range(len(input_sequence[idx])):
                input_x = input_sequence[idx][sid].unsqueeze(0).permute(1, 0, 2)
                output, attention_weights = self.attention_ls[idx](
                    query=input_x, key=input_x, value=input_x
                )

                # Take the mean across the sequence dimension
                pooled_output = output.mean(dim=0)
                curr_pooled_output_ls.append(pooled_output.view(-1))

            pooled_output_ls.append(torch.stack(curr_pooled_output_ls))

        # Classification layer
        logits = self.classifier(torch.cat(pooled_output_ls, dim=-1))

        return logits


# split_method, learned_concepts, suffix="", T=1,
def learn_concept_main(
    full_data_folder,
    full_log_dir,
    img_embeddings,
    hash_val,
    valid_img_embeddings,
    full_concept_count_ls,
    lr,
    sub_embedding_size,
    method="ours",
    cosine=False,
    epochs=200,
    existing_concept_ls=None,
    existing_cluster_dist_ls=None,
    existing_cluster_label_ls=None,
    existing_cluster_centroid_ls=None,
    seed=0,
    cross_entropy=False,
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    full_out_folder = os.path.join(full_data_folder, "output/")
    # if full_image:
    # full_concept_count_ls = [full_concept_count_ls[0]]
    # if existing_concept_ls is not None and existing_cluster_dist_ls is not None and existing_cluster_label_ls is not None and existing_cluster_centroid_ls is not None:
    #     existing_concept_ls = [existing_concept_ls]
    #     existing_cluster_dist_ls = [existing_cluster_dist_ls]
    #     existing_cluster_label_ls = [existing_cluster_label_ls]
    #     existing_cluster_centroid_ls = [existing_cluster_centroid_ls]

    # for h_idx in range(len(full_concept_count_ls)):
    concept_count_ls = full_concept_count_ls
    # n_patches = patch_count_ls[h_idx]
    # if not full_image:
    #     activations = activations_ls[h_idx]
    #     input_embs = activations#torch.cat(activations, dim=0)
    #     valid_input_embs = valid_activations_ls[h_idx]
    # else:
    input_embs = img_embeddings
    valid_input_embs = valid_img_embeddings

    input_embs = preprocessing_data(input_embs)
    valid_input_embs = preprocessing_data(valid_input_embs)
    # if existing_concept_ls is not None:
    #     for existing_concept in existing_concept_ls[h_idx]:
    #         input_embs = ablate_concepts(input_embs, existing_concept)
    #         valid_input_embs = ablate_concepts(valid_input_embs, existing_concept)

    all_concept_ls = []
    cluster_dist_ls = []
    cluster_label_ls = []
    cluster_centroid_ls = []

    if method == "ours":
        start = time.time()
        silhouette = Silhouette()
        for idx in range(len(concept_count_ls)):
            if (
                existing_concept_ls is not None
                and existing_cluster_dist_ls is not None
                and existing_cluster_label_ls is not None
                and existing_cluster_centroid_ls is not None
                and h_idx < len(existing_concept_ls)
                and idx < len(existing_concept_ls[h_idx])
            ):
                curr_attrs_concepts = existing_concept_ls[idx]
                curr_cluster_dist = existing_cluster_dist_ls[idx]
                curr_cluster_labels = existing_cluster_label_ls[idx]
                curr_cluster_centroids = existing_cluster_centroid_ls[idx]

                cluster_label_ls.append(curr_cluster_labels)
                cluster_centroid_ls.append(curr_cluster_centroids.cpu())
                all_concept_ls.append(curr_attrs_concepts.cpu())
                cluster_dist_ls.append(curr_cluster_dist.cpu())

                (
                    activations,
                    cluster_dist,
                    cluster_labels,
                    cluster_centroids,
                    (center, std),
                ) = get_cluster_center_after_projection(
                    curr_attrs_concepts,
                    input_embs,
                    concept_count_ls[idx],
                    device,
                    cosine=cosine,
                )
                valid_activations, valid_cluster_dist, valid_cluster_labels = (
                    get_cluster_dist(
                        curr_attrs_concepts,
                        valid_input_embs,
                        cluster_centroids,
                        device,
                        cosine=cosine,
                    )
                )
                s_score = silhouette.score(
                    activations.cpu(), cluster_labels.cpu(), cosine=cosine
                )
                valid_s_score = silhouette.score(
                    valid_activations.cpu(), valid_cluster_labels.cpu(), cosine=cosine
                )
                print("overall silhouette score: ", s_score)
                print("overall valid silhouette score: ", valid_s_score)
            else:
                # curr_attrs_concepts0 =torch.load("temp_concepts.pt")
                # input_embs = ablate_concepts(input_embs, curr_attrs_concepts0)
                # if idx == 2:
                #     print()
                curr_attrs_concepts, _ = learn_split_subspace(
                    input_embs,
                    n_components=sub_embedding_size,
                    n_clusters=concept_count_ls[idx],
                    lr=lr,
                    device=device,
                    cosine=cosine,
                    epochs=epochs,
                    cross_entropy=cross_entropy,
                )
                with torch.no_grad():
                    (
                        activations,
                        cluster_dist,
                        cluster_labels,
                        cluster_centroids,
                        (center, std),
                    ) = get_cluster_center_after_projection(
                        curr_attrs_concepts,
                        input_embs,
                        concept_count_ls[idx],
                        device,
                        cosine=cosine,
                    )
                    # valid_activations, valid_cluster_dist, valid_cluster_labels, valid_cluster_centroids, _ = get_cluster_center_after_projection(curr_attrs_concepts, valid_input_embs, concept_count_ls[idx], device, cosine=cosine)
                    valid_activations, valid_cluster_dist, valid_cluster_labels = (
                        get_cluster_dist(
                            curr_attrs_concepts,
                            valid_input_embs,
                            cluster_centroids,
                            device,
                            cosine=cosine,
                        )
                    )
                    cluster_dist_ls.append(cluster_dist.cpu())
                    if type(cluster_labels) is torch.Tensor:
                        cluster_labels = cluster_labels.cpu()
                    cluster_label_ls.append(cluster_labels)
                    cluster_centroid_ls.append(cluster_centroids.cpu())
                    all_concept_ls.append(curr_attrs_concepts.cpu())
                    # activations = torch.mm(input_embs.cpu(), curr_attrs_concepts.cpu().t())
                    # if cosine:
                    #     activations = normalize_and_center(activations)
                    # else:
                    #     activations = normalize_and_center(activations)
                    s_score = silhouette.score(
                        activations.cpu(), cluster_labels, cosine=cosine, fast=False
                    )
                    valid_s_score = silhouette.score(
                        valid_activations.cpu(),
                        valid_cluster_labels.cpu(),
                        cosine=cosine,
                        fast=False,
                    )
                    print("overall silhouette score: ", s_score)
                    print("overall valid silhouette score: ", valid_s_score)
            input_embs = ablate_concepts(input_embs, curr_attrs_concepts)
            valid_input_embs = ablate_concepts(valid_input_embs, curr_attrs_concepts)
        # total_concepts = sum(concept_count_ls)

        duration = time.time() - start
        save_learned_concepts(
            full_out_folder,
            all_concept_ls,
            cluster_dist_ls,
            cluster_label_ls,
            cluster_centroid_ls,
            duration,
            "ours",
            epochs,
            concept_count_ls,
            sub_embedding_size,
            hash_val,
            cosine=cosine,
            seed=seed,
            cross_entropy=cross_entropy,
        )
    else:
        total_concepts = sum(concept_count_ls)

        if method in unsupervised_method_dict:
            kwargs = {}
            if method == "ace_svm":
                kwargs = {"use_svm": True}

            method_class = unsupervised_method_dict[method](total_concepts, **kwargs)
            start = time.time()
            method_class.extract_concepts(input_embs)
            learned_concepts = method_class.return_concepts()
            duration = time.time() - start
            cluster_concept_activations = method_class.compute_sample_activations(
                input_embs
            )
        # if method == "pca":
        #     reducer = PCA(n_components=total_concepts).fit(input_embs)
        #     W = reducer.components_.astype(np.float32)
        #     learned_concepts = normalize(W)

        # if full_image:
        # cluster_concept_activations = get_concept_activations_per_sample0(input_embs, torch.from_numpy(learned_concepts))
        save_learned_concepts(
            full_out_folder,
            method_class,
            cluster_concept_activations,
            None,
            None,
            duration,
            method,
            epochs,
            total_concepts,
            sub_embedding_size,
            hash_val,
            cosine=cosine,
            seed=seed,
        )


def construct_dataset_by_activations(
    img_per_patch,
    cluster_label_ls,
    cluster_dist_ls,
    alpha=0.1,
    threshold=0.9,
    topk=None,
):
    unique_sample_ids = torch.tensor(img_per_patch).unique()
    activation_sample_ls = []
    for sample_id in tqdm(unique_sample_ids):
        sample_idx = torch.where(torch.tensor(img_per_patch) == sample_id)[0].numpy()
        sample_cluster_labels = [
            cluster_label_ls[idx][sample_idx] for idx in range(len(cluster_label_ls))
        ]
        sample_cluster_dist = [
            cluster_dist_ls[idx][sample_idx] for idx in range(len(cluster_dist_ls))
        ]

        sample_cluster_high_likelihood_boolean = torch.ones(
            len(sample_idx), dtype=torch.bool
        )
        sample_cluster_all_label_likelihood_ls = []

        for attr_idx in range(len(sample_cluster_labels)):
            curr_sample_cluster_labels = sample_cluster_labels[attr_idx]
            curr_sample_cluster_dist = sample_cluster_dist[attr_idx]
            curr_sample_cluster_dist_given_label = curr_sample_cluster_dist[
                torch.arange(len(curr_sample_cluster_labels)),
                curr_sample_cluster_labels.reshape(-1),
            ]
            # curr_sample_cluster_label_likelyhood = torch.exp(-alpha*curr_sample_cluster_dist_given_label)
            curr_sample_cluster_all_label_likelyhood = (
                curr_sample_cluster_dist  # torch.exp(-alpha*curr_sample_cluster_dist)
            )
            # if  topk is None:
            #     sample_cluster_high_likelihood_boolean = torch.logical_and(sample_cluster_high_likelihood_boolean, curr_sample_cluster_label_likelyhood > threshold)
            sample_cluster_all_label_likelihood_ls.append(
                curr_sample_cluster_all_label_likelyhood
            )

        sample_cluster_label_likelihood_tensor = torch.cat(
            sample_cluster_all_label_likelihood_ls, dim=-1
        )
        if topk is None:
            sample_cluster_label_likelihood_high_likelihood = (
                sample_cluster_label_likelihood_tensor[
                    sample_cluster_high_likelihood_boolean
                ]
            )
        else:
            topk_ids = torch.topk(sample_cluster_label_likelihood_tensor, topk, dim=-1)[
                1
            ]
            sample_cluster_label_likelihood_high_likelihood = (
                sample_cluster_label_likelihood_tensor[topk_ids]
            )

        activation_sample_ls.append(sample_cluster_label_likelihood_high_likelihood)

    return activation_sample_ls


def construct_dataset_by_activations_full_image(
    cluster_label_ls, cluster_dist_ls, alpha=0.1
):
    # unique_sample_ids = torch.tensor(img_per_patch).unique()
    activation_sample_ls = []
    # for sample_id in tqdm(unique_sample_ids):
    #     sample_idx = torch.where(torch.tensor(img_per_patch) == sample_id)[0]
    #     sample_cluster_labels = [cluster_label_ls[idx][sample_idx] for idx in range(len(cluster_label_ls))]
    #     sample_cluster_dist = [cluster_dist_ls[idx][sample_idx] for idx in range(len(cluster_dist_ls))]

    #     sample_cluster_high_likelihood_boolean = torch.ones(len(sample_idx), dtype=torch.bool)
    sample_cluster_all_label_likelihood_ls = []

    for attr_idx in range(len(cluster_label_ls)):
        # curr_sample_cluster_labels = cluster_label_ls[attr_idx]
        curr_sample_cluster_dist = cluster_dist_ls[attr_idx]
        # curr_sample_cluster_dist_given_label = curr_sample_cluster_dist[torch.arange(len(curr_sample_cluster_labels)), curr_sample_cluster_labels.reshape(-1)]
        # curr_sample_cluster_label_likelyhood = torch.exp(-alpha*curr_sample_cluster_dist_given_label)
        curr_sample_cluster_all_label_likelyhood = (
            1 - curr_sample_cluster_dist
        )  # torch.exp(-alpha*curr_sample_cluster_dist)
        # if  topk is None:
        #     sample_cluster_high_likelihood_boolean = torch.logical_and(sample_cluster_high_likelihood_boolean, curr_sample_cluster_label_likelyhood > threshold)
        sample_cluster_all_label_likelihood_ls.append(
            curr_sample_cluster_all_label_likelyhood
        )

    sample_cluster_label_likelihood_tensor = torch.cat(
        sample_cluster_all_label_likelihood_ls, dim=-1
    )
    # if topk is None:
    #     sample_cluster_label_likelihood_high_likelihood = sample_cluster_label_likelihood_tensor[sample_cluster_high_likelihood_boolean]
    # else:
    #     topk_ids = torch.topk(sample_cluster_label_likelihood_tensor, topk, dim=-1)[1]
    #     sample_cluster_label_likelihood_high_likelihood = sample_cluster_label_likelihood_tensor[topk_ids]

    # activation_sample_ls.append(sample_cluster_label_likelihood_high_likelihood)

    return sample_cluster_label_likelihood_tensor


def get_cluster_dist_all(
    split_space_ls, data, cluster_centroids_ls, device, cosine=False
):
    cluster_dist_ls = []
    cluster_label_ls = []
    data = preprocessing_data(data)
    for idx in range(len(split_space_ls)):
        cluster_centroid = cluster_centroids_ls[idx]
        cluster_centroid = cluster_centroid.to(device)
        project_data, cluster_dist, cluster_label = get_cluster_dist(
            split_space_ls[idx], data, cluster_centroid, device, cosine=cosine
        )
        silhouette = Silhouette()
        s_score = silhouette.score(
            project_data.cpu(), cluster_label.cpu(), cosine=cosine
        )
        print("overall silhouette score: ", s_score)
        cluster_dist_ls.append(cluster_dist.cpu())
        cluster_label_ls.append(cluster_label.cpu().numpy())
        data = ablate_concepts(data, split_space_ls[idx])
    return cluster_dist_ls, cluster_label_ls


# def obtain_image_gt_attr_embeddings(img_embeddings, attr_labels):


def obtain_image_gt_attr_embeddings(img_embeddings, attr_labels):
    img_attr_embeddings = []
    for idx in range(attr_labels.shape[1]):
        img_attr_embeddings.append(
            torch.mean(img_embeddings[attr_labels[:, idx] == 1], dim=0)
        )
    return img_attr_embeddings


def obtain_image_gt_bg_embeddings(img_embeddings, bg_labels):
    unique_bg_labels = torch.tensor(bg_labels).unique()
    bg_embeddings_mappings = dict()
    for bg_label in unique_bg_labels:
        bg_embeddings_mappings[bg_label] = torch.mean(
            img_embeddings[bg_labels == bg_label], dim=0
        )
        # bg_embeddings.append()
    # bg_embeddings = torch.stack(bg_embeddings)
    return bg_embeddings_mappings


def obtain_gt_concept_embeddings(img_embeddings, concept_labels):
    unique_concept_labels = concept_labels.unique().tolist()
    concept_embeddings_mappings = dict()
    for label in unique_concept_labels:
        concept_embeddings_mappings[label] = torch.mean(
            img_embeddings[concept_labels == label], dim=0
        )
    return concept_embeddings_mappings


def obtain_gt_concept_bg_embeddings(img_embeddings, concept_labels, bg_labels):
    unique_bg_labels = torch.tensor(bg_labels).unique()
    unique_concept_labels = concept_labels.unique().tolist()
    bg_label_concept_label_embedding_mappings = dict()
    for bg_label in unique_bg_labels:
        for concept_label in unique_bg_labels:
            bg_label_concept_label_embedding_mappings[(bg_label, concept_label)] = (
                torch.mean(
                    img_embeddings[
                        (bg_labels == bg_label) & (concept_labels == concept_label)
                    ],
                    dim=0,
                )
            )
    return bg_label_concept_label_embedding_mappings


def classification_with_joint_concept_models(
    full_log_dir,
    full_data_dir,
    train_images,
    train_labels,
    train_hash,
    valid_images,
    valid_labels,
    test_images,
    test_labels,
    total_concepts,
    method="ct",
    epochs=200,
    batch_size=32,
    sub_embedding_size=20,
    cosine=False,
    qualitative=False,
    seed=0,
):
    full_out_folder = os.path.join(full_data_dir, "output/")
    num_classes = len(train_labels.unique())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert method in joint_method_dict
    start = time.time()
    method_class = joint_method_dict[method](
        total_concepts,
        num_classes,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        log_dir=full_log_dir,
    )
    method_class.training(
        train_images, train_labels, valid_images, valid_labels, test_images, test_labels
    )
    duration = time.time() - start
    train_acc = method_class.testing(train_images, train_labels)
    valid_acc = method_class.testing(valid_images, valid_labels)
    test_acc = method_class.testing(test_images, test_labels)
    print("final train acc: ", train_acc)
    print("final valid acc: ", valid_acc)
    print("final test acc: ", test_acc)
    save_learned_concepts(
        full_out_folder,
        method_class,
        None,
        None,
        None,
        duration,
        method,
        epochs,
        total_concepts,
        sub_embedding_size,
        train_hash,
        cosine=cosine,
        seed=seed,
    )


def classification_with_learned_concepts(
    full_data_folder,
    other_info,
    train_attr_labels,
    valid_attr_labels,
    test_attr_labels,
    train_images,
    valid_images,
    test_images,
    log_dir,
    train_embeddings,
    valid_embeddings,
    test_embeddings,
    train_hash_val,
    valid_hash_val,
    test_hash_val,
    labels,
    valid_labels,
    test_labels,
    full_concept_count_ls,
    method="ours",
    sub_embedding_size=20,
    cosine=False,
    epochs=200,
    qualitative=False,
):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # if full_image:
    # full_concept_count_ls = [full_concept_count_ls[0]]
    full_out_folder = os.path.join(full_data_folder, "output/")
    # method, n_patches, concept_count, sub_embedding_size, samples_hash, full_image=False
    all_train_activation_sample_ls = []
    all_valid_activation_sample_ls = []
    all_test_activation_sample_ls = []
    if qualitative:
        evaluate_cross_similarity(labels, other_info[0], train_embeddings)
    # for idx in range(len(full_concept_count_ls)):
    concept_count_ls = full_concept_count_ls  # [idx]
    concept_count = sum(concept_count_ls)
    # patch_count = patch_count_ls[idx]

    # valid_all_concept_ls, valid_cluster_dist_ls, valid_cluster_label_ls, valid_cluster_centroid_ls = load_learned_concepts(method, n_patches, concept_count, valid_hash_val)
    # test_all_concept_ls, test_cluster_dist_ls, test_cluster_label_ls, test_cluster_centroid_ls = load_learned_concepts(method, n_patches, concept_count, test_hash_val)
    if method == "ours":
        (
            all_concept_ls,
            train_cluster_dist_ls,
            train_cluster_label_ls,
            cluster_centroid_ls,
        ) = load_learned_concepts0(
            full_out_folder,
            method,
            epochs,
            concept_count_ls,
            sub_embedding_size=sub_embedding_size,
            samples_hash=train_hash_val,
            cosine=cosine,
        )

        # if not full_image_classification:
        #     train_cluster_dist_ls, train_cluster_label_ls = get_cluster_dist_all(all_concept_ls, train_embeddings[idx], cluster_centroid_ls, device, cosine=cosine)
        #     valid_cluster_dist_ls, valid_cluster_label_ls = get_cluster_dist_all(all_concept_ls, valid_embeddings[idx], cluster_centroid_ls, device, cosine=cosine)
        #     test_cluster_dist_ls, test_cluster_label_ls = get_cluster_dist_all(all_concept_ls, test_embeddings[idx], cluster_centroid_ls, device, cosine=cosine)

        #     train_activation_sample_ls = construct_dataset_by_activations(train_img_per_patch[idx], train_cluster_label_ls, train_cluster_dist_ls)
        #     valid_activation_sample_ls = construct_dataset_by_activations(valid_img_per_patch[idx], valid_cluster_label_ls, valid_cluster_dist_ls)
        #     test_activation_sample_ls = construct_dataset_by_activations(test_img_per_patch[idx], test_cluster_label_ls, test_cluster_dist_ls)
        # else:
        train_cluster_dist_ls, train_cluster_label_ls = get_cluster_dist_all(
            all_concept_ls, train_embeddings, cluster_centroid_ls, device, cosine=cosine
        )
        valid_cluster_dist_ls, valid_cluster_label_ls = get_cluster_dist_all(
            all_concept_ls, valid_embeddings, cluster_centroid_ls, device, cosine=cosine
        )
        test_cluster_dist_ls, test_cluster_label_ls = get_cluster_dist_all(
            all_concept_ls, test_embeddings, cluster_centroid_ls, device, cosine=cosine
        )
        train_activation_sample_ls = construct_dataset_by_activations_full_image(
            train_cluster_label_ls, train_cluster_dist_ls
        )
        valid_activation_sample_ls = construct_dataset_by_activations_full_image(
            valid_cluster_label_ls, valid_cluster_dist_ls
        )
        test_activation_sample_ls = construct_dataset_by_activations_full_image(
            test_cluster_label_ls, test_cluster_dist_ls
        )

        if qualitative:
            bg_labels = None
            if other_info is not None:
                bg_gt_embeddings = obtain_image_gt_bg_embeddings(
                    train_embeddings, other_info[0]
                )
                bg_labels = other_info[0]
            attr_name_ids_mapping = aggregate_attr_info(
                os.path.join(full_data_folder, "attributes.txt")
            )
            attr_name_ls, train_agg_attr_labels, _ = aggregate_attr_ids(
                attr_name_ids_mapping, train_attr_labels
            )
            train_attr_embeddings = obtain_image_gt_attr_embeddings(
                train_embeddings, train_agg_attr_labels
            )
            evaluate_learned_concepts_predictions_full(
                bg_labels, train_labels, train_activation_sample_ls
            )
            do_qualitative_study_main(
                attr_name_ls,
                log_dir,
                bg_labels,
                train_agg_attr_labels,
                train_labels,
                train_images,
                train_activation_sample_ls,
                train_cluster_label_ls,
                concept_count_ls,
            )
            compositionality_evaluation(
                all_concept_ls,
                cluster_centroid_ls,
                bg_labels,
                train_labels,
                train_activation_sample_ls,
                train_embeddings,
                train_cluster_label_ls,
                concept_count_ls,
            )

    elif method == "GT":
        train_activation_sample_ls = torch.nn.functional.one_hot(train_labels)
        valid_activation_sample_ls = torch.nn.functional.one_hot(valid_labels)
        test_activation_sample_ls = torch.nn.functional.one_hot(test_labels)
        if other_info is not None:
            train_bg_labels, valid_bg_labels, test_bg_labels = other_info
            num_bg_count = len(
                torch.cat([train_bg_labels, valid_bg_labels, test_bg_labels]).unique()
            )
            train_activation_sample_ls = torch.cat(
                [
                    train_activation_sample_ls,
                    torch.nn.functional.one_hot(
                        train_bg_labels, num_classes=num_bg_count
                    ),
                ],
                dim=-1,
            )
            valid_activation_sample_ls = torch.cat(
                [
                    valid_activation_sample_ls,
                    torch.nn.functional.one_hot(
                        valid_bg_labels, num_classes=num_bg_count
                    ),
                ],
                dim=-1,
            )
            test_activation_sample_ls = torch.cat(
                [
                    test_activation_sample_ls,
                    torch.nn.functional.one_hot(
                        test_bg_labels, num_classes=num_bg_count
                    ),
                ],
                dim=-1,
            )
    elif method in unsupervised_method_dict:
        # save_learned_concepts(learned_concepts, cluster_concept_activations, None, None, method, n_patches, total_concepts, sub_embedding_size, hash_val, full_image=full_image, cosine=cosine)
        (
            method_class,
            train_cluster_dist_ls,
            train_cluster_label_ls,
            cluster_centroid_ls,
        ) = load_learned_concepts0(
            full_out_folder,
            method,
            epochs,
            concept_count,
            sub_embedding_size=sub_embedding_size,
            samples_hash=train_hash_val,
            cosine=cosine,
        )
        # train_activation_sample_ls = train_cluster_dist_ls
        # if full_image_classification:
        train_activation_sample_ls = method_class.compute_sample_activations(
            train_embeddings
        )
        valid_activation_sample_ls = method_class.compute_sample_activations(
            valid_embeddings
        )
        test_activation_sample_ls = method_class.compute_sample_activations(
            test_embeddings
        )
        # train_activation_sample_ls = get_concept_activations_per_sample0(train_embeddings, torch.from_numpy(all_concept_ls))
        # valid_activation_sample_ls = get_concept_activations_per_sample0(valid_embeddings, torch.from_numpy(all_concept_ls))
        # test_activation_sample_ls = get_concept_activations_per_sample0(test_embeddings, torch.from_numpy(all_concept_ls))
        # else:
        #     train_activation_sample_ls = get_concept_activations_per_sample0_by_images(method_class, train_embeddings[idx], train_img_per_patch[idx])
        #     valid_activation_sample_ls = get_concept_activations_per_sample0_by_images(method_class, valid_embeddings[idx], valid_img_per_patch[idx])
        #     test_activation_sample_ls = get_concept_activations_per_sample0_by_images(method_class, test_embeddings[idx], test_img_per_patch[idx])

        if qualitative:
            if other_info is not None:
                bg_gt_embeddings = obtain_image_gt_bg_embeddings(
                    train_embeddings, other_info[0]
                )
                bg_labels = other_info[0]
            evaluate_learned_concepts_predictions_full(
                bg_labels, train_labels, train_activation_sample_ls
            )
            compositionality_evaluation_baseline(
                all_concept_ls,
                bg_labels,
                train_labels,
                train_activation_sample_ls,
                train_embeddings,
                concept_count,
            )

    # all_train_activation_sample_ls.append(train_activation_sample_ls)
    # all_valid_activation_sample_ls.append(valid_activation_sample_ls)
    # all_test_activation_sample_ls.append(test_activation_sample_ls)
    if qualitative:
        exit(1)

    # if full_image_classification:
    train_data = train_activation_sample_ls
    valid_data = valid_activation_sample_ls
    test_data = test_activation_sample_ls
    # else:
    #     train_data = all_train_activation_sample_ls
    #     valid_data = all_valid_activation_sample_ls
    #     test_data = all_test_activation_sample_ls
    # net = create_deep_set_net_for_programs(input_size=sum(concept_count_ls), output_size=len(labels.unique()))

    # log_dir, sample_concept_activations, sample_labels, valid_concept_activations_per_samples, test_concept_activations_per_samples, valid_labels, test_labels, mod=None
    train_eval_classification_model_by_concepts(
        log_dir, train_data, labels, valid_data, test_data, valid_labels, test_labels
    )

    # train_deepsets_models(log_dir, net, optimizer, criterion, samples, labels, valid_concept_activations_per_samples, valid_labels, epoch=100, batch_size=64)

    print()

    # return all_concept_ls, cluster_centroid_ls


def derive_representative_labels_per_clusters(
    attr_count,
    k,
    bg_labels,
    train_attr_labels_curr_cluster,
    train_bg_labels_curr_cluster,
    train_labels_curr_cluster,
    train_labels,
    attr_name_ls,
):
    mean_attr_labels_curr_cluster = torch.mean(train_attr_labels_curr_cluster, dim=0)
    attr_highest_frequency = mean_attr_labels_curr_cluster.max()
    attr_highest_frequency_id = mean_attr_labels_curr_cluster.argmax()
    is_bg_concept = False
    if bg_labels is not None:
        bg_label_highest_freq = -1
        bg_label_highest_freq_id = -1
        for label in train_bg_labels_curr_cluster.unique():
            bg_label_freq = torch.mean(
                (train_bg_labels_curr_cluster == label).type(torch.float)
            )
            if bg_label_freq > bg_label_highest_freq:
                bg_label_highest_freq = bg_label_freq
                bg_label_highest_freq_id = label

        # if bg_label_highest_freq >= attr_highest_frequency:
        #     print(" attr: ", attr_count, "concept::", k, " bg label: ", bg_label_highest_freq, bg_label_highest_freq_id)
        #     is_bg_concept = True
    print(
        " attr: ",
        attr_count,
        "concept::",
        k,
        " bg label: ",
        bg_label_highest_freq,
        bg_label_highest_freq_id,
    )
    label_highest_freq = -1
    label_highest_freq_id = -1
    label_freq_tensor = torch.zeros(len(train_labels.unique()))
    for label in train_labels_curr_cluster.unique():
        label_freq = torch.mean((train_labels_curr_cluster == label).type(torch.float))
        label_freq_tensor[label] = label_freq
        if label_freq > label_highest_freq:
            label_highest_freq = label_freq
            label_highest_freq_id = label
    print("label frequency::", label_freq_tensor.tolist())
    if label_highest_freq >= attr_highest_frequency:
        print(
            " attr: ",
            attr_count,
            "concept::",
            k,
            " label: ",
            label_highest_freq,
            label_highest_freq_id,
        )

    if not is_bg_concept:
        print(
            " attr: ",
            attr_count,
            "concept::",
            k,
            " attr label: ",
            attr_highest_frequency,
            attr_highest_frequency_id,
            attr_name_ls[attr_highest_frequency_id],
        )


def derive_representative_labels_per_clusters2(
    attr_count,
    k,
    bg_labels,
    train_bg_labels_curr_cluster,
    train_labels_curr_cluster,
    train_labels,
):
    # mean_attr_labels_curr_cluster = torch.mean(train_attr_labels_curr_cluster, dim=0)
    # attr_highest_frequency = mean_attr_labels_curr_cluster.max()
    # attr_highest_frequency_id = mean_attr_labels_curr_cluster.argmax()
    is_bg_concept = True
    if bg_labels is not None:
        bg_label_highest_freq = -1
        bg_label_highest_freq_id = -1
        for label in train_bg_labels_curr_cluster.unique():
            bg_label_freq = torch.mean(
                (train_bg_labels_curr_cluster == label).type(torch.float)
            )
            if bg_label_freq > bg_label_highest_freq:
                bg_label_highest_freq = bg_label_freq.item()
                bg_label_highest_freq_id = label.item()

        # if bg_label_highest_freq >= attr_highest_frequency:
        #     print(" attr: ", attr_count, "concept::", k, " bg label: ", bg_label_highest_freq, bg_label_highest_freq_id)
        #     is_bg_concept = True
    print(
        " attr: ",
        attr_count,
        "concept::",
        k,
        " bg label: ",
        bg_label_highest_freq,
        bg_label_highest_freq_id,
    )
    label_highest_freq = -1
    label_highest_freq_id = -1
    label_freq_tensor = torch.zeros(len(train_labels.unique()))
    for label in train_labels_curr_cluster.unique():
        label_freq = torch.mean((train_labels_curr_cluster == label).type(torch.float))
        label_freq_tensor[label] = label_freq
        if label_freq > label_highest_freq:
            label_highest_freq = label_freq.item()
            label_highest_freq_id = label.item()
    print("label frequency::", label_freq_tensor.tolist())
    if label_highest_freq >= bg_label_highest_freq:
        is_bg_concept = False
        print(
            " attr: ",
            attr_count,
            "concept::",
            k,
            " label: ",
            label_highest_freq,
            label_highest_freq_id,
        )

    return label_highest_freq_id, bg_label_highest_freq_id, is_bg_concept
    # if not is_bg_concept:
    #     print(" attr: ", attr_count, "concept::", k, " attr label: ", attr_highest_frequency, attr_highest_frequency_id, attr_name_ls[attr_highest_frequency_id])


def do_qualitative_study_main(
    attr_name_ls,
    full_log_dir,
    bg_labels,
    train_attr_labels,
    train_labels,
    train_images,
    train_sample_ls,
    train_cluster_label_ls,
    concept_count_ls,
):
    # if full_image:
    total_count = 0
    attr_count = 0
    for count in concept_count_ls:
        sub_folder_for_attr = os.path.join(full_log_dir, "attr_" + str(attr_count))
        if os.path.exists(sub_folder_for_attr):
            shutil.rmtree(sub_folder_for_attr)
            # os.rmdir(sub_folder_for_attr)
        os.makedirs(sub_folder_for_attr, exist_ok=True)
        curr_train_cluster_label_ls = train_cluster_label_ls[attr_count]
        for k in range(count):
            sub_sub_folder_for_attr = os.path.join(
                full_log_dir, "attr_" + str(attr_count), "concept_" + str(k)
            )
            os.makedirs(sub_sub_folder_for_attr, exist_ok=True)
            train_sample_ls_curr_cluster = train_sample_ls[
                curr_train_cluster_label_ls == k, total_count + k
            ]
            train_attr_labels_curr_cluster = train_attr_labels[
                curr_train_cluster_label_ls == k
            ]
            if bg_labels is not None:
                train_bg_labels_curr_cluster = bg_labels[
                    curr_train_cluster_label_ls == k
                ]
            train_labels_curr_cluster = train_labels[curr_train_cluster_label_ls == k]

            derive_representative_labels_per_clusters(
                attr_count,
                k,
                bg_labels,
                train_attr_labels_curr_cluster,
                train_bg_labels_curr_cluster,
                train_labels_curr_cluster,
                train_labels,
                attr_name_ls,
            )

            similar_sample_idx = torch.topk(
                train_sample_ls_curr_cluster.view(-1),
                k=min(10, len(train_sample_ls_curr_cluster.view(-1))),
            )[1]
            total_similar_sample_idx = torch.nonzero(
                torch.from_numpy(curr_train_cluster_label_ls) == k
            ).view(-1)[similar_sample_idx]

            curr_attr_labels = train_attr_labels_curr_cluster[similar_sample_idx]
            curr_labels = train_labels_curr_cluster[similar_sample_idx]
            curr_bg_labels = None
            if bg_labels is not None:
                curr_bg_labels = train_bg_labels_curr_cluster[similar_sample_idx]

            print("top k information::")
            derive_representative_labels_per_clusters(
                attr_count,
                k,
                bg_labels,
                curr_attr_labels,
                curr_bg_labels,
                curr_labels,
                train_labels,
                attr_name_ls,
            )

            most_prob_labels = torch.sum(curr_attr_labels, dim=0).max()
            most_prob_label_id = torch.sum(curr_attr_labels, dim=0).argmax()
            most_prob_labels, most_prob_label_id = torch.topk(
                torch.sum(curr_attr_labels, dim=0), k=10
            )

            train_labels_curr_cluster[similar_sample_idx]
            similar_sample_idx = total_similar_sample_idx.tolist()
            for sample_idx in range(len(similar_sample_idx)):
                img = train_images[similar_sample_idx[sample_idx]]
                img.save(
                    os.path.join(
                        sub_sub_folder_for_attr,
                        "sample_"
                        + str(sample_idx)
                        + "_"
                        + str(similar_sample_idx[sample_idx])
                        + ".jpg",
                    )
                )
            print(
                " attr: ",
                attr_count,
                "concept::",
                k,
                " similar labels: ",
                most_prob_labels,
                most_prob_label_id,
            )
        total_count += count
        attr_count += 1

    print()


#     for idx in range(len(train_sample_ls)):
def evaluate_cross_similarity(train_labels, bg_labels, train_embeddings):
    unique_train_labels = train_labels.unique().tolist()
    unique_train_labels.sort()
    unique_bg_labels = bg_labels.unique().tolist()
    unique_bg_labels.sort()
    gt_concept_embedding_ls = []
    train_embeddings = preprocessing_data(train_embeddings)
    label_name_ls = ["label_" + str(l) for l in unique_train_labels] + [
        "bg_label_" + str(l) for l in unique_bg_labels
    ]
    for label in unique_train_labels:
        gt_concept_embedding_ls.append(
            train_embeddings[train_labels == label].mean(dim=0)
        )

    for bg_label in unique_bg_labels:
        gt_concept_embedding_ls.append(
            train_embeddings[bg_labels == bg_label].mean(dim=0)
        )

    gt_concept_embedding_tensor = torch.stack(gt_concept_embedding_ls)

    cos_sim_mat = torch.mm(
        gt_concept_embedding_tensor, gt_concept_embedding_tensor.T
    ) / (
        torch.norm(gt_concept_embedding_tensor, dim=-1, keepdim=True)
        * torch.norm(gt_concept_embedding_tensor, dim=-1, keepdim=True).T
    )

    plt.imshow(cos_sim_mat.numpy(), cmap="viridis", interpolation="nearest")
    plt.colorbar()  # Add colorbar to show the scale of similarities
    # Adding labels and ticks
    num_items = len(label_name_ls)
    for i in range(num_items):
        for j in range(num_items):
            plt.text(
                j,
                i,
                f"{cos_sim_mat[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
            )
    # color_shape_ls = [item if "_" not in item else item.replace("_", "\n") for item in color_shape_ls]
    plt.xticks(range(num_items), label_name_ls)
    plt.yticks(range(num_items), label_name_ls)
    # plt.xlabel('Item')
    # plt.ylabel('Item')
    # plt.title('Cross Similarities between Items')

    plt.savefig("cross_similarities_CUB.png")
    print()


def evaluate_compositionality_score(
    compose_gt_concept_embedding0,
    compose_gt_concept_embedding1,
    compose_gt_concept_embedding,
):
    X = torch.stack([compose_gt_concept_embedding0, compose_gt_concept_embedding1])
    sim_score = compositionality_eval(
        compose_gt_concept_embedding,
        X,
        labels=torch.ones([len(compose_gt_concept_embedding), len(X)]),
    )
    # lr = LinearRegression()
    # X = torch.stack([compose_gt_concept_embedding0, compose_gt_concept_embedding1]).numpy()
    # lr.fit(X.T, compose_gt_concept_embedding.view(-1,1).numpy())
    # gt_sim = np.dot(np.dot(lr.coef_, X).reshape(-1), compose_gt_concept_embedding.view(1,-1).numpy().reshape(-1))/(np.linalg.norm(np.dot(lr.coef_, X))*np.linalg.norm(compose_gt_concept_embedding))

    return sim_score


def compositionality_evaluation(
    proj_space_ls,
    cluster_center_ls,
    bg_labels,
    train_labels,
    train_sample_ls,
    orig_train_embeddings,
    train_cluster_label_ls,
    concept_count_ls,
):
    # if full_image:
    total_count = 0
    attr_count = 0
    all_bg_label_ls = []
    all_label_ls = []
    all_gt_concept_embedding_ls = []
    all_concept_embedding_ls = []
    orig_train_embeddings = preprocessing_data(orig_train_embeddings)
    train_embeddings = orig_train_embeddings.clone()
    all_bg_flag_ls = []
    cluster_label_id_ls = []
    for count in concept_count_ls:
        curr_train_cluster_label_ls = train_cluster_label_ls[attr_count]
        curr_proj_space = proj_space_ls[attr_count]
        # bg_label_ls = []
        # label_ls = []
        # bg_flag_ls = []
        concept_embedding_ls = []
        gt_concept_embedding_ls = []
        alpha1_times_m1 = (
            train_embeddings
            @ proj_space_ls[0].T
            @ torch.inverse(proj_space_ls[0] @ proj_space_ls[0].T)
            @ proj_space_ls[0]
        )
        train_embeddings2 = train_embeddings - alpha1_times_m1
        alpha2 = train_embeddings2 @ proj_space_ls[1].T
        alpha2_times_m2 = (
            alpha2
            @ proj_space_ls[1]
            @ torch.inverse(proj_space_ls[1].T @ proj_space_ls[1])
        )

        for k in range(count):
            curr_cluster_center = cluster_center_ls[attr_count][k]
            # concept_embedding = (curr_proj_space.T @ curr_cluster_center.view(-1,1)).view(-1)

            train_sample_ls_curr_cluster = train_sample_ls[
                curr_train_cluster_label_ls == k, total_count + k
            ]
            # train_attr_labels_curr_cluster = train_attr_labels[curr_train_cluster_label_ls==k]
            if bg_labels is not None:
                train_bg_labels_curr_cluster = bg_labels[
                    curr_train_cluster_label_ls == k
                ]
            train_labels_curr_cluster = train_labels[curr_train_cluster_label_ls == k]

            label_highest_freq_id, bg_label_highest_freq_id, is_bg_concept = (
                derive_representative_labels_per_clusters2(
                    attr_count,
                    k,
                    bg_labels,
                    train_bg_labels_curr_cluster,
                    train_labels_curr_cluster,
                    train_labels,
                )
            )
            if is_bg_concept:
                subset_training_embeddings = train_embeddings[
                    bg_labels == bg_label_highest_freq_id
                ]
            else:
                subset_training_embeddings = train_embeddings[
                    train_labels == label_highest_freq_id
                ]

            gt_concept_embeddings = torch.mean(subset_training_embeddings, dim=0)
            concept_embedding = torch.mean(
                train_embeddings[curr_train_cluster_label_ls == k], dim=0
            )
            gt_concept_embedding_ls.append(gt_concept_embeddings)
            concept_embedding_ls.append(concept_embedding)
            print(
                torch.dot(gt_concept_embeddings, concept_embedding)
                / (torch.norm(gt_concept_embeddings) * torch.norm(concept_embedding))
            )

            all_bg_label_ls.append(bg_label_highest_freq_id)
            all_label_ls.append(label_highest_freq_id)
            all_bg_flag_ls.append(is_bg_concept)
            cluster_label_id_ls.append((attr_count, k))
            # derive_representative_labels_per_clusters(attr_count, k, bg_labels, train_attr_labels_curr_cluster, train_bg_labels_curr_cluster, train_labels_curr_cluster, train_labels, attr_name_ls)

            similar_sample_idx = torch.topk(
                train_sample_ls_curr_cluster.view(-1),
                k=min(10, len(train_sample_ls_curr_cluster.view(-1))),
            )[1]
            total_similar_sample_idx = torch.nonzero(
                torch.from_numpy(curr_train_cluster_label_ls) == k
            ).view(-1)[similar_sample_idx]

            # curr_attr_labels = train_attr_labels_curr_cluster[similar_sample_idx]
            curr_labels = train_labels_curr_cluster[similar_sample_idx]
            train_embeddings_curr_cluster = train_embeddings[
                curr_train_cluster_label_ls == k
            ]
            curr_bg_labels = None
            if bg_labels is not None:
                curr_bg_labels = train_bg_labels_curr_cluster[similar_sample_idx]

            print("top k information::")
            (
                label_highest_freq_id_topk,
                bg_label_highest_freq_id_topk,
                is_bg_concept_topk,
            ) = derive_representative_labels_per_clusters2(
                attr_count, k, bg_labels, curr_bg_labels, curr_labels, train_labels
            )
            if is_bg_concept_topk:
                subset_training_embeddings = train_embeddings_curr_cluster[
                    similar_sample_idx
                ][curr_bg_labels == bg_label_highest_freq_id_topk]
            else:
                subset_training_embeddings = train_embeddings_curr_cluster[
                    similar_sample_idx
                ][curr_labels == label_highest_freq_id_topk]

            gt_concept_embeddings_topk = torch.mean(subset_training_embeddings, dim=0)
            print(
                torch.dot(gt_concept_embeddings_topk, concept_embedding)
                / (
                    torch.norm(gt_concept_embeddings_topk)
                    * torch.norm(concept_embedding)
                )
            )

            # derive_representative_labels_per_clusters(attr_count, k, bg_labels, curr_attr_labels, curr_bg_labels, curr_labels, train_labels, attr_name_ls)

            # most_prob_labels = torch.sum(curr_attr_labels, dim=0).max()
            # most_prob_label_id = torch.sum(curr_attr_labels, dim=0).argmax()
            # most_prob_labels, most_prob_label_id= torch.topk(torch.sum(curr_attr_labels, dim=0),k=10)

            # train_labels_curr_cluster[similar_sample_idx]
            # similar_sample_idx = total_similar_sample_idx.tolist()
            # for sample_idx in range(len(similar_sample_idx)):
            #     img = train_images[similar_sample_idx[sample_idx]]
            #     img.save(os.path.join(sub_sub_folder_for_attr, "sample_" + str(sample_idx) + "_" + str(similar_sample_idx[sample_idx]) + ".jpg"))
            # print(" attr: ", attr_count, "concept::", k, " similar labels: ", most_prob_labels, most_prob_label_id)
        # all_bg_label_ls.append(bg_label_ls)
        # all_label_ls.append(label_ls)

        # train_embeddings = ablate_concepts(train_embeddings, curr_proj_space)
        all_concept_embedding_ls.append(torch.stack(concept_embedding_ls))
        all_gt_concept_embedding_ls.append(torch.stack(gt_concept_embedding_ls))
        # all_bg_flag_ls.append(bg_flag_ls)
        total_count += count
        attr_count += 1

    all_concept_embedding_tensor = torch.cat(all_concept_embedding_ls)
    all_gt_concept_embedding_tensor = torch.cat(all_gt_concept_embedding_ls)
    single_concept_embedding_sims = all_concept_embedding_tensor.unsqueeze(
        1
    ) @ all_gt_concept_embedding_tensor.unsqueeze(2)
    single_concept_embedding_sims = single_concept_embedding_sims.view(-1) / (
        torch.norm(all_concept_embedding_tensor, dim=-1)
        * torch.norm(all_gt_concept_embedding_tensor, dim=-1)
    )
    print("single concept error::", torch.mean(single_concept_embedding_sims))

    all_bg_flag_tensor = torch.tensor(all_bg_flag_ls).view(-1)
    all_bg_label_tensor = torch.tensor(all_bg_label_ls).view(-1)
    all_label_tensor = torch.tensor(all_label_ls).view(-1)
    all_concept_embedding_tensor = torch.cat(all_concept_embedding_ls)
    bg_flag_idx_ls = (all_bg_flag_tensor == 1).nonzero().view(-1)
    non_bg_flag_idx_ls = (all_bg_flag_tensor == 0).nonzero().view(-1)
    error_ls = []
    assert (
        torch.max(bg_flag_idx_ls) > torch.max(non_bg_flag_idx_ls)
        and torch.min(bg_flag_idx_ls) > torch.max(non_bg_flag_idx_ls)
    ) or (
        torch.max(bg_flag_idx_ls) < torch.max(non_bg_flag_idx_ls)
        and torch.max(bg_flag_idx_ls) < torch.min(non_bg_flag_idx_ls)
    )

    train_embeddings = orig_train_embeddings.clone()
    sim0_ls = []
    sim1_ls = []
    sim2_ls = []
    gt_compositionality_score_ls = []
    gt_sample_compositionality_score_ls = []
    compositionality_score_ls = []
    sample_compositionality_score_ls = []
    compositionality_score_on_GT_ls = []
    sample_compositionality_score_on_GT_ls = []
    for bg_idx in bg_flag_idx_ls:
        for non_bg_idx in non_bg_flag_idx_ls:
            bg_label = all_bg_label_tensor[bg_idx]
            non_bg_label = all_label_tensor[non_bg_idx]
            bg_concept_embedding = all_concept_embedding_tensor[bg_idx]
            non_bg_concept_embedding = all_concept_embedding_tensor[non_bg_idx]
            subset_sample_embeddings = train_embeddings[
                (train_labels == non_bg_label) & (bg_labels == bg_label)
            ]
            subset_sample_embeddings0 = train_embeddings[(train_labels == non_bg_label)]
            subset_sample_embeddings1 = train_embeddings[(bg_labels == bg_label)]
            compose_gt_concept_embedding0 = torch.mean(subset_sample_embeddings0, dim=0)
            compose_gt_concept_embedding1 = torch.mean(subset_sample_embeddings1, dim=0)
            compose_gt_concept_embedding = torch.mean(subset_sample_embeddings, dim=0)
            # lr = LinearRegression()
            # X = torch.stack([compose_gt_concept_embedding0, compose_gt_concept_embedding1]).numpy()
            # lr.fit(X.T, compose_gt_concept_embedding.view(-1,1).numpy())
            # gt_sim = np.dot(np.dot(lr.coef_, X).reshape(-1), compose_gt_concept_embedding.view(1,-1).numpy().reshape(-1))/(np.linalg.norm(np.dot(lr.coef_, X))*np.linalg.norm(compose_gt_concept_embedding))
            gt_sim = evaluate_compositionality_score(
                compose_gt_concept_embedding0,
                compose_gt_concept_embedding1,
                compose_gt_concept_embedding.unsqueeze(0),
            )
            print("ground truth compositionality score::", gt_sim)
            gt_compositionality_score_ls.append(gt_sim)

            sample_gt_sim = evaluate_compositionality_score(
                compose_gt_concept_embedding0,
                compose_gt_concept_embedding1,
                subset_sample_embeddings,
            )
            print("sample ground truth compositionality score::", sample_gt_sim)
            gt_sample_compositionality_score_ls.append(sample_gt_sim)
            # pred_y = lr.predict(X)

            non_bg_attr_idx, non_bg_cluster_idx = cluster_label_id_ls[non_bg_idx]
            non_bg_cluster_labels = train_cluster_label_ls[non_bg_attr_idx]
            bg_attr_idx, bg_cluster_idx = cluster_label_id_ls[bg_idx]
            bg_cluster_labels = train_cluster_label_ls[bg_attr_idx]

            compose_concept_embedding = torch.mean(
                train_embeddings[
                    (non_bg_cluster_labels == non_bg_cluster_idx)
                    & (bg_cluster_labels == bg_cluster_idx)
                ],
                dim=0,
            )
            compose_concept_embedding0 = torch.mean(
                train_embeddings[(non_bg_cluster_labels == non_bg_cluster_idx)], dim=0
            )
            compose_concept_embedding1 = torch.mean(
                train_embeddings[(bg_cluster_labels == bg_cluster_idx)], dim=0
            )
            sim = evaluate_compositionality_score(
                compose_concept_embedding0,
                compose_concept_embedding1,
                compose_concept_embedding.unsqueeze(0),
            )
            print("compositionality score::", sim)
            compositionality_score_ls.append(sim)

            sample_sim = evaluate_compositionality_score(
                compose_concept_embedding0,
                compose_concept_embedding1,
                subset_sample_embeddings,
            )
            print("sample compositionality score::", sample_sim)
            sample_compositionality_score_ls.append(sample_sim)

            sim = evaluate_compositionality_score(
                compose_gt_concept_embedding0,
                compose_gt_concept_embedding1,
                compose_concept_embedding.unsqueeze(0),
            )
            print("compositionality score::", sim)
            compositionality_score_on_GT_ls.append(sim)

            sample_sim = evaluate_compositionality_score(
                compose_gt_concept_embedding0,
                compose_gt_concept_embedding1,
                subset_sample_embeddings,
            )
            print("sample compositionality score::", sample_sim)
            sample_compositionality_score_on_GT_ls.append(sample_sim)

            sim0 = torch.dot(
                compose_gt_concept_embedding @ proj_space_ls[0].T,
                cluster_center_ls[0][bg_idx - min(bg_flag_idx_ls)],
            ) / (
                torch.norm(compose_gt_concept_embedding @ proj_space_ls[0].T)
                * torch.norm(cluster_center_ls[0][bg_idx - min(bg_flag_idx_ls)])
            )
            sim1 = torch.dot(
                compose_gt_concept_embedding @ proj_space_ls[1].T,
                cluster_center_ls[1][non_bg_idx - min(non_bg_flag_idx_ls)],
            ) / (
                torch.norm(compose_gt_concept_embedding @ proj_space_ls[1].T)
                * torch.norm(cluster_center_ls[1][non_bg_idx - min(non_bg_flag_idx_ls)])
            )
            compose_gt_concept_embedding0 = ablate_concepts(
                compose_gt_concept_embedding, proj_space_ls[0]
            )
            sim2 = torch.dot(
                compose_gt_concept_embedding0 @ proj_space_ls[1].T,
                cluster_center_ls[1][non_bg_idx - min(non_bg_flag_idx_ls)],
            ) / (
                torch.norm(compose_gt_concept_embedding0 @ proj_space_ls[1].T)
                * torch.norm(cluster_center_ls[1][non_bg_idx - min(non_bg_flag_idx_ls)])
            )
            print(
                "similarities between gt concept and derived concepts::",
                sim0,
                sim1,
                sim2,
            )
            sim0_ls.append(sim0.item())
            sim1_ls.append(sim1.item())
            sim2_ls.append(sim2.item())
            # error = eval_compositionality_metrics(subset_sample_embeddings, [bg_concept_embedding, non_bg_concept_embedding])
            # error_ls.append(error)
            # print("bg label::", bg_label, " non bg label::", non_bg_label, " cosine sim::", cosine_sim)

    print(
        "average sample ground truth compositionality score::",
        torch.mean(torch.tensor(gt_sample_compositionality_score_ls)),
    )
    print(
        "average ground truth compositionality score::",
        torch.mean(torch.tensor(gt_compositionality_score_ls)),
    )

    print(
        "average compositionality score::",
        torch.mean(torch.tensor(compositionality_score_ls)),
    )
    print(
        "average sample compositionality score::",
        torch.mean(torch.tensor(sample_compositionality_score_ls)),
    )

    print(
        "average compositionality score on GT::",
        torch.mean(torch.tensor(compositionality_score_on_GT_ls)),
    )
    print(
        "average sample compositionality score on GT::",
        torch.mean(torch.tensor(sample_compositionality_score_on_GT_ls)),
    )

    print(
        "average error::",
        torch.mean(torch.tensor(sim0_ls + sim1_ls)),
        torch.mean(torch.tensor(sim0_ls + sim2_ls)),
    )

    # for attr_idx in range(len(all_concept_embedding_ls)):
    #     all_concept_embedding_ls[attr_idx] = torch.stack(all_concept_embedding_ls[attr_idx])

    print()


def evaluate_learned_concepts_predictions_full(bg_labels, train_labels, concept_scores):
    bg_onehot_labels = torch.nn.functional.one_hot(bg_labels)
    train_onehot_labels = torch.nn.functional.one_hot(train_labels)
    gt_onehot_labels = torch.cat([bg_onehot_labels, train_onehot_labels], dim=-1)
    evaluate_learned_concepts_predictions(gt_onehot_labels, concept_scores)


def compositionality_evaluation_baseline(
    cluster_center_ls,
    bg_labels,
    train_labels,
    train_sample_ls,
    orig_train_embeddings,
    concept_count,
):
    # all_concept_ls, bg_labels, train_labels, train_activation_sample_ls, train_embeddings, concept_count
    # if full_image:
    total_count = 0
    attr_count = 0
    all_bg_label_ls = []
    all_label_ls = []
    all_gt_concept_embedding_ls = []
    all_concept_embedding_ls = []
    orig_train_embeddings = preprocessing_data(orig_train_embeddings)
    train_embeddings = orig_train_embeddings.clone()
    all_bg_flag_ls = []
    cluster_center_ls = torch.from_numpy(cluster_center_ls)
    # train_concept_sims = train_embeddings @ cluster_center_ls.T/(torch.norm(train_embeddings, dim=-1).unsqueeze(1)*torch.norm(cluster_center_ls, dim=-1).unsqueeze(0))
    train_concept_pred_labels = train_sample_ls.argmax(dim=-1)

    cluster_label_id_ls = []

    for count in range(concept_count):
        concept_embedding = cluster_center_ls[count]
        # bg_label_ls = []
        # label_ls = []
        # bg_flag_ls = []
        concept_embedding_ls = []
        gt_concept_embedding_ls = []

        # for k in range(count):
        #     curr_cluster_center = cluster_center_ls[attr_count][k]
        #     concept_embedding = (curr_proj_space.T @ curr_cluster_center.view(-1,1)).view(-1)

        train_sample_ls_curr_cluster = train_sample_ls[
            train_concept_pred_labels == count, count
        ]
        #     # train_attr_labels_curr_cluster = train_attr_labels[curr_train_cluster_label_ls==k]
        if bg_labels is not None:
            train_bg_labels_curr_cluster = bg_labels[train_concept_pred_labels == count]
        train_labels_curr_cluster = train_labels[train_concept_pred_labels == count]

        label_highest_freq_id, bg_label_highest_freq_id, is_bg_concept = (
            derive_representative_labels_per_clusters2(
                count,
                0,
                bg_labels,
                train_bg_labels_curr_cluster,
                train_labels_curr_cluster,
                train_labels,
            )
        )
        if is_bg_concept:
            subset_training_embeddings = train_embeddings[
                bg_labels == bg_label_highest_freq_id
            ]
        else:
            subset_training_embeddings = train_embeddings[
                train_labels == label_highest_freq_id
            ]

        gt_concept_embeddings = torch.mean(subset_training_embeddings, dim=0)

        gt_concept_embedding_ls.append(gt_concept_embeddings)
        concept_embedding_ls.append(concept_embedding)
        print(
            torch.dot(gt_concept_embeddings, concept_embedding)
            / (torch.norm(gt_concept_embeddings) * torch.norm(concept_embedding))
        )

        all_bg_label_ls.append(bg_label_highest_freq_id)
        all_label_ls.append(label_highest_freq_id)
        all_bg_flag_ls.append(is_bg_concept)
        cluster_label_id_ls.append(count)

        # derive_representative_labels_per_clusters(attr_count, k, bg_labels, train_attr_labels_curr_cluster, train_bg_labels_curr_cluster, train_labels_curr_cluster, train_labels, attr_name_ls)

        similar_sample_idx = torch.topk(
            train_sample_ls_curr_cluster.view(-1),
            k=min(10, len(train_sample_ls_curr_cluster.view(-1))),
        )[1]
        # total_similar_sample_idx = torch.nonzero(torch.from_numpy(train_sample_ls)==k).view(-1)[similar_sample_idx]

        # curr_attr_labels = train_attr_labels_curr_cluster[similar_sample_idx]
        curr_labels = train_labels_curr_cluster[similar_sample_idx]
        train_embeddings_curr_cluster = train_embeddings[
            train_concept_pred_labels == count
        ]
        curr_bg_labels = None
        if bg_labels is not None:
            curr_bg_labels = train_bg_labels_curr_cluster[similar_sample_idx]

        print("top k information::")
        (
            label_highest_freq_id_topk,
            bg_label_highest_freq_id_topk,
            is_bg_concept_topk,
        ) = derive_representative_labels_per_clusters2(
            count, 0, bg_labels, curr_bg_labels, curr_labels, train_labels
        )
        if is_bg_concept_topk:
            subset_training_embeddings = train_embeddings_curr_cluster[
                similar_sample_idx
            ][curr_bg_labels == bg_label_highest_freq_id_topk]
        else:
            subset_training_embeddings = train_embeddings_curr_cluster[
                similar_sample_idx
            ][curr_labels == label_highest_freq_id_topk]

        gt_concept_embeddings_topk = torch.mean(subset_training_embeddings, dim=0)
        print(
            torch.dot(gt_concept_embeddings_topk, concept_embedding)
            / (torch.norm(gt_concept_embeddings_topk) * torch.norm(concept_embedding))
        )

        # derive_representative_labels_per_clusters(attr_count, k, bg_labels, curr_attr_labels, curr_bg_labels, curr_labels, train_labels, attr_name_ls)

        # most_prob_labels = torch.sum(curr_attr_labels, dim=0).max()
        # most_prob_label_id = torch.sum(curr_attr_labels, dim=0).argmax()
        # most_prob_labels, most_prob_label_id= torch.topk(torch.sum(curr_attr_labels, dim=0),k=10)

        # train_labels_curr_cluster[similar_sample_idx]
        # similar_sample_idx = total_similar_sample_idx.tolist()
        # for sample_idx in range(len(similar_sample_idx)):
        #     img = train_images[similar_sample_idx[sample_idx]]
        #     img.save(os.path.join(sub_sub_folder_for_attr, "sample_" + str(sample_idx) + "_" + str(similar_sample_idx[sample_idx]) + ".jpg"))
        # print(" attr: ", attr_count, "concept::", k, " similar labels: ", most_prob_labels, most_prob_label_id)
        # all_bg_label_ls.append(bg_label_ls)
        # all_label_ls.append(label_ls)

        # train_embeddings = ablate_concepts(train_embeddings, curr_proj_space)
        all_concept_embedding_ls.append(torch.stack(concept_embedding_ls))
        all_gt_concept_embedding_ls.append(torch.stack(gt_concept_embedding_ls))
        # all_bg_flag_ls.append(bg_flag_ls)
        total_count += count
        attr_count += 1

    all_concept_embedding_tensor = torch.cat(all_concept_embedding_ls)
    all_gt_concept_embedding_tensor = torch.cat(all_gt_concept_embedding_ls)
    single_concept_embedding_sims = all_concept_embedding_tensor.unsqueeze(
        1
    ) @ all_gt_concept_embedding_tensor.unsqueeze(2)
    single_concept_embedding_sims = single_concept_embedding_sims.view(-1) / (
        torch.norm(all_concept_embedding_tensor, dim=-1)
        * torch.norm(all_gt_concept_embedding_tensor, dim=-1)
    )
    print("single concept error::", torch.mean(single_concept_embedding_sims))

    all_bg_flag_tensor = torch.tensor(all_bg_flag_ls).view(-1)
    all_bg_label_tensor = torch.tensor(all_bg_label_ls).view(-1)
    all_label_tensor = torch.tensor(all_label_ls).view(-1)
    all_concept_embedding_tensor = torch.cat(all_concept_embedding_ls)
    bg_flag_idx_ls = (all_bg_flag_tensor == 1).nonzero().view(-1)
    non_bg_flag_idx_ls = (all_bg_flag_tensor == 0).nonzero().view(-1)
    error_ls = []
    # assert (torch.max(bg_flag_idx_ls) > torch.max(non_bg_flag_idx_ls) and torch.min(bg_flag_idx_ls) > torch.max(non_bg_flag_idx_ls))  or (torch.max(bg_flag_idx_ls) < torch.max(non_bg_flag_idx_ls) and torch.max(bg_flag_idx_ls) < torch.min(non_bg_flag_idx_ls))

    train_embeddings = orig_train_embeddings.clone()
    sim0_ls = []
    sim1_ls = []
    gt_sim_ls = []
    gt_sample_sim_ls = []
    compositionality_score_ls = []
    sample_compositionality_score_ls = []

    compositionality_score_on_gt_ls = []
    sample_compositionality_on_gt_score_ls = []

    train_bg_activations = train_sample_ls[:, bg_flag_idx_ls]
    pred_bg_labels = train_bg_activations.argmax(dim=-1)
    pred_bg_labels = bg_flag_idx_ls[pred_bg_labels]
    train_label_activations = train_sample_ls[:, non_bg_flag_idx_ls]
    pred_concept_labels = train_label_activations.argmax(dim=-1)
    pred_concept_labels = non_bg_flag_idx_ls[pred_concept_labels]

    for bg_idx in bg_flag_idx_ls:
        for non_bg_idx in non_bg_flag_idx_ls:
            bg_label = all_bg_label_tensor[bg_idx]
            non_bg_label = all_label_tensor[non_bg_idx]
            # bg_concept_embedding = all_concept_embedding_tensor[bg_idx]
            # non_bg_concept_embedding = all_concept_embedding_tensor[non_bg_idx]
            subset_sample_embeddings = train_embeddings[
                (train_labels == non_bg_label) & (bg_labels == bg_label)
            ]
            subset_sample_embeddings0 = train_embeddings[(train_labels == non_bg_label)]
            subset_sample_embeddings1 = train_embeddings[(bg_labels == bg_label)]

            compose_gt_concept_embedding = torch.mean(subset_sample_embeddings, dim=0)
            compose_gt_concept_embedding0 = torch.mean(subset_sample_embeddings0, dim=0)
            compose_gt_concept_embedding1 = torch.mean(subset_sample_embeddings1, dim=0)

            sim0 = torch.dot(
                compose_gt_concept_embedding, cluster_center_ls[bg_idx]
            ) / (
                torch.norm(compose_gt_concept_embedding)
                * torch.norm(cluster_center_ls[bg_idx])
            )
            sim1 = torch.dot(
                compose_gt_concept_embedding, cluster_center_ls[non_bg_idx]
            ) / (
                torch.norm(compose_gt_concept_embedding)
                * torch.norm(cluster_center_ls[non_bg_idx])
            )

            gt_sim = evaluate_compositionality_score(
                compose_gt_concept_embedding0,
                compose_gt_concept_embedding1,
                compose_gt_concept_embedding.unsqueeze(0),
            )
            print("ground-truth compositionality score::", gt_sim)
            gt_sim_ls.append(gt_sim)

            gt_sample_sim = evaluate_compositionality_score(
                compose_gt_concept_embedding0,
                compose_gt_concept_embedding1,
                subset_sample_embeddings,
            )
            print("ground-truth sample compositionality score::", gt_sample_sim)
            gt_sample_sim_ls.append(gt_sample_sim)

            compose_concept_embedding = torch.mean(
                train_embeddings[
                    (pred_concept_labels == non_bg_idx) & (pred_bg_labels == bg_idx)
                ],
                dim=0,
            )
            compose_concept_embedding0 = torch.mean(
                train_embeddings[(pred_concept_labels == non_bg_idx)], dim=0
            )
            compose_concept_embedding1 = torch.mean(
                train_embeddings[(pred_bg_labels == bg_idx)], dim=0
            )
            sim = evaluate_compositionality_score(
                compose_concept_embedding0,
                compose_concept_embedding1,
                compose_concept_embedding.unsqueeze(0),
            )
            print("compositionality score::", sim)
            compositionality_score_ls.append(sim)

            sample_sim = evaluate_compositionality_score(
                compose_concept_embedding0,
                compose_concept_embedding1,
                subset_sample_embeddings,
            )
            print("sample compositionality score::", sample_sim)
            sample_compositionality_score_ls.append(sample_sim)

            sim = evaluate_compositionality_score(
                compose_gt_concept_embedding0,
                compose_gt_concept_embedding1,
                compose_concept_embedding.unsqueeze(0),
            )
            print("compositionality score on GT::", sim)
            compositionality_score_on_gt_ls.append(sim)

            sample_sim = evaluate_compositionality_score(
                compose_gt_concept_embedding0,
                compose_gt_concept_embedding1,
                subset_sample_embeddings,
            )
            print("sample compositionality score on GT::", sample_sim)
            sample_compositionality_on_gt_score_ls.append(sample_sim)

            # compose_gt_concept_embedding0 = ablate_concepts(compose_gt_concept_embedding, proj_space_ls[0])
            # sim2 = torch.dot(compose_gt_concept_embedding0 @ proj_space_ls[1].T, cluster_center_ls[1][non_bg_idx - min(non_bg_flag_idx_ls)])/(torch.norm(compose_gt_concept_embedding0 @ proj_space_ls[1].T)*torch.norm(cluster_center_ls[1][non_bg_idx - min(non_bg_flag_idx_ls)]))
            print("similarities between gt concept and derived concepts::", sim0, sim1)
            sim0_ls.append(sim0.item())
            sim1_ls.append(sim1.item())
            # error = eval_compositionality_metrics(subset_sample_embeddings, [bg_concept_embedding, non_bg_concept_embedding])
            # error_ls.append(error)
            # print("bg label::", bg_label, " non bg label::", non_bg_label, " cosine sim::", cosine_sim)

    print(
        "average ground truth compositionality score::",
        torch.mean(torch.tensor(gt_sim_ls)),
    )
    print(
        "average sample ground truth compositionality score::",
        torch.mean(torch.tensor(gt_sample_sim_ls)),
    )

    print(
        "average compositionality score::",
        torch.mean(torch.tensor(compositionality_score_ls)),
    )
    print(
        "average sample compositionality score::",
        torch.mean(torch.tensor(sample_compositionality_score_ls)),
    )

    print(
        "average compositionality score on GT::",
        torch.mean(torch.tensor(compositionality_score_on_gt_ls)),
    )
    print(
        "average sample compositionality score on GT::",
        torch.mean(torch.tensor(sample_compositionality_on_gt_score_ls)),
    )

    print("average similarity score:", torch.mean(torch.tensor(sim0_ls + sim1_ls)))

    print()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    set_rand_seed(args.seed)

    # patch_count_ls = [4]

    if args.dataset_name == "controlled_cub" or args.dataset_name == "celeba_subset2":
        # concept_count_ls = [[2,4],[2,4]]
        concept_count_ls = [2, 4]  # ]*len(patch_count_ls)
    # elif args.dataset_name == "CUB_subset":
    #     concept_count_ls = [3,3]#*len(patch_count_ls)
    else:
        concept_count_ls = [
            args.concept_num_per_attr
        ] * args.num_attrs  # ]*len(patch_count_ls)
    print(concept_count_ls)

    # patch_count_ls = sorted(patch_count_ls)
    # patch_count_str = ",".join([str(x) for x in patch_count_ls])
    args.split_method = args.split_method.lower()
    if args.split_method == "ours":
        log_folder_suffix = get_suffix(
            args.split_method,
            args.epochs,
            concept_count_ls,
            args.projection_size,
            None,
            cosine=args.cosine_sim,
        )
    else:
        log_folder_suffix = get_suffix(
            args.split_method,
            args.epochs,
            sum(concept_count_ls),
            None,
            None,
            cosine=False,
        )

    root_dir = os.path.dirname(os.path.realpath(__file__))
    args.dataset_name = args.dataset_name.lower()
    # get the data from the concept bottleneck paper
    if args.dataset_name.lower().startswith("celeba"):
        full_data_dir = os.path.join(args.data_dir, "celeba")
    elif args.dataset_name.lower().startswith("cub"):
        full_data_dir = os.path.join(args.data_dir, "cub")
    else:
        full_data_dir = os.path.join(args.data_dir, args.dataset_name.lower())
    os.makedirs(full_data_dir, exist_ok=True)

    (
        train_images,
        valid_images,
        test_images,
        train_labels,
        valid_labels,
        test_labels,
        train_attr_labels,
        valid_attr_labels,
        test_attr_labels,
        other_info,
    ) = get_image_data_ls(
        args,
        full_data_dir,
        root_dir,
        get_bounding_box=False,
        use_attr=True,
        get_bg_info=(args.dataset_name == "controlled_cub"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224')
    # model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224').to(device)
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    raw_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    processor = lambda images: raw_processor(
        images=images,
        return_tensors="pt",
        padding=False,
        do_resize=False,
        do_center_crop=False,
    )["pixel_values"]
    img_processor = lambda images: raw_processor(images=images, return_tensors="pt")[
        "pixel_values"
    ]

    def vit_forward(imgs, masks=None):
        # inputs = processor(imgs, return_tensors="pt").to("cuda")

        with torch.no_grad():
            # Select the CLS token embedding from the last hidden layer
            return model.get_image_features(pixel_values=imgs)

    full_log_dir = os.path.join(full_data_dir, "logs_" + log_folder_suffix)
    print("log dir: ", full_log_dir)
    os.makedirs(full_log_dir, exist_ok=True)

    # full_train_img_embedding_file = os.path.join(full_log_dir, "train_image_embeddings.pkl")
    # full_valid_img_embedding_file = os.path.join(full_log_dir, "valid_image_embeddings.pkl")
    # full_test_img_embedding_file = os.path.join(full_log_dir, "test_image_embeddings.pkl")

    # if os.path.exists(full_train_img_embedding_file) and os.path.exists(full_valid_img_embedding_file) and os.path.exists(full_test_img_embedding_file):
    #     train_embeddings, train_labels, train_attr_labels = load_objs(full_train_img_embedding_file)
    #     valid_embeddings, valid_labels, valid_attr_labels = load_objs(full_valid_img_embedding_file)
    #     test_embeddings, test_labels, test_attr_labels = load_objs(full_test_img_embedding_file)
    # else:

    # img_output_dir = os.path.join(full_log_dir, "image_output")
    # os.makedirs(img_output_dir, exist_ok=True)
    # train_img_output_dir = os.path.join(img_output_dir, "train")
    # if not os.path.exists(train_img_output_dir):
    #     os.makedirs(train_img_output_dir, exist_ok=True)
    # valid_img_output_dir = os.path.join(img_output_dir, "valid")
    # if not os.path.exists(valid_img_output_dir):
    #     os.makedirs(valid_img_output_dir, exist_ok=True)
    # test_img_output_dir = os.path.join(img_output_dir, "test")
    # if not os.path.exists(test_img_output_dir):
    #     os.makedirs(test_img_output_dir, exist_ok=True)

    train_hash = hashfn(train_images)
    valid_hash = hashfn(valid_images)
    test_hash = hashfn(test_images)

    # patch_count_ls = [32]

    # if not os.path.exists(full_train_img_embedding_file):
    train_embeddings = obtain_image_embeddings(
        train_images,
        train_labels,
        vit_forward,
        img_processor,
        args,
        device,
        full_data_dir,
    )
    # output_images_with_bboxes(train_images, train_bboxes, train_img_output_dir)
    # else:
    #     train_embeddings, train_patch_emb, train_masks, train_bboxes, train_img_per_patch = obtain_image_embeddings(train_images, train_labels, vit_forward, img_processor, args, device, patch_count_ls=patch_count_ls)

    # output_images_with_bboxes(train_images, train_masks, train_bboxes, train_img_output_dir, train_attr_labels, idx_attr_property_name_mappings, attr_property_val_idx_mappings)
    # save_objs((train_embeddings, train_labels, train_attr_labels), full_train_img_embedding_file)
    # get_subsets_by_bird_parts(train_images, train_bboxes, train_patch_emb)
    # if not os.path.exists(full_valid_img_embedding_file):
    valid_embeddings = obtain_image_embeddings(
        valid_images,
        valid_labels,
        vit_forward,
        img_processor,
        args,
        device,
        full_data_dir,
    )

    test_embeddings = obtain_image_embeddings(
        test_images,
        test_labels,
        vit_forward,
        img_processor,
        args,
        device,
        full_data_dir,
    )
    # output_images_with_bboxes(test_images, test_bboxes, test_img_output_dir)
    # else:
    #     test_embeddings, test_patch_emb, test_masks, test_bboxes, test_img_per_patch = obtain_image_embeddings(test_images, test_labels, vit_forward, img_processor, args, device, patch_count_ls=patch_count_ls)
    # save_objs((test_embeddings, test_labels, test_attr_labels), full_test_img_embedding_file)
    # output_images_with_bboxes(test_images, test_masks, test_bboxes, test_img_output_dir, test_attr_labels, idx_attr_property_name_mappings, attr_property_val_idx_mappings)
    print()

    # concept_count_ls = [10, 10]
    # all_existing_concept_ls = None
    (
        existing_concept_ls,
        existing_cluster_dist_ls,
        existing_cluster_label_ls,
        existing_cluster_centroid_ls,
    ) = None, None, None, None
    if args.existing_concept_logs is not None:
        all_existing_concept_ls = []
        # for concept_count in concept_count_ls:
        (
            existing_concept_ls,
            existing_cluster_dist_ls,
            existing_cluster_label_ls,
            existing_cluster_centroid_ls,
        ) = load_objs(args.existing_concept_logs)
        # existing_concept_ls, _,_,_ = load_learned_concepts0(args.method, patch_count_ls, concept_count_ls, sub_embedding_size=args.projection_size, samples_hash=train_hash, full_image=args.full_image, cosine=args.cosine)
        # all_existing_concept_ls.append(existing_concept_ls)

    if not args.do_classification:
        if args.cosine_sim:
            lr = args.lr
        else:
            lr = args.lr
        if args.split_method in joint_method_dict:
            total_concepts = sum(concept_count_ls)
            for seed in range(3):
                classification_with_joint_concept_models(
                    full_log_dir,
                    full_data_dir,
                    train_embeddings,
                    train_labels,
                    train_hash,
                    valid_embeddings,
                    valid_labels,
                    test_embeddings,
                    test_labels,
                    total_concepts,
                    method=args.split_method,
                    epochs=args.epochs,
                    sub_embedding_size=args.projection_size,
                    cosine=args.cosine_sim,
                    seed=seed,
                )
        elif (
            args.split_method in unsupervised_method_dict or args.split_method == "ours"
        ):
            for seed in range(3):
                learn_concept_main(
                    full_data_dir,
                    full_log_dir,
                    train_embeddings,
                    train_hash,
                    valid_embeddings,
                    full_concept_count_ls=concept_count_ls,
                    lr=lr,
                    sub_embedding_size=args.projection_size,
                    method=args.split_method,
                    cosine=args.cosine_sim,
                    epochs=args.epochs,
                    existing_concept_ls=existing_concept_ls,
                    existing_cluster_dist_ls=existing_cluster_dist_ls,
                    existing_cluster_label_ls=existing_cluster_label_ls,
                    existing_cluster_centroid_ls=existing_cluster_centroid_ls,
                    seed=seed,
                    cross_entropy=args.cross_entropy,
                )
    elif args.do_classification_neural:
        train_eval_classification_model_by_concepts(
            None,
            train_embeddings,
            train_labels,
            valid_embeddings,
            test_embeddings,
            valid_labels,
            test_labels,
            full_image=True,
            epochs=0,
        )
    else:
        # if not args.full_image_classification:
        #     train_input_embs = train_patch_emb#torch.cat(train_patch_emb, dim=0)
        #     valid_input_embs = valid_patch_emb#torch.cat(valid_patch_emb, dim=0)
        #     test_input_embs = test_patch_emb#torch.cat(test_patch_emb, dim=0)
        # else:
        train_input_embs = train_embeddings
        valid_input_embs = valid_embeddings
        test_input_embs = test_embeddings

        train_center = torch.mean(train_input_embs, dim=0)
        train_embeddings = train_input_embs - train_center
        valid_embeddings = valid_input_embs - train_center
        test_embeddings = test_input_embs - train_center

        # if args.split_method not in joint_method_dict:
        #     classification_with_learned_concepts(full_data_dir, other_info, train_attr_labels, valid_attr_labels, test_attr_labels, train_images, valid_images, test_images, full_log_dir, train_input_embs, valid_input_embs, test_input_embs, train_hash, valid_hash, test_hash, train_labels, valid_labels, test_labels, concept_count_ls, method=args.split_method, sub_embedding_size=args.projection_size, cosine=args.cosine_sim,  epochs=args.epochs, qualitative=args.qualitative)
        # else:
        # if args.split_method in joint_method_dict:
        #     total_concepts = sum(concept_count_ls)
        #     for seed in range(3):
        #         classification_with_joint_concept_models(full_log_dir, full_data_dir, train_embeddings, train_labels, train_hash, valid_embeddings, valid_labels, test_embeddings, test_labels, total_concepts, method=args.split_method, epochs=args.epochs, sub_embedding_size=args.projection_size, cosine=args.cosine_sim, seed=seed)

        full_out_folder = os.path.join(full_data_dir, "output/")
        if args.split_method != "ours":
            concept_count_ls = sum(concept_count_ls)

        gt_concepts = []
        for i in range(train_attr_labels.shape[1]):
            gt_concepts.append(
                torch.mean(train_embeddings[train_attr_labels[:, i] == 1], dim=0)
            )
        gt_concepts = normalize(torch.stack(gt_concepts))

        all_results = []
        for seed in range(3):
            results = {}
            if args.split_method != "gt" and args.split_method != "random":
                (
                    all_concept_ls,
                    train_cluster_dist_ls,
                    train_cluster_label_ls,
                    cluster_centroid_ls,
                    duration,
                ) = load_learned_concepts0(
                    full_out_folder,
                    args.split_method,
                    args.epochs,
                    concept_count_ls,
                    sub_embedding_size=args.projection_size,
                    samples_hash=train_hash,
                    cosine=args.cosine_sim,
                    seed=seed,
                    cross_entropy=args.cross_entropy,
                )
                results["duration"] = duration

            if args.split_method == "ours":
                concepts = []
                for i in range(len(all_concept_ls)):
                    concepts.append(cluster_centroid_ls[i] @ all_concept_ls[i])
                concepts = normalize(torch.cat(concepts, dim=0))
            elif args.split_method == "ct":
                concepts = torch.tensor(all_concept_ls.return_concepts()).cpu()[0]
            elif args.split_method == "gt":
                concepts = gt_concepts
                results["duration"] = 0
            elif args.split_method == "random":
                # set seed
                torch.manual_seed(seed)
                concepts = torch.randn_like(gt_concepts)
                results["duration"] = 0
            else:
                concepts = torch.tensor(all_concept_ls.return_concepts()).cpu().float()

            print(concepts.shape)

            # print("Max cosine similarity to each GT concept:")
            gt_cosim, gt_matches = concept_gt_match(concepts, gt_concepts)
            # attr_names = [f"attr_{i}" for i in range(gt_concepts.shape[0])]
            # for i, name in enumerate(attr_names):
            #     print(f"{name}: {gt_cosim[i].item():.3f}")

            # print(f"{args.split_method} & " + " & ".join([f"{gt_cosim[i].item():.3f}" for i in range(gt_concepts.shape[0])]) + " \\\\")

            # print("Avg. cosine:", gt_cosim.mean().item())
            results["mean_cosim"] = gt_cosim.mean().item()

            sample_concept_scores = cosim(train_embeddings, concepts)

            # print("Max AUC to each GT concept:")
            gt_label_matches, gt_auc, signs = concept_gt_match_labels(
                sample_concept_scores,
                train_attr_labels,
                allow_neg=args.split_method != "ours",
            )
            # for i, name in enumerate(attr_names):
            #     print(f"{name}: {gt_auc[i].item():.3f}")

            # print(f"{args.split_method} & " + " & ".join([f"{gt_auc[i].item():.3f}" for i in range(gt_concepts.shape[0])]) + " \\\\")
            # print("Avg. AUC:", gt_auc.mean().item())
            # print()
            results["mean_auc"] = gt_auc.mean().item()

            # gt_label_matches = gt_matches
            # print("Matches:", gt_label_matches)
            # Order the learned concepts to match the GT concepts and align their signs with the positive GT label
            matched_concepts = normalize(
                concepts[gt_label_matches] * signs.unsqueeze(1)
            )

            # print("Compositionality MAP:")
            unique_compositions = torch.unique(train_attr_labels, dim=0)
            comp_labels = torch.zeros(
                (train_attr_labels.shape[0], unique_compositions.shape[0])
            )
            for i, comp in enumerate(unique_compositions):
                comp_labels[(train_attr_labels == comp).all(dim=1), i] = 1
            # print(np.mean(compositional_f1(train_embeddings, matched_concepts, torch.tensor(train_attr_labels), comp_labels)))
            results["map"] = np.mean(
                compositional_f1(
                    train_embeddings,
                    matched_concepts,
                    torch.tensor(train_attr_labels),
                    comp_labels,
                )
            )

            # Get the composed concepts
            # composed_concepts, aucs, cos = concept_match(
            #     matched_concepts[:6],
            #     torch.tensor(grammar_labels[train_test_labels == 1, :6]),
            #     torch.tensor(grammar_labels[train_test_labels == 1, 7:]),
            #     torch.tensor(grammar_labels[train_test_labels == 0, 7:]),
            #     sample_emb[train_test_labels == 1],
            #     sample_emb[train_test_labels == 0])
            # print("Compositional AUCs:", aucs)

            # print("Compositionality score:")
            # print(compositionality_eval(test_embeddings, torch.tensor(matched_concepts), torch.tensor(test_attr_labels)))
            results["cscore"] = compositionality_eval(
                test_embeddings,
                torch.tensor(matched_concepts),
                torch.tensor(test_attr_labels),
            )
            all_results.append(results)

        merged_results = {}
        for key in all_results[0].keys():
            merged_results[key] = np.mean([r[key] for r in all_results])
            merged_results[key + "_std"] = np.std([r[key] for r in all_results])

        for key, value in merged_results.items():
            print(key, f"{value:.3f}")

        if os.path.exists("results/cub.csv"):
            df = pd.read_csv("results/cub.csv").to_dict("records")
        else:
            df = []
        merged_results["dataset"] = f"{args.dataset_name}"
        merged_results["model"] = "CLIP"
        merged_results["method"] = args.split_method
        if args.cross_entropy:
            merged_results["method"] += "_ce"
        df.append(merged_results)
        df = pd.DataFrame(df).to_csv("results/cub.csv", index=False)

        # print("GT Max AUC to each GT concept:")
        # sample_concept_scores = cosim(train_embeddings, gt_concepts)
        # gt_label_matches, gt_auc, signs = concept_gt_match_labels(sample_concept_scores, train_attr_labels)
        # print(f"GT & " + " & ".join([f"{gt_auc[i].item():.3f}" for i in range(gt_concepts.shape[0])]) + " \\\\")
        # print("GT Avg. AUC:", gt_auc.mean().item())
        # print()

        # print("GT Compositionality MAP:")
        # print(np.mean(compositional_f1(train_embeddings, gt_concepts, torch.tensor(train_attr_labels), comp_labels)))

        # print("GT Compositionality score:")
        # print(compositionality_eval(test_embeddings, torch.tensor(gt_concepts), torch.tensor(test_attr_labels)))

        # random_concepts = normalize(torch.randn_like(concepts))
        # gt_cosim, gt_matches = concept_gt_match(random_concepts, gt_concepts)
        # print("Random avg. cosine:", gt_cosim.mean().item())
        # sample_concept_scores = cosim(train_embeddings, random_concepts)
        # gt_label_matches, gt_auc, signs = concept_gt_match_labels(sample_concept_scores, train_attr_labels)
        # matched_concepts = normalize(random_concepts[gt_label_matches] * signs.unsqueeze(1))

        # print(f"Random & " + " & ".join([f"{gt_auc[i].item():.3f}" for i in range(gt_concepts.shape[0])]) + " \\\\")
        # print("Random Avg. AUC:", gt_auc.mean().item())
        # print()

        # print("Random Compositionality MAP:")
        # print(np.mean(compositional_f1(train_embeddings, matched_concepts, torch.tensor(train_attr_labels), comp_labels)))

        # print("Random Compositionality score:")
        # print(compositionality_eval(test_embeddings, torch.tensor(matched_concepts), torch.tensor(test_attr_labels)))

    # if args.select_images:
    #     sub_train_img_output_dir = os.path.join(img_output_dir, "sub_train")
    #     sub_valid_img_output_dir = os.path.join(img_output_dir, "sub_valid")
    #     sub_test_img_output_dir = os.path.join(img_output_dir, "sub_test")

    #     new_sub_train_img_output_dir = os.path.join(img_output_dir, "sub_train_by_sc")
    #     os.makedirs(new_sub_train_img_output_dir, exist_ok=True)
    #     select_samples_by_shape_color_combination(train_img_output_dir, sub_train_img_output_dir, new_sub_train_img_output_dir, len(train_images), shape_ls=["dagger", "spatulate", "hooked_seabird"], color_ls=["black", "grey", "buff"], topk=40)
    #     select_large_enough_images(train_img_output_dir, sub_train_img_output_dir, len(train_images))
    #     select_large_enough_images(valid_img_output_dir, sub_valid_img_output_dir, len(valid_images))
    #     select_large_enough_images(test_img_output_dir, sub_test_img_output_dir, len(test_images))
    #     exit(1)

    # mlp1 = train_mlps(train_embeddings, train_attr_labels[:, 0:9], hidden_size=256)
    # mlp2 = train_mlps(train_embeddings, train_attr_labels[:, 278:293], hidden_size=256)

    # cos_sim_mat = torch.zeros((mlp1.fc1.weight.shape[0], mlp2.fc1.weight.shape[0]))

    # for idx1 in range(mlp1.fc1.weight.shape[0]):
    #     for idx2 in range(mlp2.fc1.weight.shape[0]):
    #         vec1 = mlp1.fc1.weight[idx1].view(-1)
    #         vec2 = mlp2.fc1.weight[idx2].view(-1)
    #         cos_sim = torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))
    #         # cos_sim = torch.nn.functional.cosine_similarity(vec1, vec2)

    #         cos_sim_mat[idx1, idx2] = cos_sim
    # print()
