from __future__ import annotations

import os
import sys

import yaml

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import argparse
import pickle

import numpy as np

# from celeba_loader import prepare_celeba_data
import torch
from CUB_loader import load_data
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import random

from concept_learning import ConceptLearner
from utils import kmeans_clustering

# from ham_loader import *
# from imagenet_loader import *

all_celeba_attr_labels = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]


def normalize_and_center(x, center=None, std=None):
    return x


def compute_cosine_similarity_matrix(data1, data2):
    data1_c = torch.where(
        torch.logical_and(data1 > 0, data1 < 1e-4), torch.full_like(data1, 1e-4), data1
    )
    data2_c = torch.where(
        torch.logical_and(data2 < 0, data2 > -1e-4),
        torch.full_like(data2, -1e-4),
        data2,
    )
    sim_matrix = torch.mm(data1_c, data2_c.T) / (
        torch.norm(data1_c, dim=1).unsqueeze(1) * torch.norm(data2_c, dim=1) + 1e-4
    )
    return sim_matrix


def get_cluster_dist(
    split_space, data, cluster_centroids, device, cosine=False, center=None, std=None
):
    if type(split_space) is torch.Tensor:
        split_space = split_space.to(device)
        data = data.to(device)
        project = torch.matmul(data, split_space.t())
    else:
        project = split_space(data.to(device))

    if cosine:
        project = normalize_and_center(project, center=center, std=std)
    else:
        project = normalize_and_center(project, center=center, std=std)

    if not cosine:
        cluster_dist = torch.sqrt(
            torch.sum(
                (project.detach().unsqueeze(1) - cluster_centroids.unsqueeze(0)) ** 2,
                dim=-1,
            )
        )
    else:
        # cluster_dist = 1 - torch.mm(project.detach(), cluster_centroids.T)/(torch.norm(project.detach(), dim=1).unsqueeze(1)*torch.norm(cluster_centroids, dim=1))
        cluster_dist = 1 - compute_cosine_similarity_matrix(
            project.detach(), cluster_centroids
        )
    cluster_label = torch.argmin(cluster_dist, dim=-1)
    return project, cluster_dist, cluster_label


def normalize(x):
    if type(x) == torch.Tensor:
        return x / (x.norm(dim=1)[:, None] + 1e-6)
    else:
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)


def get_cluster_center_after_projection(
    split_space, data, n_clusters, device, cosine=False
):
    if split_space is not None:
        if type(split_space) is torch.Tensor:
            split_space = split_space.to(device)
            data = data.to(device)
            project = torch.matmul(data, split_space.t())
        else:
            project = split_space(data.to(device))
    else:
        project = data.to(device)

    center = torch.mean(project, dim=0)
    std = torch.std(project, dim=0)
    # project = normalize_and_center(project, center=center, std=std)
    # center, std = None, None
    if cosine:
        project = normalize_and_center(project)
    else:
        project = normalize_and_center(project)
    # kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(project.detach().cpu().numpy())
    # cluster_centroids = torch.from_numpy(kmeans.cluster_centers_).to(device)
    # cluster_labels = kmeans.labels_
    cluster_labels, cluster_centroids = kmeans_clustering(
        project, n_clusters, cosine=cosine
    )

    if not cosine:
        cluster_dist = torch.sqrt(
            torch.sum(
                (project.detach().unsqueeze(1) - cluster_centroids.unsqueeze(0)) ** 2,
                dim=-1,
            )
        )
    else:
        # cluster_dist = 1 - torch.mm(project.detach(), cluster_centroids.T)/(torch.norm(project.detach(), dim=1).unsqueeze(1)*torch.norm(cluster_centroids, dim=1))
        cluster_dist = 1 - compute_cosine_similarity_matrix(
            project.detach(), cluster_centroids
        )
    return project, cluster_dist, cluster_labels, cluster_centroids, (center, std)


def evaluate_learned_concepts_predictions(gt_labels, concept_scores):
    best_truth_score_ls = []
    for k in range(gt_labels.shape[1]):
        learned_idx = 0
        truth_sign = 1
        best_truth_score = 0
        for i in range(concept_scores.shape[1]):
            if roc_auc_score(gt_labels[:, k], concept_scores[:, i]) > best_truth_score:
                best_truth_score = roc_auc_score(gt_labels[:, k], concept_scores[:, i])
                learned_idx = i
                truth_sign = 1
        print(k, learned_idx, truth_sign, best_truth_score)
        best_truth_score_ls.append(best_truth_score)
    print("mean::", np.mean(np.array(best_truth_score_ls)))


def pred_deepsets_models(model, samples, batch_size=64):
    model.eval()
    pred_label_ls = []
    pred_prob_ls = []
    with torch.no_grad():
        for k in range(0, len(samples[0]), batch_size):
            end_idx = min(k + batch_size, len(samples[0]))
            sub_samples = [
                [samples[gid][idx] for idx in range(k, end_idx)]
                for gid in range(len(samples))
            ]
            if torch.cuda.is_available():
                sub_samples = [
                    [sample[sid].cuda() for sid in range(len(sample))]
                    for sample in sub_samples
                ]
            output = model(sub_samples)
            pred_labels = torch.argmax(output, dim=1)
            pred_probs = torch.softmax(output, dim=1)
            pred_label_ls.append(pred_labels.cpu())
            pred_prob_ls.append(pred_probs.cpu())
    pred_label_tensors = torch.cat(pred_label_ls)
    pred_prob_tensors = torch.cat(pred_prob_ls)
    return pred_label_tensors.numpy(), pred_prob_tensors.numpy()


def evaluate_performance(gt_labels, pred_labels, pred_probs):
    accuracy = (
        np.sum(gt_labels.numpy().reshape(-1) == pred_labels.reshape(-1))
        * 1.0
        / len(gt_labels)
    )
    print("accuracy::", accuracy)
    return accuracy


def train_deepsets_models(
    log_dir,
    model,
    optimizer,
    criterion,
    samples,
    labels,
    valid_concept_activations_per_samples,
    valid_labels,
    epoch=100,
    batch_size=64,
):
    model.train()
    best_val_acc = 0
    for _ in range(epoch):
        all_sample_ids = list(range(len(labels)))
        random.shuffle(all_sample_ids)
        total_loss = 0
        for k in range(0, len(all_sample_ids), batch_size):
            end_idx = min(k + batch_size, len(all_sample_ids))
            sub_samples = [
                [samples[gid][all_sample_ids[idx]] for idx in range(k, end_idx)]
                for gid in range(len(samples))
            ]
            sub_labels = [labels[all_sample_ids[idx]] for idx in range(k, end_idx)]
            if torch.cuda.is_available():
                sub_samples = [
                    [sample[sid].cuda() for sid in range(len(sample))]
                    for sample in sub_samples
                ]
                sub_labels = [label.cuda() for label in sub_labels]
            optimizer.zero_grad()
            output = model(sub_samples)
            sub_labels = torch.tensor(sub_labels)
            if torch.cuda.is_available():
                sub_labels = sub_labels.cuda()
            loss = criterion(output, sub_labels)
            loss.backward()
            total_loss += loss.item() * (end_idx - k)
            optimizer.step()
        pred_val_labels, pred_val_probs = pred_deepsets_models(
            model, valid_concept_activations_per_samples
        )
        val_acc = evaluate_performance(valid_labels, pred_val_labels, pred_val_probs)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pt"))
        print("average training loss::", total_loss / len(all_sample_ids))
        print("validation accuracy::", val_acc)
        print("best validation accuracy::", best_val_acc)
    return model


def obtain_image_embeddings(
    images, labels, vit_forward, processor, args, device, full_data_dir
):
    # samples: list[PIL.Image], input_to_latent, input_processor, device: str = 'cpu'
    # cl = ConceptLearner(images, labels, vit_forward, processor, args.dataset_name, device)
    cl = ConceptLearner(images, vit_forward, processor, device)
    # bbox x1,y1,x2,y2
    # image_embs = get_image_embeddings(images, cl.input_processor, cl.input_to_latent)
    # patch_emb_ls = []
    # masks_ls = []
    # bboxes_ls = []
    # img_per_patch_ls = []
    # for idx in range(len(patch_count_ls)):
    #     patch_count = patch_count_ls[idx]
    #     patches, img_emb = cl.get_patches(patch_count, images=images, method="none")
    #     img_emb, patch_emb, masks, bboxes, img_per_patch = cl.get_patches(patch_count, images=images, method="none")
    #     patch_emb_ls.append(patch_emb)
    #     masks_ls.append(masks)
    #     bboxes_ls.append(bboxes)
    #     img_per_patch_ls.append(img_per_patch)
    # output_folder = os.path.join(full_data_dir, "output")

    # if args.dataset_name.lower().startswith("cub") or args.dataset_name.lower().startswith("celeba") or args.dataset_name.lower() == "ham" or args.dataset_name.lower().startswith("imagenet"):
    #     patches, img_emb = cl.get_patches(output_folder, args.dataset_name, images=images, method="slic")
    # else:
    patches, img_emb = cl.get_patches(
        8, images=images, method="none", output_dir=full_data_dir
    )
    return img_emb


def get_suffix(
    method,
    epochs,
    concept_count,
    sub_embedding_size,
    samples_hash,
    full_image=True,
    cosine=False,
):
    if type(concept_count) == list:
        concept_count_str = ",".join([str(c) for c in concept_count])
    else:
        concept_count_str = concept_count
    if sub_embedding_size is None:
        sub_embedding_str = ""
    else:
        sub_embedding_str = f"_{sub_embedding_size}"

    file_key = get_file_key(method, epochs, concept_count_str, sub_embedding_size)
    if samples_hash is None:
        suffix = f"{file_key}{'_full_image' if full_image else ''}{'_cosine' if cosine else ''}"
    else:
        suffix = f"{file_key}_{samples_hash}{'_full_image' if full_image else ''}{'_cosine' if cosine else ''}"
    return suffix


def get_file_key(method, epochs, concept_count_str, sub_embedding_size):
    concat_str = f"{method}_{epochs}_{concept_count_str}_{sub_embedding_size}"
    # hash_val = hashfn(concat_str)
    # return hash_val
    return concat_str


def save_learned_concepts(
    full_out_folder,
    all_concept_ls,
    cluster_dist_ls,
    cluster_label_ls,
    cluster_centroid_ls,
    duration,
    method,
    epochs,
    concept_count,
    sub_embedding_size,
    samples_hash,
    full_image=True,
    cosine=False,
    seed=0,
    cross_entropy=False,
):
    if type(concept_count) == list:
        concept_count_str = ",".join([str(c) for c in concept_count])
    else:
        concept_count_str = concept_count
    file_key = get_file_key(method, epochs, concept_count_str, sub_embedding_size)
    saved_file_name = f"{full_out_folder}/saved_concepts_{file_key}_{samples_hash}{'_full_image' if full_image else ''}{'_cosine' if cosine else ''}{'_ce' if cross_entropy else ''}_{seed}.pkl"
    if not os.path.exists(f"{full_out_folder}/"):
        os.mkdir(f"{full_out_folder}/")
    save_objs(
        (
            all_concept_ls,
            cluster_dist_ls,
            cluster_label_ls,
            cluster_centroid_ls,
            duration,
        ),
        saved_file_name,
    )


def load_learned_concepts0(
    full_out_folder,
    method,
    epochs,
    concept_count,
    sub_embedding_size,
    samples_hash,
    full_image=True,
    cosine=False,
    seed=0,
    cross_entropy=False,
):
    if type(concept_count) == list:
        concept_count_str = ",".join([str(c) for c in concept_count])
    else:
        concept_count_str = concept_count
    file_key = get_file_key(method, epochs, concept_count_str, sub_embedding_size)
    saved_file_name = f"{full_out_folder}/saved_concepts_{file_key}_{samples_hash}{'_full_image' if full_image else ''}{'_cosine' if cosine else ''}{'_ce' if cross_entropy else ''}_{seed}.pkl"
    # if os.path.exists(saved_file_name):
    print("Loading cached patches")
    print(samples_hash)
    all_concept_ls, cluster_dist_ls, cluster_label_ls, cluster_centroid_ls, duration = (
        load_objs(saved_file_name)
    )
    return (
        all_concept_ls,
        cluster_dist_ls,
        cluster_label_ls,
        cluster_centroid_ls,
        duration,
    )


def load_learned_concepts(
    full_out_folder,
    method,
    epochs,
    concept_count,
    samples_hash,
    full_image=True,
    cosine=False,
    sub_embedding_size=100,
):
    if type(concept_count) == list:
        concept_count_str = ",".join([str(c) for c in concept_count])
    else:
        concept_count_str = concept_count
    file_key = get_file_key(method, epochs, concept_count_str, sub_embedding_size)
    saved_file_name = f"{full_out_folder}/saved_concepts_{file_key}_{samples_hash}{'_full_image' if full_image else ''}{'_cosine' if cosine else ''}.pkl"
    # if os.path.exists(saved_file_name):
    print("Loading cached patches")
    print(samples_hash)
    all_concept_ls, cluster_dist_ls, cluster_label_ls, cluster_centroid_ls = load_objs(
        saved_file_name
    )
    return all_concept_ls, cluster_dist_ls, cluster_label_ls, cluster_centroid_ls


def save_objs(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_objs(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def prepare_cub_data(
    root_dir,
    loaded_config,
    base_dir="/data2/wuyinjun/CUB/",
    use_attr=False,
    get_bounding_box=False,
    controlled_data=False,
):
    train_data_path = os.path.join(base_dir, "train.pkl")
    val_data_path = train_data_path.replace("train.pkl", "val.pkl")
    test_data_path = train_data_path.replace("train.pkl", "test.pkl")

    concept_transform = None

    train_dl = load_data(
        pkl_paths=[train_data_path],
        use_attr=use_attr,
        no_img=False,
        batch_size=loaded_config["batch_size"],
        uncertain_label=False,
        n_class_attr=2,
        image_dir="images",
        resampling=False,
        root_dir=root_dir,
        num_workers=loaded_config["num_workers"],
        concept_transform=concept_transform,
        return_raw_img=True,
        get_bounding_box=get_bounding_box,
        base_dir=base_dir,
        controlled_data=controlled_data,
    )
    val_dl = load_data(
        pkl_paths=[val_data_path],
        use_attr=use_attr,
        no_img=False,
        batch_size=loaded_config["batch_size"],
        uncertain_label=False,
        n_class_attr=2,
        image_dir="images",
        resampling=False,
        root_dir=root_dir,
        num_workers=loaded_config["num_workers"],
        concept_transform=concept_transform,
        return_raw_img=True,
        get_bounding_box=get_bounding_box,
        base_dir=base_dir,
        controlled_data=controlled_data,
    )

    test_dl = load_data(
        pkl_paths=[test_data_path],
        use_attr=use_attr,
        no_img=False,
        batch_size=loaded_config["batch_size"],
        uncertain_label=False,
        n_class_attr=2,
        image_dir="images",
        resampling=False,
        root_dir=root_dir,
        num_workers=loaded_config["num_workers"],
        concept_transform=concept_transform,
        return_raw_img=True,
        get_bounding_box=get_bounding_box,
        base_dir=base_dir,
        controlled_data=controlled_data,
    )

    return train_dl, val_dl, test_dl


def get_image_info_ls(train_dl):
    images = []
    label_ls = []
    attr_label_ls = []
    bbox_ls = []
    for idx in tqdm(range(len(train_dl.dataset))):
        # for idx in tqdm(range(20)):
        if not train_dl.dataset.use_attr:
            if not train_dl.dataset.get_bounding_box:
                (raw_img, img), y = train_dl.dataset[idx]
            else:
                (raw_img, img), y, bbox = train_dl.dataset[idx]
                bbox_ls.append(bbox)
        else:
            if not train_dl.dataset.get_bounding_box:
                (raw_img, img), y, attr_labels = train_dl.dataset[idx]
            else:
                (raw_img, img), y, attr_labels, bbox = train_dl.dataset[idx]
                bbox_ls.append(bbox)
            attr_label_ls.append(attr_labels)
        images.append(raw_img)
        label_ls.append(y)
    if len(attr_label_ls) > 0:
        if len(bbox_ls) > 0:
            return (
                images,
                torch.tensor(label_ls),
                torch.stack(attr_label_ls),
                torch.tensor(bbox_ls),
            )
        else:
            return images, torch.tensor(label_ls), torch.stack(attr_label_ls)
    elif len(bbox_ls) > 0:
        return images, torch.tensor(label_ls), torch.tensor(bbox_ls)
    else:
        return images, torch.tensor(label_ls)


def get_all_bg_labels(train_dl):
    bg_label_ls = []
    for idx in tqdm(range(len(train_dl.dataset))):
        img_data = train_dl.dataset.data[idx]
        bg_label = img_data["bg_label"]
        bg_label_ls.append(bg_label)
    bg_label_tensor = torch.tensor(bg_label_ls)
    return bg_label_tensor


def get_single_sub_images_by_bounding_box(image, bbox):
    # Convert the PIL image to a NumPy array
    image_array = np.array(image)

    # Extract the sub-image using array slicing
    sub_image = image_array[bbox[1] : bbox[3] + bbox[1], bbox[0] : bbox[2] + bbox[0], :]

    # Create a new PIL image from the sub-image array
    sub_image_pil = Image.fromarray(sub_image)
    # sub_image_pil.save("example2.png")
    return sub_image_pil


def get_all_sub_images_by_bounding_box(image_ls, bbox_ls):
    sub_image_ls = []
    for idx in tqdm(range(len(image_ls))):
        image = image_ls[idx]
        bbox = bbox_ls[idx]
        sub_image = get_single_sub_images_by_bounding_box(image, bbox.long().tolist())
        sub_image_ls.append(sub_image)
    return sub_image_ls


def select_common_labels_celeba(
    train_samples,
    valid_samples,
    test_samples,
    train_labels,
    valid_labels,
    test_labels,
    train_attr_labels,
    valid_attr_labels,
    test_attr_labels,
):
    train_label_set = set(train_labels.unique().tolist())
    valid_label_set = set(valid_labels.unique().tolist())
    test_label_set = set(test_labels.unique().tolist())
    common_label_set = train_label_set.intersection(valid_label_set).intersection(
        test_label_set
    )
    common_label_tensor = torch.tensor(list(common_label_set))

    common_label_tensor = common_label_tensor.sort()[0]

    train_idx = torch.nonzero(
        train_labels.view(-1, 1) == common_label_tensor.view(1, -1)
    )[:, 0]
    valid_idx = torch.nonzero(
        valid_labels.view(-1, 1) == common_label_tensor.view(1, -1)
    )[:, 0]
    test_idx = torch.nonzero(
        test_labels.view(-1, 1) == common_label_tensor.view(1, -1)
    )[:, 0]

    train_samples = [train_samples[idx] for idx in train_idx]
    valid_samples = [valid_samples[idx] for idx in valid_idx]
    test_samples = [test_samples[idx] for idx in test_idx]

    transformed_train_labels = (
        train_labels[train_idx].view(-1, 1) == common_label_tensor.view(1, -1)
    ).nonzero()[:, 1]
    transformed_valid_labels = (
        valid_labels[valid_idx].view(-1, 1) == common_label_tensor.view(1, -1)
    ).nonzero()[:, 1]
    transformed_test_labels = (
        test_labels[test_idx].view(-1, 1) == common_label_tensor.view(1, -1)
    ).nonzero()[:, 1]

    train_attr_labels = train_attr_labels[train_idx]
    valid_attr_labels = valid_attr_labels[valid_idx]
    test_attr_labels = test_attr_labels[test_idx]

    return (
        train_samples,
        valid_samples,
        test_samples,
        transformed_train_labels,
        transformed_valid_labels,
        transformed_test_labels,
        train_attr_labels,
        valid_attr_labels,
        test_attr_labels,
    )


def subset_celeba(
    train_images,
    valid_images,
    test_images,
    train_labels,
    valid_labels,
    test_labels,
    train_attr_labels,
    valid_attr_labels,
    test_attr_labels,
):
    label_idx_ls = torch.tensor(list(range(10)))
    train_idx = torch.nonzero(train_labels.view(-1, 1) == label_idx_ls.view(1, -1))[
        :, 0
    ]
    valid_idx = torch.nonzero(valid_labels.view(-1, 1) == label_idx_ls.view(1, -1))[
        :, 0
    ]
    test_idx = torch.nonzero(test_labels.view(-1, 1) == label_idx_ls.view(1, -1))[:, 0]
    train_images = [train_images[idx] for idx in train_idx]
    valid_images = [valid_images[idx] for idx in valid_idx]
    test_images = [test_images[idx] for idx in test_idx]
    train_attr_labels = train_attr_labels[train_idx]
    valid_attr_labels = valid_attr_labels[valid_idx]
    test_attr_labels = test_attr_labels[test_idx]
    return (
        train_images,
        valid_images,
        test_images,
        train_labels[train_idx],
        valid_labels[valid_idx],
        test_labels[test_idx],
        train_attr_labels,
        valid_attr_labels,
        test_attr_labels,
    )


def subset_celeba2_0(
    train_images,
    valid_images,
    test_images,
    train_attr_labels,
    valid_attr_labels,
    test_attr_labels,
    train_labels,
    valid_labels,
    test_labels,
):
    hair_color_attr = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"]
    hair_color_idx_ls = [all_celeba_attr_labels.index(attr) for attr in hair_color_attr]
    hair_color_idx_tensor = torch.tensor(hair_color_idx_ls)
    print("hair color index::", hair_color_idx_tensor)

    hair_style_attr = ["Straight_Hair", "Wavy_Hair"]
    hair_style_idx_ls = [all_celeba_attr_labels.index(attr) for attr in hair_style_attr]
    hair_style_idx_tensor = torch.tensor(hair_style_idx_ls)
    print("hair style index::", hair_style_idx_tensor)

    all_images = train_images + valid_images + test_images
    all_origin_labels = torch.cat([train_labels, valid_labels, test_labels])
    all_attr_labels = torch.cat(
        [train_attr_labels, valid_attr_labels, test_attr_labels]
    )

    specified_combination_count = 100

    color_style_combination_ids_mappings = dict()

    all_combination_id_ls = []

    all_labels = []

    label_idx = 0
    for color_idx in hair_color_idx_ls:
        for style_idx in hair_style_idx_ls:
            curr_combination_ids = torch.nonzero(
                torch.logical_and(
                    all_attr_labels[:, color_idx] + all_attr_labels[:, style_idx] >= 2,
                    torch.sum(all_attr_labels[:, hair_color_idx_tensor], dim=-1)
                    + torch.sum(all_attr_labels[:, hair_style_idx_tensor], dim=-1)
                    == 2,
                )
            )[:, 0]
            # curr_labels = all_labels[curr_combination_ids]
            # curr_unique_labels = torch.unique(curr_labels)
            # curr_unique_label_count = torch.sum(curr_unique_labels.view(-1,1) == curr_labels.view(1,-1), dim=0)
            curr_labels = all_origin_labels[curr_combination_ids]
            curr_unique_labels = torch.unique(curr_labels)
            curr_unique_label_count = torch.sum(
                curr_unique_labels.view(-1, 1) == curr_labels.view(1, -1), dim=1
            )
            max_label = curr_unique_labels[torch.argmax(curr_unique_label_count)]
            curr_combination_ids = curr_combination_ids[
                torch.nonzero(curr_labels == max_label)[:, 0]
            ]
            print("origin label::", all_origin_labels[curr_combination_ids])

            curr_combination_count = len(curr_combination_ids)
            if specified_combination_count < curr_combination_count:
                rand_combination_ids = torch.randperm(curr_combination_count)[
                    :specified_combination_count
                ]
                curr_combination_ids = curr_combination_ids[rand_combination_ids]
            color_style_combination_ids_mappings[(color_idx, style_idx)] = (
                curr_combination_ids
            )
            all_labels.extend([label_idx] * len(curr_combination_ids))
            print(
                all_celeba_attr_labels[color_idx],
                all_celeba_attr_labels[style_idx],
                len(curr_combination_ids),
                label_idx,
            )
            label_idx += 1
            all_combination_id_ls.append(curr_combination_ids)

    all_combination_id_ls = torch.cat(all_combination_id_ls)
    random_idx = torch.randperm(len(all_combination_id_ls))
    all_label_tensor = torch.tensor(all_labels)

    train_sample_ids = all_combination_id_ls[random_idx[: int(len(random_idx) * 0.7)]]
    valid_sample_ids = all_combination_id_ls[
        random_idx[int(len(random_idx) * 0.7) : int(len(random_idx) * 0.8)]
    ]
    test_sample_ids = all_combination_id_ls[random_idx[int(len(random_idx) * 0.8) :]]

    train_labels = all_label_tensor[random_idx[: int(len(random_idx) * 0.7)]]
    valid_labels = all_label_tensor[
        random_idx[int(len(random_idx) * 0.7) : int(len(random_idx) * 0.8)]
    ]
    test_labels = all_label_tensor[random_idx[int(len(random_idx) * 0.8) :]]

    # color_style_combination_ls = list(color_style_combination_ids_mappings.keys())
    # random.shuffle(color_style_combination_ls)
    # train_combination_ls = color_style_combination_ls[0:-2]
    # valid_combination_ls = color_style_combination_ls[-2:-1]
    # test_combination_ls = color_style_combination_ls[-1:]

    # all_combination_id_ls = []
    # for combination_idx in train_combination_ls:
    #     curr_combination_ids = color_style_combination_ids_mappings[combination_idx]
    #     all_combination_id_ls.append(curr_combination_ids)

    # train_sample_ids = torch.cat(all_combination_id_ls)

    # valid_sample_ids = color_style_combination_ids_mappings[valid_combination_ls[0]]

    # test_sample_ids = color_style_combination_ids_mappings[test_combination_ls[0]]

    train_images = [all_images[idx] for idx in train_sample_ids.tolist()]
    valid_images = [all_images[idx] for idx in valid_sample_ids.tolist()]
    test_images = [all_images[idx] for idx in test_sample_ids.tolist()]

    # train_labels = all_labels[train_sample_ids]
    # valid_labels = all_labels[valid_sample_ids]
    # test_labels = all_labels[test_sample_ids]

    train_attr_labels = all_attr_labels[train_sample_ids]
    valid_attr_labels = all_attr_labels[valid_sample_ids]
    test_attr_labels = all_attr_labels[test_sample_ids]

    # train_idx = torch.nonzero(train_attr_labels[:,hair_color_idx_ls].sum(dim=-1) == 1)[:,0]
    # valid_idx = torch.nonzero(valid_attr_labels[:,hair_color_idx_ls].sum(dim=-1) == 1)[:,0]
    # test_idx = torch.nonzero(test_attr_labels[:,hair_color_idx_ls].sum(dim=-1) == 1)[:,0]

    # label_idx_ls = torch.tensor(list(range(10)))
    # train_idx = torch.nonzero(train_labels.view(-1,1) == label_idx_ls.view(1,-1))[:,0]
    # valid_idx = torch.nonzero(valid_labels.view(-1,1) == label_idx_ls.view(1,-1))[:,0]
    # test_idx = torch.nonzero(test_labels.view(-1,1) == label_idx_ls.view(1,-1))[:,0]
    # train_images = [train_images[idx] for idx in train_idx]
    # valid_images = [valid_images[idx] for idx in valid_idx]
    # test_images = [test_images[idx] for idx in test_idx]
    print("train_images::", len(train_images))
    print("valid_images::", len(valid_images))
    print("test_images::", len(test_images))
    print("unique train labels::", len(torch.unique(train_labels)))
    print("unique valid labels::", len(torch.unique(valid_labels)))
    print("unique test labels::", len(torch.unique(test_labels)))
    return (
        train_images,
        valid_images,
        test_images,
        train_labels,
        valid_labels,
        test_labels,
        train_attr_labels,
        valid_attr_labels,
        test_attr_labels,
    )


# def subset_cub2_0(train_images, valid_images, test_images, train_attr_labels, valid_attr_labels, test_attr_labels, train_labels, valid_labels, test_labels, random=False):
#     # hair_color_attr = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"]
#     # hair_color_idx_ls = [all_celeba_attr_labels.index(attr) for attr in hair_color_attr]
#     hair_color_idx_ls = [250, 261, 260]
#     hair_color_name_ls = ["brown", "white", "black"]
#     hair_color_idx_tensor = torch.tensor(hair_color_idx_ls) - 1

#     # hair_style_attr = ["Straight_Hair","Wavy_Hair"]
#     # hair_style_idx_ls = [all_celeba_attr_labels.index(attr) for attr in hair_style_attr]
#     hair_style_idx_ls = [219, 221, 218]
#     hair_style_name_ls = ["small", "medium", "large"]
#     hair_style_idx_tensor = torch.tensor(hair_style_idx_ls) - 1


#     all_images = train_images + valid_images + test_images
#     all_origin_labels = torch.cat([train_labels, valid_labels, test_labels])
#     all_attr_labels = torch.cat([train_attr_labels, valid_attr_labels, test_attr_labels])

#     if random:
#         specified_combination_count = 100
#     else:
#         specified_combination_count = 10

#     color_style_combination_ids_mappings = dict()

#     all_combination_id_ls = []

#     all_labels = []

#     label_idx = 0
#     hair_color_idx_ls = hair_color_idx_tensor.tolist()
#     hair_style_idx_ls = hair_style_idx_tensor.tolist()
#     for color_idx in hair_color_idx_ls:
#         for style_idx in hair_style_idx_ls:
#             curr_combination_ids = torch.nonzero(torch.logical_and(all_attr_labels[:,color_idx] + all_attr_labels[:,style_idx] >= 2, torch.sum(all_attr_labels[:,hair_color_idx_tensor],dim=-1) + torch.sum(all_attr_labels[:,hair_style_idx_tensor], dim=-1) == 2))[:,0]
#             # curr_combination_ids = torch.nonzero(all_attr_labels[:,color_idx] + all_attr_labels[:,style_idx] >= 2)[:,0]
#             if not random:
#                 curr_labels = all_origin_labels[curr_combination_ids]
#                 curr_unique_labels = torch.unique(curr_labels)
#                 curr_unique_label_count = torch.sum(curr_unique_labels.view(-1,1) == curr_labels.view(1,-1), dim=1)
#                 max_label_ls = torch.topk(curr_unique_label_count, k=3)[1]
#                 max_label = curr_unique_labels[max_label_ls]
#                 # curr_combination_ids = curr_combination_ids[torch.nonzero(curr_labels == max_label)[:,0]]
#                 curr_combination_ids = curr_combination_ids[torch.sum(curr_labels.view(-1,1) == max_label.view(1,-1), dim=1).nonzero().view(-1)]
#                 print("origin label::", all_origin_labels[curr_combination_ids].sort()[0])
#                 curr_combination_count = len(curr_combination_ids)
#                 if specified_combination_count < curr_combination_count:
#                     # rand_combination_ids = torch.randperm(curr_combination_count)[:specified_combination_count]
#                     # curr_combination_ids = curr_combination_ids[rand_combination_ids]
#                     curr_combination_ids = curr_combination_ids[:specified_combination_count]
#             else:
#                 curr_combination_count = len(curr_combination_ids)
#                 if specified_combination_count < curr_combination_count:
#                     rand_combination_ids = torch.randperm(curr_combination_count)[:specified_combination_count]
#                     curr_combination_ids = curr_combination_ids[rand_combination_ids]
#             color_style_combination_ids_mappings[(color_idx, style_idx)] = curr_combination_ids
#             all_labels.extend([label_idx]*len(curr_combination_ids))
#             print(color_idx + 1, style_idx + 1, len(curr_combination_ids), label_idx)
#             label_idx += 1
#             all_combination_id_ls.append(curr_combination_ids)

#     all_combination_id_ls = torch.cat(all_combination_id_ls)
#     random_idx = torch.randperm(len(all_combination_id_ls))
#     all_label_tensor = torch.tensor(all_labels)

#     train_sample_ids = all_combination_id_ls[random_idx[:int(len(random_idx)*0.6)]]
#     valid_sample_ids = all_combination_id_ls[random_idx[int(len(random_idx)*0.6):int(len(random_idx)*0.8)]]
#     test_sample_ids = all_combination_id_ls[random_idx[int(len(random_idx)*0.8):]]

#     train_labels = all_label_tensor[random_idx[:int(len(random_idx)*0.6)]]
#     valid_labels = all_label_tensor[random_idx[int(len(random_idx)*0.6):int(len(random_idx)*0.8)]]
#     test_labels = all_label_tensor[random_idx[int(len(random_idx)*0.8):]]

#     # train_labels = all_origin_labels[train_sample_ids]
#     # valid_labels = all_origin_labels[valid_sample_ids]
#     # test_labels = all_origin_labels[test_sample_ids]

#     # color_style_combination_ls = list(color_style_combination_ids_mappings.keys())
#     # random.shuffle(color_style_combination_ls)
#     # train_combination_ls = color_style_combination_ls[0:-2]
#     # valid_combination_ls = color_style_combination_ls[-2:-1]
#     # test_combination_ls = color_style_combination_ls[-1:]

#     # all_combination_id_ls = []
#     # for combination_idx in train_combination_ls:
#     #     curr_combination_ids = color_style_combination_ids_mappings[combination_idx]
#     #     all_combination_id_ls.append(curr_combination_ids)

#     # train_sample_ids = torch.cat(all_combination_id_ls)

#     # valid_sample_ids = color_style_combination_ids_mappings[valid_combination_ls[0]]

#     # test_sample_ids = color_style_combination_ids_mappings[test_combination_ls[0]]

#     train_images = [all_images[idx] for idx in train_sample_ids.tolist()]
#     valid_images = [all_images[idx] for idx in valid_sample_ids.tolist()]
#     test_images = [all_images[idx] for idx in test_sample_ids.tolist()]

#     # train_labels = all_labels[train_sample_ids]
#     # valid_labels = all_labels[valid_sample_ids]
#     # test_labels = all_labels[test_sample_ids]

#     train_attr_labels = all_attr_labels[train_sample_ids][:, torch.cat([hair_color_idx_tensor, hair_style_idx_tensor], dim=-1)]
#     valid_attr_labels = all_attr_labels[valid_sample_ids][:, torch.cat([hair_color_idx_tensor, hair_style_idx_tensor], dim=-1)]
#     test_attr_labels = all_attr_labels[test_sample_ids][:, torch.cat([hair_color_idx_tensor, hair_style_idx_tensor], dim=-1)]

#     # train_idx = torch.nonzero(train_attr_labels[:,hair_color_idx_ls].sum(dim=-1) == 1)[:,0]
#     # valid_idx = torch.nonzero(valid_attr_labels[:,hair_color_idx_ls].sum(dim=-1) == 1)[:,0]
#     # test_idx = torch.nonzero(test_attr_labels[:,hair_color_idx_ls].sum(dim=-1) == 1)[:,0]


#     # label_idx_ls = torch.tensor(list(range(10)))
#     # train_idx = torch.nonzero(train_labels.view(-1,1) == label_idx_ls.view(1,-1))[:,0]
#     # valid_idx = torch.nonzero(valid_labels.view(-1,1) == label_idx_ls.view(1,-1))[:,0]
#     # test_idx = torch.nonzero(test_labels.view(-1,1) == label_idx_ls.view(1,-1))[:,0]
#     # train_images = [train_images[idx] for idx in train_idx]
#     # valid_images = [valid_images[idx] for idx in valid_idx]
#     # test_images = [test_images[idx] for idx in test_idx]
#     print("train_images::", len(train_images))
#     print("valid_images::", len(valid_images))
#     print("test_images::", len(test_images))
#     print("unique train labels::", len(torch.unique(train_labels)))
#     print("unique valid labels::", len(torch.unique(valid_labels)))
#     print("unique test labels::", len(torch.unique(test_labels)))
#     name_ls = hair_color_name_ls + hair_style_name_ls
#     return train_images, valid_images, test_images, train_labels, valid_labels, test_labels, train_attr_labels, valid_attr_labels, test_attr_labels, name_ls


def transforming_origin_labels_to_cg_labels(
    all_origin_labels, cg_species_label_id_mappings, species_label_id_mappings
):
    species_class_ls = set(cg_species_label_id_mappings.keys())
    species_class_ls = list(species_class_ls)
    species_class_ls.sort()
    # species_class_id_ls = [idx for idx in range(len(species_class_ls))]
    transformed_cg_species_label_id_mappings = {
        key - 1: species_class_ls.index(species_label_id_mappings[key])
        for key in species_label_id_mappings
    }
    all_transformed_labels = torch.tensor(
        [
            transformed_cg_species_label_id_mappings[all_origin_labels[idx].item()]
            for idx in range(len(all_origin_labels))
        ]
    )
    return all_transformed_labels


def subset_cub2_0(
    data_dir,
    train_images,
    valid_images,
    test_images,
    train_attr_labels,
    valid_attr_labels,
    test_attr_labels,
    train_labels,
    valid_labels,
    test_labels,
    random=False,
):
    # hair_color_attr = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"]
    # hair_color_idx_ls = [all_celeba_attr_labels.index(attr) for attr in hair_color_attr]
    hair_color_idx_ls = [250, 261, 260]
    hair_color_name_ls = ["brown", "white", "black"]
    hair_color_idx_tensor = torch.tensor(hair_color_idx_ls) - 1

    # hair_style_attr = ["Straight_Hair","Wavy_Hair"]
    # hair_style_idx_ls = [all_celeba_attr_labels.index(attr) for attr in hair_style_attr]
    hair_style_idx_ls = [219, 221, 218]
    hair_style_name_ls = ["small", "medium", "large"]
    hair_style_idx_tensor = torch.tensor(hair_style_idx_ls) - 1

    (
        cg_species_label_set,
        cg_species_label_id_mappings,
        species_label_id_mappings,
        multi_species_set,
    ) = obtain_all_cg_species_label_ls(os.path.join(data_dir, "classes.txt"))

    all_images = train_images + valid_images + test_images
    all_origin_labels = torch.cat([train_labels, valid_labels, test_labels])
    all_transformed_labels = transforming_origin_labels_to_cg_labels(
        all_origin_labels, cg_species_label_id_mappings, species_label_id_mappings
    )
    all_attr_labels = torch.cat(
        [train_attr_labels, valid_attr_labels, test_attr_labels]
    )

    if random:
        specified_combination_count = 100
    else:
        specified_combination_count = 30

    color_style_combination_ids_mappings = dict()

    all_combination_id_ls = []

    ood_all_combination_id_ls = []

    all_labels = []

    ood_labels = []

    label_idx = 0
    ood_label_idx = len(hair_color_idx_ls) * len(hair_style_idx_ls) - 2
    hair_color_idx_ls = hair_color_idx_tensor.tolist()
    hair_style_idx_ls = hair_style_idx_tensor.tolist()
    for color_idx in hair_color_idx_ls:
        for style_idx in hair_style_idx_ls:
            curr_combination_ids = torch.nonzero(
                torch.logical_and(
                    all_attr_labels[:, color_idx] + all_attr_labels[:, style_idx] >= 2,
                    torch.sum(all_attr_labels[:, hair_color_idx_tensor], dim=-1)
                    + torch.sum(all_attr_labels[:, hair_style_idx_tensor], dim=-1)
                    == 2,
                )
            )[:, 0]
            # curr_combination_ids = torch.nonzero(all_attr_labels[:,color_idx] + all_attr_labels[:,style_idx] >= 2)[:,0]
            if not random:
                curr_labels = all_transformed_labels[curr_combination_ids]
                curr_unique_labels = torch.unique(curr_labels)
                curr_unique_label_count = torch.sum(
                    curr_unique_labels.view(-1, 1) == curr_labels.view(1, -1), dim=1
                )
                max_label_ls = torch.topk(curr_unique_label_count, k=3)[1]
                max_label = curr_unique_labels[max_label_ls]
                # curr_combination_ids = curr_combination_ids[torch.nonzero(curr_labels == max_label)[:,0]]
                curr_combination_id_ls = []
                for max_l_idx in range(max_label.view(-1).shape[0]):
                    curr_combination_id_ls.append(
                        curr_combination_ids[
                            torch.sum(
                                curr_labels.view(-1, 1)
                                == max_label[max_l_idx].view(1, -1),
                                dim=1,
                            )
                            .nonzero()
                            .view(-1)
                        ]
                    )
                curr_combination_ids = torch.cat(curr_combination_id_ls)
                print("origin label::", all_transformed_labels[curr_combination_ids])
                curr_combination_count = len(curr_combination_ids)
                if specified_combination_count < curr_combination_count:
                    # rand_combination_ids = torch.randperm(curr_combination_count)[:specified_combination_count]
                    # curr_combination_ids = curr_combination_ids[rand_combination_ids]
                    curr_combination_ids = curr_combination_ids[
                        :specified_combination_count
                    ]
            else:
                curr_combination_count = len(curr_combination_ids)
                if specified_combination_count < curr_combination_count:
                    rand_combination_ids = torch.randperm(curr_combination_count)[
                        :specified_combination_count
                    ]
                    curr_combination_ids = curr_combination_ids[rand_combination_ids]
            color_style_combination_ids_mappings[(color_idx, style_idx)] = (
                curr_combination_ids
            )

            if (
                color_idx == hair_color_idx_ls[0] and style_idx == hair_style_idx_ls[0]
            ) or (
                color_idx == hair_color_idx_ls[-1]
                and style_idx == hair_style_idx_ls[-1]
            ):
                ood_all_combination_id_ls.append(curr_combination_ids)
                ood_labels.extend([ood_label_idx] * len(curr_combination_ids))
                print(
                    color_idx + 1,
                    style_idx + 1,
                    len(curr_combination_ids),
                    ood_label_idx,
                )
                ood_label_idx += 1
            else:
                print(
                    color_idx + 1, style_idx + 1, len(curr_combination_ids), label_idx
                )
                all_combination_id_ls.append(curr_combination_ids)
                all_labels.extend([label_idx] * len(curr_combination_ids))
                label_idx += 1

    all_combination_id_ls = torch.cat(all_combination_id_ls)
    ood_all_combination_id_ls = torch.cat(ood_all_combination_id_ls)
    random_idx = torch.randperm(len(all_combination_id_ls))
    all_label_tensor = torch.tensor(all_labels)
    ood_label_tensor = torch.tensor(ood_labels)

    train_sample_ids = all_combination_id_ls[random_idx[: int(len(random_idx) * 0.6)]]
    valid_sample_ids = all_combination_id_ls[
        random_idx[int(len(random_idx) * 0.6) : int(len(random_idx) * 0.8)]
    ]
    test_sample_ids = all_combination_id_ls[random_idx[int(len(random_idx) * 0.8) :]]

    train_labels = all_label_tensor[random_idx[: int(len(random_idx) * 0.6)]]
    valid_labels = all_label_tensor[
        random_idx[int(len(random_idx) * 0.6) : int(len(random_idx) * 0.8)]
    ]
    test_labels = all_label_tensor[random_idx[int(len(random_idx) * 0.8) :]]

    # train_labels = all_origin_labels[train_sample_ids]
    # valid_labels = all_origin_labels[valid_sample_ids]
    # test_labels = all_origin_labels[test_sample_ids]

    # color_style_combination_ls = list(color_style_combination_ids_mappings.keys())
    # random.shuffle(color_style_combination_ls)
    # train_combination_ls = color_style_combination_ls[0:-2]
    # valid_combination_ls = color_style_combination_ls[-2:-1]
    # test_combination_ls = color_style_combination_ls[-1:]

    # all_combination_id_ls = []
    # for combination_idx in train_combination_ls:
    #     curr_combination_ids = color_style_combination_ids_mappings[combination_idx]
    #     all_combination_id_ls.append(curr_combination_ids)

    # train_sample_ids = torch.cat(all_combination_id_ls)

    # valid_sample_ids = color_style_combination_ids_mappings[valid_combination_ls[0]]

    # test_sample_ids = color_style_combination_ids_mappings[test_combination_ls[0]]

    train_images = [all_images[idx] for idx in train_sample_ids.tolist()]
    valid_images = [all_images[idx] for idx in valid_sample_ids.tolist()]
    test_images = [all_images[idx] for idx in test_sample_ids.tolist()]
    ood_images = [all_images[idx] for idx in ood_all_combination_id_ls.tolist()]

    # train_labels = all_labels[train_sample_ids]
    # valid_labels = all_labels[valid_sample_ids]
    # test_labels = all_labels[test_sample_ids]

    train_attr_labels = all_attr_labels[train_sample_ids][
        :, torch.cat([hair_color_idx_tensor, hair_style_idx_tensor], dim=-1)
    ]
    valid_attr_labels = all_attr_labels[valid_sample_ids][
        :, torch.cat([hair_color_idx_tensor, hair_style_idx_tensor], dim=-1)
    ]
    test_attr_labels = all_attr_labels[test_sample_ids][
        :, torch.cat([hair_color_idx_tensor, hair_style_idx_tensor], dim=-1)
    ]
    ood_attr_labels = all_attr_labels[ood_all_combination_id_ls][
        :, torch.cat([hair_color_idx_tensor, hair_style_idx_tensor], dim=-1)
    ]
    # train_idx = torch.nonzero(train_attr_labels[:,hair_color_idx_ls].sum(dim=-1) == 1)[:,0]
    # valid_idx = torch.nonzero(valid_attr_labels[:,hair_color_idx_ls].sum(dim=-1) == 1)[:,0]
    # test_idx = torch.nonzero(test_attr_labels[:,hair_color_idx_ls].sum(dim=-1) == 1)[:,0]

    # label_idx_ls = torch.tensor(list(range(10)))
    # train_idx = torch.nonzero(train_labels.view(-1,1) == label_idx_ls.view(1,-1))[:,0]
    # valid_idx = torch.nonzero(valid_labels.view(-1,1) == label_idx_ls.view(1,-1))[:,0]
    # test_idx = torch.nonzero(test_labels.view(-1,1) == label_idx_ls.view(1,-1))[:,0]
    # train_images = [train_images[idx] for idx in train_idx]
    # valid_images = [valid_images[idx] for idx in valid_idx]
    # test_images = [test_images[idx] for idx in test_idx]
    print("train_images::", len(train_images))
    print("valid_images::", len(valid_images))
    print("test_images::", len(test_images))
    print("unique train labels::", len(torch.unique(train_labels)))
    print("unique valid labels::", len(torch.unique(valid_labels)))
    print("unique test labels::", len(torch.unique(test_labels)))
    name_ls = hair_color_name_ls + hair_style_name_ls
    return (
        train_images,
        valid_images,
        test_images,
        train_labels,
        valid_labels,
        test_labels,
        train_attr_labels,
        valid_attr_labels,
        test_attr_labels,
        (name_ls, ood_images, ood_labels, ood_attr_labels),
    )


def prepare_ham_data(base_dir="/data2/wuyinjun/ham/"):
    if (
        os.path.exists(os.path.join(base_dir, "train_dl"))
        and os.path.exists(os.path.join(base_dir, "valid_dl"))
        and os.path.exists(os.path.join(base_dir, "test_dl"))
    ):
        train_dl = torch.load(os.path.join(base_dir, "train_dl"))
        valid_dl = torch.load(os.path.join(base_dir, "valid_dl"))
        test_dl = torch.load(os.path.join(base_dir, "test_dl"))
    else:
        # iamge_files = read_all_ham_image_files(os.path.join(base_dir, "jpg"))
        # labels= read_ham_labels(base_dir)
        (
            train_images,
            train_labels,
            valid_images,
            valid_labels,
            test_images,
            test_labels,
        ) = get_ham_train_test_split(base_dir)
        train_dl = load_ham_data(
            train_images, train_labels, batch_size=128, is_training=True
        )
        valid_dl = load_ham_data(
            valid_images, valid_labels, batch_size=128, is_training=False
        )
        test_dl = load_ham_data(
            test_images, test_labels, batch_size=128, is_training=False
        )
        # torch.save(train_dl, os.path.join(base_dir, "train_dl"))
        # torch.save(valid_dl, os.path.join(base_dir, "valid_dl"))
        # torch.save(test_dl, os.path.join(base_dir, "test_dl"))
    # ham_Dataset(train_images, train_labels, valid_images, valid_labels, test_images, test_labels, use_attr=use_attr, get_bounding_box=get_bounding_box, base_dir=base_dir)
    return train_dl, valid_dl, test_dl


def get_image_data_ls(
    args,
    full_data_dir,
    root_dir,
    use_attr=False,
    get_bounding_box=False,
    get_bg_info=False,
):
    train_attr_labels, valid_attr_labels, test_attr_labels = None, None, None
    other_info = None
    if args.dataset_name.lower().startswith("cub"):
        config_file = os.path.join(root_dir, "CUB_config.yaml")
        with open(config_file) as f:
            loaded_config = yaml.load(f, Loader=yaml.FullLoader)

        print(loaded_config)
        train_dl, valid_dl, test_dl = prepare_cub_data(
            root_dir,
            loaded_config,
            use_attr=use_attr,
            get_bounding_box=get_bounding_box,
            base_dir=full_data_dir,
        )
    elif args.dataset_name.lower().startswith("celeba"):
        config_file = os.path.join(root_dir, "celeba_config.yaml")
        with open(config_file) as f:
            loaded_config = yaml.load(f, Loader=yaml.FullLoader)

        print(loaded_config)
        train_dl, valid_dl, test_dl = prepare_celeba_data(
            root_dir,
            loaded_config,
            use_attr=use_attr,
            get_bounding_box=get_bounding_box,
            base_dir=full_data_dir,
        )

    elif args.dataset_name.lower() == "controlled_cub":
        config_file = os.path.join(root_dir, "CUB_config.yaml")
        with open(config_file) as f:
            loaded_config = yaml.load(f, Loader=yaml.FullLoader)

        print(loaded_config)
        train_dl, valid_dl, test_dl = prepare_cub_data(
            root_dir,
            loaded_config,
            use_attr=use_attr,
            get_bounding_box=get_bounding_box,
            base_dir=full_data_dir,
            controlled_data=True,
        )

    elif args.dataset_name.lower() == "ham":
        train_dl, valid_dl, test_dl = prepare_ham_data(base_dir=full_data_dir)

    elif args.dataset_name.lower() == "imagenetwf":
        full_data_dir = os.path.join(full_data_dir, "imagewoof2")
        train_dl, valid_dl, test_dl = prepare_imagenet_data(base_dir=full_data_dir)

    if train_dl.dataset.use_attr:
        if train_dl.dataset.get_bounding_box:
            train_data_file = os.path.join(full_data_dir, "train_data_attr_bbox.pkl")
            valid_data_file = os.path.join(full_data_dir, "valid_data_attr_bbox.pkl")
            test_data_file = os.path.join(full_data_dir, "test_data_attr_bbox.pkl")
        else:
            train_data_file = os.path.join(full_data_dir, "train_data_attr.pkl")
            valid_data_file = os.path.join(full_data_dir, "valid_data_attr.pkl")
            test_data_file = os.path.join(full_data_dir, "test_data_attr.pkl")
    elif train_dl.dataset.get_bounding_box:
        train_data_file = os.path.join(full_data_dir, "train_data_bbox.pkl")
        valid_data_file = os.path.join(full_data_dir, "valid_data_bbox.pkl")
        test_data_file = os.path.join(full_data_dir, "test_data_bbox.pkl")
    else:
        train_data_file = os.path.join(full_data_dir, "train_data.pkl")
        valid_data_file = os.path.join(full_data_dir, "valid_data.pkl")
        test_data_file = os.path.join(full_data_dir, "test_data.pkl")

    if not os.path.exists(train_data_file):
        all_train_info = get_image_info_ls(train_dl)
        save_objs(all_train_info, train_data_file)
    else:
        # train_images, train_labels
        all_train_info = load_objs(train_data_file)

    if not os.path.exists(valid_data_file):
        # valid_images, valid_labels = get_image_info_ls(valid_dl)
        all_valid_info = get_image_info_ls(valid_dl)
        save_objs(all_valid_info, valid_data_file)
    else:
        # valid_images, valid_labels = load_objs(valid_data_file)
        all_valid_info = load_objs(valid_data_file)

    if not os.path.exists(test_data_file):
        # test_images, test_labels = get_image_info_ls(test_dl)
        all_test_info = get_image_info_ls(test_dl)
        save_objs(all_test_info, test_data_file)
    else:
        # test_images, test_labels = load_objs(test_data_file)
        all_test_info = load_objs(test_data_file)

    if train_dl.dataset.use_attr:
        if train_dl.dataset.get_bounding_box:
            train_images, train_labels, train_attr_labels, train_bboxes = all_train_info
            valid_images, valid_labels, valid_attr_labels, valid_bboxes = all_valid_info
            test_images, test_labels, test_attr_labels, test_bboxes = all_test_info
        else:
            train_images, train_labels, train_attr_labels = all_train_info
            valid_images, valid_labels, valid_attr_labels = all_valid_info
            test_images, test_labels, test_attr_labels = all_test_info
    elif train_dl.dataset.get_bounding_box:
        train_images, train_labels, train_bboxes = all_train_info
        valid_images, valid_labels, valid_bboxes = all_valid_info
        test_images, test_labels, test_bboxes = all_test_info
    else:
        train_images, train_labels = all_train_info
        valid_images, valid_labels = all_valid_info
        test_images, test_labels = all_test_info

    # if train_dl.dataset.get_bounding_box:
    #     # visualize_bbox_in_image(train_images[0], train_bboxes[0])
    #     print("get train image bounding box::")
    #     train_images = get_all_sub_images_by_bounding_box(train_images, train_bboxes)

    #     print("get valid image bounding box::")
    #     valid_images = get_all_sub_images_by_bounding_box(valid_images, valid_bboxes)

    #     print("get test image bounding box::")
    #     test_images = get_all_sub_images_by_bounding_box(test_images, test_bboxes)

    if args.dataset_name.lower().startswith("celeba"):
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
        ) = select_common_labels_celeba(
            train_images,
            valid_images,
            test_images,
            train_labels,
            valid_labels,
            test_labels,
            train_attr_labels,
            valid_attr_labels,
            test_attr_labels,
        )
        if args.dataset_name.lower() == "celeba_subset":
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
            ) = subset_celeba(
                train_images,
                valid_images,
                test_images,
                train_labels,
                valid_labels,
                test_labels,
                train_attr_labels,
                valid_attr_labels,
                test_attr_labels,
            )
        elif args.dataset_name == "celeba_subset2":
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
            ) = subset_celeba2_0(
                train_images,
                valid_images,
                test_images,
                train_attr_labels,
                valid_attr_labels,
                test_attr_labels,
                train_labels,
                valid_labels,
                test_labels,
            )

    if args.dataset_name.startswith("cub"):
        if args.dataset_name == "cub_subset":
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
            ) = subset_cub2_0(
                full_data_dir,
                train_images,
                valid_images,
                test_images,
                train_attr_labels,
                valid_attr_labels,
                test_attr_labels,
                train_labels,
                valid_labels,
                test_labels,
            )

    if args.dataset_name.lower() == "controlled_cub" and get_bg_info:
        train_bg_labels = get_all_bg_labels(train_dl)
        valid_bg_labels = get_all_bg_labels(valid_dl)
        test_bg_labels = get_all_bg_labels(test_dl)
        other_info = (train_bg_labels, valid_bg_labels, test_bg_labels)

    return (
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
    )


def get_concept_activations_per_sample0(image_embs, learned_concepts):
    concept_activations = torch.mm(learned_concepts, torch.t(image_embs)) / torch.mm(
        torch.norm(learned_concepts, dim=-1).view(-1, 1),
        torch.norm(image_embs, dim=-1).view(1, -1),
    )

    return concept_activations.t()


def get_concept_activations_per_sample0_by_images(
    method_class, image_embs, image_per_patch
):
    concept_activations_ls = []
    image_per_patch = torch.tensor(image_per_patch)
    unique_image_count = len(torch.unique(image_per_patch))
    for idx in tqdm(range(unique_image_count)):
        image_per_patch_idx = (image_per_patch == idx).nonzero().view(-1)
        activations = method_class.compute_sample_activations(
            image_embs[image_per_patch_idx]
        )
        # activations = get_concept_activations_per_sample0(image_embs[image_per_patch_idx], learned_concepts)
        concept_activations_ls.append(activations)

    return concept_activations_ls


def set_rand_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def aggregate_attr_info(attr_file):
    attr_name_ids_mapping = dict()
    with open(attr_file) as f:
        for line in f:
            attr_idx, attr_full_name = line.split(" ")
            attr_name = attr_full_name.split("::")[-1].strip()
            if attr_name not in attr_name_ids_mapping:
                attr_name_ids_mapping[attr_name] = []
            attr_name_ids_mapping[attr_name].append(int(attr_idx) - 1)
    return attr_name_ids_mapping


def obtain_all_species_label_ls(image_class_file):
    cg_species_label_set = set()
    cg_species_label_id_mappings = dict()
    species_label_id_mappings = dict()
    with open(image_class_file) as f:
        for line in f:
            label_idx = int(line.split(" ")[0].strip())
            species_label = line.split(" ")[1].strip()
            cg_species_label = species_label.split(".")[1]
            cg_species_label_set.add(species_label)
            if cg_species_label not in cg_species_label_id_mappings:
                cg_species_label_id_mappings[cg_species_label] = []
            cg_species_label_id_mappings[cg_species_label].append(label_idx)
            species_label_id_mappings[label_idx] = cg_species_label
    return (
        list(cg_species_label_set),
        cg_species_label_id_mappings,
        species_label_id_mappings,
    )


def obtain_all_cg_species_label_ls(image_class_file):
    cg_species_label_set = set()
    cg_species_label_id_mappings = dict()
    species_label_id_mappings = dict()
    with open(image_class_file) as f:
        for line in f:
            label_idx = int(line.split(" ")[0].strip())
            species_label = line.split(" ")[1].strip()
            cg_species_label = species_label.split(".")[1].split("_")[-1]
            cg_species_label_set.add(cg_species_label)
            if cg_species_label not in cg_species_label_id_mappings:
                cg_species_label_id_mappings[cg_species_label] = []
            cg_species_label_id_mappings[cg_species_label].append(label_idx)
            species_label_id_mappings[label_idx] = cg_species_label

    multi_species_set = set()
    for species in list(cg_species_label_id_mappings.keys()):
        if len(cg_species_label_id_mappings[species]) > 5:
            # del cg_species_label_id_mappings[species]
            # cg_species_label_set.remove(species)
            multi_species_set.add(species)

    # for label_idx in species_label_id_mappings:
    #     if species_label_id_mappings[label_idx] in removed_species_ls:
    #         del species_label_id_mappings[label_idx]

    return (
        list(cg_species_label_set),
        cg_species_label_id_mappings,
        species_label_id_mappings,
        multi_species_set,
    )


def obtain_species_label_ls(image_class_file):
    # cg_species_label_set = set()
    # cg_species_label_id_mappings = dict()
    species_label_id_mappings = dict()
    with open(image_class_file) as f:
        for line in f:
            label_idx = int(line.split(" ")[0].strip())
            species_label = line.split(" ")[1].strip()
            # cg_species_label = species_label.split(".")[1]
            # cg_species_label_set.add(species_label)
            # if cg_species_label not in cg_species_label_id_mappings:
            #     cg_species_label_id_mappings[cg_species_label] = []
            # cg_species_label_id_mappings[cg_species_label].append(label_idx)
            species_label_id_mappings[label_idx] = species_label  # .split(".")[1]
    return species_label_id_mappings


def convert_fine_grained_label_to_coarse_grained_label(
    multi_species_set,
    fine_grained_labels,
    species_label_id_mappings,
    cg_species_label_ls,
):
    all_cg_species_label_ls = []
    all_multi_species_label_ls = set()
    for label in fine_grained_labels:
        cg_species_label = species_label_id_mappings[label]
        cg_species_label_idx = cg_species_label_ls.index(cg_species_label)
        all_cg_species_label_ls.append(cg_species_label_idx)
        if cg_species_label in multi_species_set:
            all_multi_species_label_ls.add(cg_species_label_idx)
    return torch.tensor(all_cg_species_label_ls), all_multi_species_label_ls


def obtain_fine_grained_label_ls(image_class_file):
    with open(image_class_file) as f:
        fine_grained_labels = [int(line.split(" ")[1].strip()) for line in f]
    return fine_grained_labels


def aggregate_attr_ids(attr_name_ids_mapping, attr_label_tensor):
    attr_name_ls = list(attr_name_ids_mapping.keys())
    print("all attribute names::", attr_name_ls)
    aggregate_attr_label_tensor = torch.zeros(
        [len(attr_label_tensor), len(attr_name_ls)]
    ).type(torch.bool)
    aggregate_count_attr_label_tensor = torch.zeros(
        [len(attr_label_tensor), len(attr_name_ls)]
    )
    for agg_attr_idx in range(len(attr_name_ls)):
        attr = attr_name_ls[agg_attr_idx]
        for attr_idx in attr_name_ids_mapping[attr]:
            aggregate_attr_label_tensor[:, agg_attr_idx] = torch.logical_or(
                aggregate_attr_label_tensor[:, agg_attr_idx],
                attr_label_tensor[:, attr_idx].type(torch.bool),
            )
            aggregate_count_attr_label_tensor[:, agg_attr_idx] += attr_label_tensor[
                :, attr_idx
            ]

    return (
        attr_name_ls,
        aggregate_attr_label_tensor.type(torch.float),
        aggregate_count_attr_label_tensor,
    )


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="CUB concept learning")
    parser.add_argument("--concept_count", type=int, default=100, help="config file")
    parser.add_argument("--pos_ratio", type=float, default=0.9, help="config file")
    parser.add_argument("--neg_ratio", type=float, default=0.1, help="config file")
    parser.add_argument("--lr", type=float, default=0.001, help="config file")
    parser.add_argument("--projection_size", type=int, default=50, help="config file")
    parser.add_argument(
        "--concept_num_per_attr", type=int, default=10, help="config file"
    )
    parser.add_argument("--num_attrs", type=int, default=2, help="config file")
    parser.add_argument("--epochs", type=int, default=400, help="config file")
    parser.add_argument("--seed", type=int, default=0, help="config file")
    parser.add_argument("--split_method", type=str, default="ours", help="config file")
    parser.add_argument("--dataset_name", type=str, default="CUB", help="config file")
    parser.add_argument(
        "--classification_method",
        type=str,
        default="one",
        help="config file",
        choices=["one", "two", "three"],
    )
    parser.add_argument("--cache_activations", action="store_true", help="config file")
    parser.add_argument("--eval_concepts", action="store_true", help="config file")
    parser.add_argument("--load_cache", action="store_true", help="config file")
    parser.add_argument(
        "--data_dir", type=str, default="/data2/wuyinjun/", help="config file"
    )
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--center", action="store_true", default=False)
    parser.add_argument("--whiten", action="store_true", default=False)
    parser.add_argument("--relation_learning", action="store_true", default=False)
    # parser.add_argument('--full_image', action='store_true', default=False)
    # parser.add_argument('--full_image_classification', action='store_true', default=False)
    parser.add_argument("--qualitative", action="store_true", default=False)
    parser.add_argument("--cosine_sim", action="store_true", default=False)
    parser.add_argument("--do_classification", action="store_true", default=False)
    parser.add_argument(
        "--do_classification_neural", action="store_true", default=False
    )
    parser.add_argument(
        "--existing_concept_logs", type=str, default=None, help="config file"
    )
    # --select_images
    parser.add_argument("--select_images", action="store_true", default=False)
    parser.add_argument("--cross_entropy", action="store_true", default=False)
    if args is not None:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    return args
