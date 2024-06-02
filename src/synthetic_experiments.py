from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    CLIPModel,
    ResNetModel,
    ViTModel,
)

from baselines.joint_baselines import concept_transformer_learner
from clevr_scene import (
    CLEVRSample,
    get_bow_labels,
    get_grammar_labels,
    get_images,
)
from compositionality_eval import compositional_f1, compositionality_eval
from concept_learning import ConceptLearner
from utils import (
    _batch_inference,
    cosim,
    load,
    normalize,
    save,
    whiten,
)


def load_clip(device):
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = model.eval()

    def img_forward(imgs):
        with torch.no_grad():
            return model.get_image_features(pixel_values=imgs)

    def text_forward(samples):
        with torch.no_grad():
            return model.get_text_features(
                input_ids=samples[0], attention_mask=samples[1]
            )

    return img_forward, text_forward, processor


def load_vit(device):
    image_processor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )

    model = (
        ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device).eval()
    )

    def img_forward(imgs):
        with torch.no_grad():
            return model(imgs).pooler_output

    return img_forward, None, image_processor


def load_resnet(device):
    image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetModel.from_pretrained("microsoft/resnet-50").to(device).eval()

    def img_forward(imgs):
        with torch.no_grad():
            return model(imgs).pooler_output.squeeze()

    return img_forward, None, image_processor


def load_clevr(data_dir, num_objects):
    scene_dir = (
        f"{data_dir}/{'one_object' if num_objects == 1 else 'two_object'}/scenes/*.json"
    )
    if os.path.exists(
        f"{data_dir}/{'one_object' if num_objects == 1 else 'two_object'}/scenes.pkl"
    ):
        all_scenes = load(
            f"{data_dir}/{'one_object' if num_objects == 1 else 'two_object'}/scenes.pkl"
        )
    else:
        all_scenes = glob.glob(scene_dir)
        save(
            all_scenes,
            f"{data_dir}/{'one_object' if num_objects == 1 else 'two_object'}/scenes.pkl",
        )
    print(len(all_scenes))
    two_samples = []
    for scene in all_scenes:
        with open(scene) as f:
            two_samples.append(
                CLEVRSample(
                    f"{data_dir}/{'one_object' if num_objects == 1 else 'two_object'}/images/",
                    json.load(f),
                    scene,
                )
            )
    samples = get_images(two_samples)
    labels = get_bow_labels(two_samples)

    grammar_labels, grammar_names, relation_mask = get_grammar_labels(two_samples, 2)
    print("Grammar labels:", grammar_labels[:10])
    print(grammar_names)
    return samples, labels, grammar_labels, grammar_names, relation_mask


def concept_gt_match(concepts, gt_concepts):
    """Return the max cosine similarity of each GT concept to a learned concept"""
    similarities = cosim(concepts, gt_concepts)
    return torch.max(similarities, dim=0)[0], torch.argmax(similarities, dim=0)


def concept_gt_match_labels(concept_scores, gt_concept_labels, allow_negative=True):
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
                allow_negative
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


def concept_threshold(concept_scores, gt_concept_labels):
    thresholds_best = torch.zeros(gt_concept_labels.shape[1])
    for cid in range(gt_concept_labels.shape[1]):
        fpr, tpr, thresholds = roc_curve(
            gt_concept_labels[:, cid], concept_scores[:, cid]
        )
        thresholds_best[cid] = torch.tensor(thresholds[np.argmax(tpr - fpr)])
    return thresholds_best


def get_concepts(
    args, concept_learner, n_concepts, device, embeddings, grammar_labels, seed
):
    if args.learn_method == "ours":
        if os.path.exists(
            f"{args.data_dir}/concepts_{args.dataset}_{args.model}_{args.learn_method}_{args.segment_method}_{seed}"
            f"{'_center' if args.center else ''}{'_normalize' if args.normalize else ''}{'_whiten' if args.whiten else ''}{'_ce' if args.cross_entropy else ''}{'_noreg' if args.no_reg else ''}.pkl"
        ):
            concepts, duration = load(
                f"{args.data_dir}/concepts_{args.dataset}_{args.model}_{args.learn_method}_{args.segment_method}_{seed}"
                f"{'_center' if args.center else ''}{'_normalize' if args.normalize else ''}{'_whiten' if args.whiten else ''}{'_ce' if args.cross_entropy else ''}{'_noreg' if args.no_reg else ''}.pkl"
            )
            embeddings = load(
                f"{args.data_dir}/embeddings_{args.dataset}_{args.model}_{args.learn_method}_{args.segment_method}.pkl"
            )
            attr_concepts = concepts
        else:
            # _, embeddings = concept_learner.get_patches(8, method="none")
            # if args.whiten:
            #     embeddings, mean, whiten_mat, dewhiten_mat = whiten(embeddings, return_fit=True)
            # else:
            #     if args.center:
            #         embeddings = embeddings - torch.mean(embeddings, dim=0)
            #     if args.normalize:
            #         embeddings = normalize(embeddings)

            start = time.time()
            concepts = concept_learner.learn_attribute_concepts(
                3,
                embeddings,
                n_concepts=[3, 3, 3],
                split_method="ours-subspace",
                use_leace=False,
                seed=seed,
                cross_entropy=args.cross_entropy,
                no_reg=args.no_reg,
            )
            duration = time.time() - start

            save(
                (concepts, duration),
                (
                    f"{args.data_dir}/concepts_{args.dataset}_{args.model}_{args.learn_method}_{args.segment_method}_{seed}"
                    f"{'_center' if args.center else ''}{'_normalize' if args.normalize else ''}{'_whiten' if args.whiten else ''}{'_ce' if args.cross_entropy else ''}{'_noreg' if args.no_reg else ''}.pkl"
                ),
            )
            save(
                embeddings,
                f"{args.data_dir}/embeddings_{args.dataset}_{args.model}_{args.learn_method}_{args.segment_method}.pkl",
            )
            attr_concepts = concepts
    elif args.learn_method == "ace" or args.learn_method == "ace-svm":
        concept_path = f"{args.data_dir}/attr_concepts_{args.dataset}_{args.model}_{args.learn_method}_{args.segment_method}_{seed}\
                {'_center' if args.center else ''}{'_normalize' if args.normalize else ''}{'_whiten' if args.whiten else ''}.pkl".replace(
            " ", ""
        )
        if os.path.exists(concept_path):
            attr_concepts, duration = load(concept_path)
        else:
            # _, embeddings = concept_learner.get_patches(8, method="none")
            # if args.whiten:
            #     embeddings, mean, whiten_mat, dewhiten_mat = whiten(embeddings, return_fit=True)
            # else:
            #     if args.center:
            #         embeddings = embeddings - torch.mean(embeddings, dim=0)
            #     if args.normalize:
            #         embeddings = normalize(embeddings)

            start = time.time()
            concepts = concept_learner.learn_ace_concepts(
                n_concepts,
                embeddings,
                use_svm=args.learn_method == "ace-svm",
                seed=seed,
            )
            duration = time.time() - start
            concepts = torch.tensor(concepts).float()
            # if args.whiten:
            #     concepts = concepts @ whiten_mat
            #     embeddings = (embeddings @ dewhiten_mat)
            attr_concepts = concepts
            save((attr_concepts, duration), concept_path)
    elif args.learn_method == "pca":
        concept_path = f"{args.data_dir}/attr_concepts_{args.dataset}_{args.model}_{args.learn_method}_{args.segment_method}_{seed}\
                {'_center' if args.center else ''}{'_normalize' if args.normalize else ''}{'_whiten' if args.whiten else ''}.pkl".replace(
            " ", ""
        )
        if os.path.exists(concept_path):
            attr_concepts, duration = load(concept_path)
        else:
            # _, embeddings = concept_learner.get_patches(8, method="none")
            # if args.whiten:
            #     embeddings, mean, whiten_mat, dewhiten_mat = whiten(embeddings, return_fit=True)
            # else:
            #     if args.center:
            #         embeddings = embeddings - torch.mean(embeddings, dim=0)
            #     if args.normalize:
            #         embeddings = normalize(embeddings)

            start = time.time()
            concepts = concept_learner.learn_pca_concepts(n_concepts, embeddings)
            duration = time.time() - start
            # if args.whiten:
            #     concepts = concepts @ whiten_mat
            #     embeddings = (embeddings @ dewhiten_mat)
            attr_concepts = concepts
            save((attr_concepts, duration), concept_path)
            save(
                embeddings,
                f"{args.data_dir}/embeddings_{args.dataset}_{args.model}_{args.learn_method}_{args.segment_method}.pkl",
            )
    elif args.learn_method == "dictlearn":
        _, embeddings = concept_learner.get_patches(8, method="none")
        if args.whiten:
            embeddings, mean, whiten_mat, dewhiten_mat = whiten(
                embeddings, return_fit=True
            )
        else:
            if args.center:
                embeddings = embeddings - torch.mean(embeddings, dim=0)
            if args.normalize:
                embeddings = normalize(embeddings)

        start = time.time()
        concepts = concept_learner.learn_dictlearn_concepts(
            n_concepts, embeddings, seed=seed
        )
        duration = time.time() - start
        # if args.whiten:
        #     concepts = concepts @ whiten_mat
        #     embeddings = (embeddings @ dewhiten_mat)

        attr_concepts = normalize(torch.tensor(concepts).float())
    elif args.learn_method == "seminmf":
        concept_path = f"{args.data_dir}/attr_concepts_{args.dataset}_{args.model}_{args.learn_method}_{args.segment_method}_{seed}\
                {'_center' if args.center else ''}{'_normalize' if args.normalize else ''}{'_whiten' if args.whiten else ''}.pkl".replace(
            " ", ""
        )
        if os.path.exists(concept_path):
            attr_concepts, duration = load(concept_path)
        else:
            # _, embeddings = concept_learner.get_patches(8, method="none")
            # if args.whiten:
            #     embeddings, mean, whiten_mat, dewhiten_mat = whiten(embeddings, return_fit=True)
            # else:
            #     if args.center:
            #         embeddings = embeddings - torch.mean(embeddings, dim=0)
            #     if args.normalize:
            #         embeddings = normalize(embeddings)

            start = time.time()
            concepts = concept_learner.learn_seminmf_concepts(
                n_concepts, embeddings, seed=seed
            )
            duration = time.time() - start
            # if args.whiten:
            #     concepts = concepts @ whiten_mat
            #     embeddings = (embeddings @ dewhiten_mat)

            attr_concepts = normalize(torch.tensor(concepts).float())
            save((attr_concepts, duration), concept_path)
    elif args.learn_method == "ct":
        concept_path = f"{args.data_dir}/attr_concepts_{args.dataset}_{args.model}_{args.learn_method}_{seed}\
                {'_center' if args.center else ''}{'_normalize' if args.normalize else ''}{'_whiten' if args.whiten else ''}.pkl".replace(
            " ", ""
        )
        if os.path.exists(concept_path):
            attr_concepts, duration = load(concept_path)
            embeddings = load(
                f"{args.data_dir}/train_embeddings_{args.dataset}_{args.model}_{args.learn_method}.pkl"
            )
        else:
            # _, embeddings = concept_learner.get_patches(8, method=lambda x: [[xi] for xi in x])
            test_emb = load(
                f"{args.data_dir}/sample_emb_{args.dataset}_{args.model}.pkl"
            )[len(embeddings) :]
            # save(embeddings, f"output/train_embeddings_{args.dataset}_{args.model}_{args.learn_method}.pkl")
            # if args.whiten:
            #     embeddings, train_center, whiten_mat, dewhiten_mat = whiten(embeddings, return_fit=True)
            # else:
            #     if args.center:
            #         train_center = torch.mean(embeddings, dim=0)
            #         embeddings = embeddings - torch.mean(embeddings, dim=0)
            #         test_emb = test_emb - train_center
            #     if args.normalize:
            #         embeddings = normalize(embeddings)

            log_dir = f"{args.data_dir}/ct"
            os.makedirs(log_dir, exist_ok=True)
            cl = concept_transformer_learner(
                n_concepts,
                9,
                device=device,
                epochs=1000,
                batch_size=32,
                seed=seed,
                log_dir=log_dir,
                embedding_dim=embeddings.shape[1],
            )
            # convert one hot grammar_labels to integer labels
            int_grammar_labels = np.argmax(grammar_labels[:, 8:], axis=1)
            train_emb, val_emb, train_y, val_y = train_test_split(
                embeddings,
                int_grammar_labels[: len(embeddings)],
                test_size=0.2,
                random_state=seed,
            )
            start = time.time()
            cl.training(
                train_emb,
                train_y,
                val_emb,
                val_y,
                test_emb,
                int_grammar_labels[len(embeddings) :],
            )
            duration = time.time() - start
            attr_concepts = normalize(cl.return_concepts().detach().cpu()[0])
            save((attr_concepts, duration), concept_path)
    elif args.learn_method == "random":
        concept_path = f"{args.data_dir}/attr_concepts_{args.dataset}_{args.model}_{args.learn_method}_{args.segment_method}_{seed}\
                {'_center' if args.center else ''}{'_normalize' if args.normalize else ''}{'_whiten' if args.whiten else ''}.pkl".replace(
            " ", ""
        )
        if os.path.exists(concept_path):
            attr_concepts, duration = load(concept_path)
        else:
            # _, embeddings = concept_learner.get_patches(8, method="none")
            # if args.whiten:
            #     embeddings, mean, whiten_mat, dewhiten_mat = whiten(embeddings, return_fit=True)
            # else:
            #     if args.center:
            #         embeddings = embeddings - torch.mean(embeddings, dim=0)
            #     if args.normalize:
            #         embeddings = normalize(embeddings)

            start = time.time()
            concepts = torch.randn(
                n_concepts,
                embeddings.shape[1],
                generator=torch.Generator().manual_seed(seed),
            )
            duration = time.time() - start
            # if args.whiten:
            #     concepts = concepts @ whiten_mat
            #     embeddings = (embeddings @ dewhiten_mat)

            attr_concepts = normalize(concepts.float())
            save((attr_concepts, duration), concept_path)

    return attr_concepts, duration


def eval_concepts(
    embeddings, concepts_pred, concepts_gt, grammar_labels_base, grammar_labels_composed
):
    results = {}

    concept_scores = cosim(embeddings, concepts_pred)

    # print("Max cosine similarity to each GT concept:")
    gt_cosim, gt_matches = concept_gt_match(concepts_pred, concepts_gt)
    # for i, name in enumerate(grammar_names):
    #     print(f"{name}: {gt_cosim[i].item():.3f}")

    # print("Avg of first 6:", torch.mean(gt_cosim[:6]).item())
    mean_cosim = torch.mean(gt_cosim[:6]).item()
    results["mean_cosim"] = mean_cosim

    # print("Max AUC to each GT concept:")
    gt_label_matches, gt_auc, signs = concept_gt_match_labels(
        concept_scores,
        grammar_labels_base,
        allow_negative=(args.learn_method != "ours"),
    )
    # for i, name in enumerate(grammar_names):
    #     print(f"{name}: {gt_auc[i].item():.3f}")

    # print("Avg of first 6:", torch.mean(gt_auc[:6]).item())
    mean_auc = torch.mean(gt_auc[:6]).item()
    results["mean_auc"] = mean_auc

    # _, gt_auc, _ = concept_gt_match_labels(cosim(embeddings, concepts_gt), grammar_labels)
    # print("GT Avg of first 6:", torch.mean(gt_auc[:6]).item())

    # Order the learned concepts to match the GT concepts and align their signs with the positive GT label
    print(gt_label_matches)
    print(signs)
    matched_concepts = concepts_pred[gt_label_matches] * signs.unsqueeze(1)
    # matched_scores = embeddings @ matched_concepts.T
    # concept_thresholds = concept_threshold(matched_scores, grammar_labels)
    # print("Thresholds:", concept_thresholds)

    # print("Compositionality MAP:")
    map = np.mean(
        compositional_f1(
            embeddings,
            matched_concepts[:6],
            torch.tensor(grammar_labels_base),
            torch.tensor(grammar_labels_composed),
        )
    )
    results["map"] = map

    # # Get the composed concepts
    # composed_concepts, aucs, cos, _1, _2, _3 = concept_match(
    #     matched_concepts[:6],
    #     torch.tensor(grammar_labels[train_test_labels == 1, :6]),
    #     torch.tensor(grammar_labels[train_test_labels == 1, 6:]),
    #     torch.tensor(grammar_labels[train_test_labels == 0, 6:]),
    #     sample_emb[train_test_labels == 1],
    #     sample_emb[train_test_labels == 0])
    # print("Compositional AUCs:", aucs)

    # print("Compositionality score:")
    cscore = compositionality_eval(
        embeddings, matched_concepts[:6], torch.tensor(grammar_labels_base)
    )
    results["cscore"] = cscore

    return results


def imbalance_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if args.model == "CLIP":
        img_forward, _, raw_processor = load_clip(device)
    else:
        assert args.model == "ViT"
        img_forward, _, raw_processor = load_vit(device)

    # Load dataset
    num_objects = int(args.dataset[-1])
    assert num_objects in [1, 2], "CLEVR dataset must be one or two objects"
    samples, labels, grammar_labels, grammar_names, relation_mask = load_clevr(
        args.data_dir, num_objects
    )
    print(len(grammar_names))
    train_samples = []
    train_labels = []
    test_samples = []
    test_labels = []
    i = 0
    print("Number of samples:", len(samples))

    # Performing the train/test split
    n_test = int(len(samples) * 0.2)
    for img, label in zip(samples, labels):
        if i >= len(samples) - n_test:  # np.sum(label[:3]) > 0:
            test_samples.append(img)
            test_labels.append(label)
        else:
            train_samples.append(img)
            train_labels.append(label)
        i += 1
    n_concepts = 9
    processor = lambda images: raw_processor(images=images, return_tensors="pt")[
        "pixel_values"
    ]
    model = img_forward
    subset_size = 100

    # Extract or load the already extracted concepts
    concept_learner = ConceptLearner(train_samples, model, processor, device)

    all_samples = (
        train_samples + test_samples
        if type(train_samples) == list
        else torch.cat((train_samples, test_samples), dim=0)
    )
    samples_processed = processor(all_samples)
    train_test_labels = torch.cat(
        [torch.ones(len(train_samples)), torch.zeros(len(test_samples))], dim=0
    )
    labels = (
        torch.tensor(train_labels + test_labels)
        if type(train_labels) == list
        else torch.cat((train_labels, test_labels), dim=0)
    )

    torch.manual_seed(42)
    perm = torch.randperm(len(all_samples))
    samples_processed = samples_processed[perm]
    labels = labels[perm]
    grammar_labels = grammar_labels[perm]
    train_test_labels = train_test_labels[perm]
    save(
        np.array(all_samples)[perm],
        f"{args.data_dir}/all_imgs_{args.dataset}_{args.model}.pkl",
    )

    if os.path.exists(f"{args.data_dir}/sample_emb_{args.dataset}_{args.model}.pkl"):
        sample_emb = load(f"{args.data_dir}/sample_emb_{args.dataset}_{args.model}.pkl")
    else:
        sample_emb = _batch_inference(
            model, samples_processed, batch_size=128, device=device
        )
        save(sample_emb, f"{args.data_dir}/sample_emb_{args.dataset}_{args.model}.pkl")

    # Process the embeddings
    if args.whiten:
        sample_emb = sample_emb - torch.mean(sample_emb, dim=0)
    else:
        if args.center:
            sample_emb = sample_emb - torch.mean(sample_emb, dim=0)
        if args.normalize:
            sample_emb = normalize(sample_emb)

    # removing the first item which is the "object" concept
    grammar_labels = grammar_labels[:, 1:]
    grammar_names = grammar_names[1:]
    relation_mask = relation_mask[1:]

    def remove_percent_red(sample_emb, grammar_labels, train_test_labels, p):
        red_mask = torch.tensor(grammar_labels[:, 0])
        random_mask = torch.rand(len(red_mask)) < (1 - p)
        mask = torch.bitwise_xor(
            torch.ones(len(red_mask)).bool(), red_mask.bool() & random_mask
        )
        return sample_emb[mask], grammar_labels[mask], train_test_labels[mask]

    all_results = []
    all_results_std = []
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        embeddings, grammar_labels_p, train_test_labels_p = remove_percent_red(
            sample_emb, grammar_labels, train_test_labels, p
        )
        # Get the GT concepts
        gt_concepts = []
        train_emb = embeddings[train_test_labels_p == 1]
        valid_mask = np.ones(grammar_labels_p.shape[1]).astype(bool)
        for lid in range(grammar_labels_p.shape[1]):
            if (
                np.sum(grammar_labels_p[train_test_labels_p == 1, lid]) == 0
                or np.sum(grammar_labels_p[train_test_labels_p == 1, lid])
                == train_emb.shape[0]
            ):
                valid_mask[lid] = False
                continue
            gt_concepts.append(
                normalize(
                    torch.mean(
                        train_emb[
                            grammar_labels_p[train_test_labels_p == 1, lid].astype(bool)
                        ],
                        dim=0,
                        keepdim=True,
                    )
                )
            )
        gt_concepts = torch.cat(gt_concepts, dim=0)
        grammar_labels_p = grammar_labels_p[:, valid_mask]

        results = []
        for seed in range(3):
            if os.path.exists(
                f"{args.data_dir}/concepts_{args.dataset}_{args.model}_ours_ablate_red_{p}_{seed}.pkl"
            ):
                concepts = load(
                    f"{args.data_dir}/concepts_{args.dataset}_{args.model}_ours_ablate_red_{p}_{seed}.pkl"
                )
            else:
                concepts = concept_learner.learn_attribute_concepts(
                    3,
                    train_emb,
                    n_concepts=[3, 3, 3],
                    split_method="ours-subspace",
                    use_leace=False,
                    seed=seed,
                    cross_entropy=args.cross_entropy,
                )
                save(
                    concepts,
                    f"{args.data_dir}/concepts_{args.dataset}_{args.model}_ours_ablate_red_{p}_{seed}.pkl",
                )

            gt_cosim, gt_matches = concept_gt_match(concepts, gt_concepts)
            mean_cosim = torch.mean(gt_cosim[:6]).item()
            results.append(mean_cosim)
        all_results.append(np.mean(results))
        all_results_std.append(np.std(results))
        print(f"{p}: {np.mean(results):.3f} ({np.std(results):.3f})")
    # output the results
    for p, mean, std in zip([0.1, 0.3, 0.5, 0.7, 0.9], all_results, all_results_std):
        print(f"{p}: {mean:.3f} ({std:.3f})")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if args.model == "CLIP":
        img_forward, _, raw_processor = load_clip(device)
    elif args.model == "ViT":
        img_forward, _, raw_processor = load_vit(device)
    else:
        assert args.model == "ResNet"
        img_forward, _, raw_processor = load_resnet(device)

    # Load dataset
    num_objects = int(args.dataset[-1])
    assert num_objects in [1, 2], "CLEVR dataset must be one or two objects"
    samples, labels, grammar_labels, grammar_names, relation_mask = load_clevr(
        args.data_dir, num_objects
    )
    print(len(grammar_names))
    train_samples = []
    train_labels = []
    test_samples = []
    test_labels = []
    i = 0
    print("Number of samples:", len(samples))
    n_test = int(len(samples) * 0.2)
    for img, label in zip(samples, labels):
        if i >= len(samples) - n_test:  # np.sum(label[:3]) > 0:
            test_samples.append(img)
            test_labels.append(label)
        else:
            train_samples.append(img)
            train_labels.append(label)
        i += 1
    n_concepts = 9
    processor = lambda images: raw_processor(images=images, return_tensors="pt")[
        "pixel_values"
    ]
    model = img_forward
    subset_size = 100

    # Extract or load the already extracted concepts
    concept_learner = ConceptLearner(train_samples, model, processor, device)

    all_samples = (
        train_samples + test_samples
        if type(train_samples) == list
        else torch.cat((train_samples, test_samples), dim=0)
    )
    samples_processed = processor(all_samples)
    train_test_labels = torch.cat(
        [torch.ones(len(train_samples)), torch.zeros(len(test_samples))], dim=0
    )
    labels = (
        torch.tensor(train_labels + test_labels)
        if type(train_labels) == list
        else torch.cat((train_labels, test_labels), dim=0)
    )

    torch.manual_seed(42)
    perm = torch.randperm(len(all_samples))
    samples_processed = samples_processed[perm]
    labels = labels[perm]
    grammar_labels = grammar_labels[perm]
    train_test_labels = train_test_labels[perm]
    save(
        np.array(all_samples)[perm],
        f"{args.data_dir}/all_imgs_{args.dataset}_{args.model}.pkl",
    )

    if os.path.exists(f"{args.data_dir}/sample_emb_{args.dataset}_{args.model}.pkl"):
        sample_emb = load(f"{args.data_dir}/sample_emb_{args.dataset}_{args.model}.pkl")
    else:
        sample_emb = _batch_inference(
            model, samples_processed, batch_size=128, device=device
        )
        save(sample_emb, f"{args.data_dir}/sample_emb_{args.dataset}_{args.model}.pkl")

    # Process the embeddings
    if args.whiten:
        sample_emb = sample_emb - torch.mean(sample_emb, dim=0)
    else:
        if args.center:
            sample_emb = sample_emb - torch.mean(sample_emb, dim=0)
        if args.normalize:
            sample_emb = normalize(sample_emb)

    # removing the first item which is the "object" concept
    grammar_labels = grammar_labels[:, 1:]
    grammar_names = grammar_names[1:]
    relation_mask = relation_mask[1:]
    print(grammar_labels[:5])

    # Get the GT concepts
    gt_concepts = []
    train_emb = sample_emb[train_test_labels == 1]
    valid_mask = np.ones(grammar_labels.shape[1]).astype(bool)
    for lid in range(grammar_labels.shape[1]):
        if (
            np.sum(grammar_labels[train_test_labels == 1, lid]) == 0
            or np.sum(grammar_labels[train_test_labels == 1, lid]) == train_emb.shape[0]
        ):
            valid_mask[lid] = False
            continue
        gt_concepts.append(
            normalize(
                torch.mean(
                    train_emb[grammar_labels[train_test_labels == 1, lid].astype(bool)],
                    dim=0,
                    keepdim=True,
                )
            )
        )
    gt_concepts = torch.cat(gt_concepts, dim=0)
    grammar_labels = grammar_labels[:, valid_mask]
    grammar_names = np.array(grammar_names)[valid_mask]

    print(gt_concepts[0])

    all_results = []
    for seed in range(3):
        if args.learn_method == "gt":
            attr_concepts = gt_concepts[:6]
            duration = 0
        else:
            attr_concepts, duration = get_concepts(
                args,
                concept_learner,
                n_concepts,
                device,
                train_emb,
                grammar_labels,
                seed,
            )

        # Run the experiment which uses the concepts (start with classification)
        all_concepts = normalize(attr_concepts)

        print("Number of concepts:", len(all_concepts))
        print(sample_emb.shape, all_concepts.shape, gt_concepts.shape)

        results = eval_concepts(
            sample_emb[train_test_labels == 0],
            all_concepts,
            gt_concepts,
            grammar_labels[train_test_labels == 0, :6],
            grammar_labels[train_test_labels == 0, 6:],
        )
        results["duration"] = duration
        all_results.append(results)

    merged_results = {}
    for key in all_results[0].keys():
        merged_results[key] = np.mean([r[key] for r in all_results])
        merged_results[key + "_std"] = np.std([r[key] for r in all_results])

    for key, value in merged_results.items():
        print(key, f"{value:.3f}")

    if os.path.exists("results/clevr.csv"):
        df = pd.read_csv("results/clevr.csv").to_dict("records")
    else:
        df = []
    merged_results["dataset"] = args.dataset
    merged_results["model"] = args.model
    merged_results["method"] = args.learn_method
    if args.cross_entropy:
        merged_results["method"] += "-ce"
    if args.no_reg:
        merged_results["method"] += "-noreg"
    df.append(merged_results)
    df = pd.DataFrame(df).to_csv("results/clevr.csv", index=False)

    # sample_concept_scores = sample_emb @ all_concepts.T

    # # GT recovery evaluation
    # # removing the first item which is the "object" concept
    # grammar_labels = grammar_labels[:, 1:]
    # grammar_names = grammar_names[1:]
    # relation_mask = relation_mask[1:]

    # # Get the GT concepts
    # gt_concepts = []
    # train_emb = sample_emb[train_test_labels == 1]
    # valid_mask = np.ones(grammar_labels.shape[1]).astype(bool)
    # for lid in range(grammar_labels.shape[1]):
    #     if np.sum(grammar_labels[train_test_labels == 1, lid]) == 0 or np.sum(grammar_labels[train_test_labels == 1, lid]) == train_emb.shape[0]:
    #         valid_mask[lid] = False
    #         continue
    #     gt_concepts.append(normalize(torch.mean(train_emb[grammar_labels[train_test_labels == 1, lid].astype(bool)], dim=0, keepdim=True)))
    # gt_concepts = torch.cat(gt_concepts, dim=0)
    # grammar_labels = grammar_labels[:, valid_mask]
    # grammar_names = np.array(grammar_names)[valid_mask]

    # print("Max cosine similarity to each GT concept:")
    # gt_cosim, gt_matches = concept_gt_match(all_concepts, gt_concepts)
    # for i, name in enumerate(grammar_names):
    #     print(f"{name}: {gt_cosim[i].item():.3f}")

    # print("Avg of first 6:", torch.mean(gt_cosim[:6]).item())

    # print("Max AUC to each GT concept:")
    # gt_label_matches, gt_auc, signs = concept_gt_match_labels(sample_concept_scores, grammar_labels)
    # for i, name in enumerate(grammar_names):
    #     print(f"{name}: {gt_auc[i].item():.3f}")

    # print("Avg of first 6:", torch.mean(gt_auc[:6]).item())

    # _, gt_auc, _ = concept_gt_match_labels(cosim(sample_emb, gt_concepts), grammar_labels)
    # print("GT Avg of first 6:", torch.mean(gt_auc[:6]).item())

    # # Order the learned concepts to match the GT concepts and align their signs with the positive GT label
    # matched_concepts = all_concepts[gt_label_matches] * signs.unsqueeze(1)
    # matched_scores = sample_emb @ matched_concepts.T
    # concept_thresholds = concept_threshold(matched_scores, grammar_labels)
    # print("Thresholds:", concept_thresholds)

    # print("Compositionality MAP:")
    # print(np.mean(compositional_f1(sample_emb[train_test_labels == 0], matched_concepts[:6], torch.tensor(grammar_labels[train_test_labels == 0, :6]), torch.tensor(grammar_labels[train_test_labels == 0, 6:]))))

    # # Get the composed concepts
    # composed_concepts, aucs, cos, _1, _2, _3 = concept_match(
    #     matched_concepts[:6],
    #     torch.tensor(grammar_labels[train_test_labels == 1, :6]),
    #     torch.tensor(grammar_labels[train_test_labels == 1, 6:]),
    #     torch.tensor(grammar_labels[train_test_labels == 0, 6:]),
    #     sample_emb[train_test_labels == 1],
    #     sample_emb[train_test_labels == 0])
    # print("Compositional AUCs:", aucs)

    # print("Compositionality score:")
    # print(compositionality_eval(sample_emb[train_test_labels == 0], torch.tensor(matched_concepts[:6]), torch.tensor(grammar_labels[train_test_labels == 0, :6])))


if __name__ == "__main__":
    # parse arguments with argparse
    parser = argparse.ArgumentParser(description="Run the concept learning experiments")

    # CUB is located at /data5/steinad/datasets/CUB_200_2011
    # MIT-States is located at /data5/steinad/datasets/mit-states
    # CLEVR is located at /home/steinad/src/clevr-dataset-gen/two_object
    parser.add_argument(
        "--data_dir", type=str, default="", help="Directory where the data is stored"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CLEVR",
        help="Dataset to run the experiments on",
        choices=["CLEVR1"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="CLIP",
        help="Model to run the experiments on",
        choices=["CLIP", "ResNet", "ViT"],
    )
    parser.add_argument(
        "--learn_method",
        type=str,
        default="ours",
        help="Method to use for concept learning",
        choices=[
            "ours",
            "ace",
            "ace-svm",
            "pca",
            "dictlearn",
            "seminmf",
            "ct",
            "random",
            "gt",
        ],
    )
    parser.add_argument(
        "--segment_method",
        type=str,
        default="window",
        help="Method to use for concept segmentation",
        choices=["none"],
    )
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--center", action="store_true", default=False)
    parser.add_argument("--whiten", action="store_true", default=False)
    parser.add_argument("--cross_entropy", action="store_true", default=False)
    parser.add_argument("--imbalance_ablation", action="store_true", default=False)
    parser.add_argument("--no_reg", action="store_true", default=False)
    args = parser.parse_args()

    if args.imbalance_ablation:
        imbalance_experiment(args)

    main(args)
