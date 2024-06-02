from __future__ import annotations

import argparse
import csv
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.joint_baselines import concept_transformer_learner
from compositionality_eval import compositional_f1, compositionality_eval
from concept_learning import ConceptLearner
from utils import TextDataset, _batch_inference, cosim, load, normalize, save


def truth_topics_dataset(
    full=False,
    zero_shot=False,
    simple_prompt=False,
    ntrain=None,
    seed=0,
    data_dir="truth/",
):
    template_str = "Consider the topic and correctness of the following fact:\nFact: {fact}.\nThe topic and probability of the fact being correct is "
    # animal_template_str = "Consider the topic of the following:\nSentence: {fact}.\nThe topic of the sentence is "

    animals_tf_data = pd.read_csv(
        f"{data_dir}/truth_topics/animals_true_false.csv", sep=","
    )
    companies_tf_data = pd.read_csv(
        f"{data_dir}/truth_topics/companies_true_false.csv", sep=","
    )
    cities_tf_data = pd.read_csv(
        f"{data_dir}/truth_topics/inventions_true_false.csv", sep=","
    )
    elements_tf_data = pd.read_csv(
        f"{data_dir}/truth_topics/elements_true_false.csv", sep=","
    )
    facts_tf_data = pd.read_csv(
        f"{data_dir}/truth_topics/facts_true_false.csv", sep=","
    )
    inventions_tf_data = pd.read_csv(
        f"{data_dir}/truth_topics/inventions_true_false.csv", sep=","
    )

    animals_data = []
    companies_data = []
    cities_data = []
    elements_data = []
    facts_data = []
    inventions_data = []
    animals_labels = []
    companies_labels = []
    cities_labels = []
    elements_labels = []
    facts_labels = []
    inventions_labels = []
    for row in animals_tf_data.values.tolist():
        animals_data.append(row[0])
        animals_labels.append([row[1], 0])
    for row in companies_tf_data.values.tolist():
        companies_data.append(row[0])
        companies_labels.append([row[1], 1])
    for row in cities_tf_data.values.tolist():
        cities_data.append(row[0])
        cities_labels.append([row[1], 2])
    for row in elements_tf_data.values.tolist():
        elements_data.append(row[0])
        elements_labels.append([row[1], 3])
    for row in facts_tf_data.values.tolist():
        facts_data.append(row[0])
        facts_labels.append([row[1], 4])
    for row in inventions_tf_data.values.tolist():
        inventions_data.append(row[0])
        inventions_labels.append([row[1], 5])

    animals_data = np.array(animals_data)
    animals_labels = np.array(animals_labels)
    companies_data = np.array(companies_data)
    companies_labels = np.array(companies_labels)
    cities_data = np.array(cities_data)
    cities_labels = np.array(cities_labels)
    elements_data = np.array(elements_data)
    elements_labels = np.array(elements_labels)
    facts_data = np.array(facts_data)
    facts_labels = np.array(facts_labels)
    inventions_data = np.array(inventions_data)
    inventions_labels = np.array(inventions_labels)

    # animals t/f and companies t
    data_all = (
        [template_str.format(fact=f) for f in animals_data[:500]]
        + [template_str.format(fact=f) for f in companies_data[:500]]
        + [template_str.format(fact=f) for f in cities_data[:500]]
    )
    labels_all = (
        animals_labels[:500].tolist()
        + companies_labels[:500].tolist()
        + cities_labels[:500].tolist()
    )

    if full:
        data_all = (
            [template_str.format(fact=f) for f in animals_data]
            + [template_str.format(fact=f) for f in companies_data]
            + [template_str.format(fact=f) for f in cities_data]
            + [template_str.format(fact=f) for f in elements_data]
            + [template_str.format(fact=f) for f in facts_data]
            + [template_str.format(fact=f) for f in inventions_data]
        )
        labels_all = (
            animals_labels.tolist()
            + companies_labels.tolist()
            + cities_labels.tolist()
            + elements_labels.tolist()
            + facts_labels.tolist()
            + inventions_labels.tolist()
        )

    np.random.seed(seed)
    data_train, data_val, labels_train, labels_val = train_test_split(
        data_all, labels_all, test_size=0.5
    )
    data_val, data_test, labels_val, labels_test = train_test_split(
        data_val, labels_val, test_size=0.5
    )

    return {
        "train": {"data": data_train, "labels": labels_train},
        "test": {"data": data_test, "labels": labels_test},
        "val": {"data": data_val, "labels": labels_val},
    }


def load_llama(device):
    model_name_or_path = "meta-llama/Llama-2-13b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="auto", torch_dtype=torch.float16, token=True
    ).eval()
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast_tokenizer,
        padding_side="left",
        legacy=False,
        token=True,
    )
    tokenizer.pad_token_id = (
        0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    )
    tokenizer.bos_token_id = 1

    tokenizer.model_max_length = 4096

    def text_forward(samples):
        with torch.no_grad():
            return model(
                input_ids=samples[0][0],
                attention_mask=samples[0][1],
                output_hidden_states=True,
            ).hidden_states[-1][:, -1, :]

    return None, text_forward, tokenizer


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


def get_concepts(args, Z_train, train_labels, Z_test, test_labels, seed=0):
    n_concepts = args.n_attr * args.n_concepts_per_attr
    if args.method == "ours":
        pathname = f"{args.dataset}_{'full_' if args.full else ''}{'ce_' if args.cross_entropy else ''}{args.method}_concepts_{args.n_attr}_{args.n_concepts_per_attr}_{args.subspace_dim}_{seed}.pkl"
    else:
        pathname = f"{args.dataset}_{'full_' if args.full else ''}{'ce_' if args.cross_entropy else ''}{args.method}_concepts_{n_concepts}_{seed}.pkl"
    if not os.path.exists(pathname):
        cl = ConceptLearner(
            samples=[None], input_to_latent=None, input_processor=None, device="cuda"
        )
        if args.method == "ours":
            start = time.time()
            concepts_per_attr = [args.n_concepts_per_attr]
            if not args.full:
                concepts_per_attr = [4, 2, 3]
            concepts = cl.learn_attribute_concepts(
                args.n_attr,
                Z_train,
                subspace_dim=args.subspace_dim,
                n_concepts=args.n_concepts_per_attr * args.n_attr,
                split_method="ours-subspace",
                use_leace=False,
                seed=seed,
                cross_entropy=args.cross_entropy,
            )
            duration = time.time() - start
        elif args.method == "ace":
            start = time.time()
            concepts = cl.learn_ace_concepts(n_concepts, Z_train, seed=seed)
            duration = time.time() - start
        elif args.method == "pca":
            start = time.time()
            concepts = cl.learn_pca_concepts(n_concepts, Z_train, seed=seed)
            duration = time.time() - start
        elif args.method == "dictlearn":
            start = time.time()
            concepts = cl.learn_dictlearn_concepts(n_concepts, Z_train, seed=seed)
            duration = time.time() - start
        elif args.method == "seminmf":
            start = time.time()
            concepts = cl.learn_seminmf_concepts(n_concepts, Z_train, seed=seed)
            duration = time.time() - start
        elif args.method == "gt":
            concepts = []
            start = time.time()
            concepts.append(
                normalize(Z_train[train_labels[:, 0] == 1].mean(0, keepdims=True))
            )
            concepts.append(
                normalize(Z_train[train_labels[:, 1] == 0].mean(0, keepdims=True))
            )
            concepts.append(
                normalize(Z_train[train_labels[:, 1] == 1].mean(0, keepdims=True))
            )
            concepts.append(
                normalize(Z_train[train_labels[:, 1] == 2].mean(0, keepdims=True))
            )
            if args.full:
                concepts.append(
                    normalize(Z_train[train_labels[:, 1] == 3].mean(0, keepdims=True))
                )
                concepts.append(
                    normalize(Z_train[train_labels[:, 1] == 4].mean(0, keepdims=True))
                )
                concepts.append(
                    normalize(Z_train[train_labels[:, 1] == 5].mean(0, keepdims=True))
                )
            concepts = torch.cat(concepts, dim=0)
            duration = time.time() - start
        elif args.method == "ct":
            log_dir = f"{args.data_dir}/ct"
            os.makedirs(log_dir, exist_ok=True)
            cl = concept_transformer_learner(
                n_concepts,
                2,
                device="cuda",
                epochs=1000,
                batch_size=32,
                log_dir=log_dir,
                embedding_dim=Z_train.shape[1],
                seed=seed,
            )
            y_train = train_labels[:, 0]
            y_test = test_labels[:, 0]
            Z_train, Z_val, y_train, y_val = train_test_split(
                Z_train, y_train, test_size=0.2, random_state=42
            )
            start = time.time()
            cl.training(Z_train, y_train, Z_val, y_val, Z_test, y_test)
            duration = time.time() - start
            concepts = normalize(cl.return_concepts().detach().cpu()[0])
        elif args.method == "random":
            torch.manual_seed(seed)
            start = time.time()
            concepts = normalize(torch.randn(9, Z_train.shape[1]))
            duration = time.time() - start
        elif args.method == "raw":
            concepts = torch.eye(Z_train.shape[1])

        pickle.dump((concepts, duration), open(pathname, "wb"))
    else:
        concepts, duration = pickle.load(open(pathname, "rb"))

    return concepts, duration


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "truth_topics":
        data = truth_topics_dataset(full=args.full, data_dir=args.data_dir)
    else:
        raise NotImplementedError

    for k, v in data.items():
        print(k, len(v["data"]))
    data_train = data["train"]["data"]
    data_test = data["val"]["data"]
    train_labels = torch.tensor(data["train"]["labels"])
    test_labels = torch.tensor(data["val"]["labels"])

    if os.path.exists(
        f"{args.data_dir}/{args.dataset}_{'full_' if args.full else ''}hiddens.pkl"
    ):
        Z_train, Z_test = load(
            f"{args.data_dir}/{args.dataset}_{'full_' if args.full else ''}hiddens.pkl"
        )
    else:
        img_forward, text_forward, raw_processor = load_llama(device)
        processor = lambda samples: TextDataset(
            raw_processor(
                text=samples, return_tensors="pt", padding=True, truncation=True
            )
        )
        model = text_forward

        Z_train = _batch_inference(
            model,
            [str(row) for row in data_train],
            processor=processor,
            batch_size=4,
            device=device,
        ).float()
        Z_test = _batch_inference(
            model,
            [str(row) for row in data_test],
            processor=processor,
            batch_size=4,
            device=device,
        ).float()
        mean = Z_train.mean(0, keepdims=True)
        Z_train = Z_train - mean
        Z_test = Z_test - mean
        save(
            (Z_train, Z_test),
            f"{args.data_dir}/{args.dataset}_{'full_' if args.full else ''}hiddens.pkl",
        )

    print(Z_train.shape, Z_test.shape)

    all_results = []
    for seed in range(3):
        results = {}
        concepts, duration = get_concepts(
            args, Z_train, train_labels, Z_test, test_labels, seed
        )
        results["duration"] = duration

        print("Number of concepts:", concepts.shape[0])
        # Get concept scores
        Z_train_scores = cosim(Z_train, concepts)
        Z_test_scores = cosim(Z_test, concepts)

        # # Evaluate downstream accuracy
        # if args.dataset == "obqa":
        #     train_labels = train_labels.flatten()[:, None]
        #     test_labels = test_labels.flatten()[:, None]

        # Z_train_scores, Z_val_scores, y_train, y_val = train_test_split(Z_train_scores, train_labels[:, 0], test_size=0.2, random_state=42)
        # lr = LogisticRegression(penalty=None, max_iter=10000).fit(Z_train_scores, y_train)
        # print("Train accuracy:", lr.score(Z_train_scores, y_train))
        # print("Val accuracy:", lr.score(Z_val_scores, y_val))
        # print("Test accuracy:", lr.score(Z_test_scores, test_labels[:, 0]))

        if args.dataset == "truth_topics":
            # Get GT concepts
            num_topics = len(np.unique(train_labels[:, 1]))
            print(num_topics)
            concept_names = [
                "truth",
                "animal",
                "company",
                "cities",
                "elements",
                "facts",
                "inventions",
            ]
            gt_concepts = []
            gt_concepts.append(
                normalize(Z_train[train_labels[:, 0] == 1].mean(0, keepdims=True))
            )
            for i in np.unique(train_labels[:, 1]):
                gt_concepts.append(
                    normalize(Z_train[train_labels[:, 1] == i].mean(0, keepdims=True))
                )
            gt_concepts = torch.cat(gt_concepts, dim=0)

            # print("Max cosine similarity to each GT concept:")
            gt_cosim, gt_matches = concept_gt_match(concepts, gt_concepts)
            # for i, name in enumerate(concept_names[:num_topics + 1]):
            #     print(f"{name}: {gt_cosim[i].item():.3f}")
            # print(f"{args.method} " + " & ".join([f"{gt_cosim[i].item():.3f}" for i in range(num_topics + 1)]) + "\\\\")
            # print(f"{args.method} & {gt_cosim[0].item():.3f} & {gt_cosim[1].item():.3f} & {gt_cosim[2].item():.3f} & {gt_cosim[3].item():.3f}\\\\")
            # print("Avg. cosine:", torch.mean(gt_cosim).item())
            results["mean_cosim"] = torch.mean(gt_cosim).item()

            # print("Max AUC to each GT concept:")
            test_oh_labels = torch.zeros(Z_test.shape[0], num_topics + 1)
            test_oh_labels[:, 0] = test_labels[:, 0] == 1
            for i in np.unique(test_labels[:, 1]):
                test_oh_labels[:, i + 1] = test_labels[:, 1] == i

            # print(torch.sum(test_oh_labels, dim=0))
            gt_label_matches, gt_auc, signs = concept_gt_match_labels(
                Z_test_scores, test_oh_labels, allow_neg=args.method != "ours"
            )
            # for i, name in enumerate(concept_names[:num_topics + 1]):
            #     print(f"{name}: {gt_auc[i].item():.3f}")
            # print(f"{args.method} & {gt_auc[0].item():.3f} & {gt_auc[1].item():.3f} & {gt_auc[2].item():.3f} & {gt_auc[3].item():.3f}\\\\")
            # print("Avg. AUC:", torch.mean(gt_auc).item())
            results["mean_auc"] = torch.mean(gt_auc).item()

            # print("Compositional F1-score")
            test_oh_labels = torch.zeros(test_labels.shape[0], num_topics + 1)
            test_oh_labels[:, 0] = test_labels[:, 0] == 1
            for i in np.unique(train_labels[:, 1]):
                test_oh_labels[:, i + 1] = test_labels[:, 1] == i
            compositional_labels = torch.zeros(
                test_labels.shape[0], 2 * len(np.unique(test_labels[:, 1]))
            )
            for i in range(2):
                for j in np.unique(train_labels[:, 1]):
                    compositional_labels[:, i * num_topics + j] = (
                        test_labels[:, 0] == i
                    ) & (test_labels[:, 1] == j)
            # print(np.mean(compositional_f1(Z_test, concepts[gt_label_matches], test_oh_labels, compositional_labels)))
            results["map"] = np.mean(
                compositional_f1(
                    Z_test,
                    concepts[gt_label_matches] * signs[:, None],
                    test_oh_labels,
                    compositional_labels,
                )
            )

            # print("Compositionality score:")
            print(gt_label_matches, signs)
            # print(concepts[gt_label_matches].shape, test_oh_labels.shape)
            # print(compositionality_eval(Z_test, concepts[gt_label_matches] * signs[:, None], test_oh_labels))
            results["cscore"] = compositionality_eval(
                Z_test, concepts[gt_label_matches] * signs[:, None], test_oh_labels
            )
            all_results.append(results)

    merged_results = {}
    for key in all_results[0].keys():
        merged_results[key] = np.mean([r[key] for r in all_results])
        merged_results[key + "_std"] = np.std([r[key] for r in all_results])

    for key, value in merged_results.items():
        print(key, f"{value:.3f}")

    if os.path.exists("results/truth.csv"):
        df = pd.read_csv("results/truth.csv").to_dict("records")
    else:
        df = []
    merged_results["dataset"] = f"{args.dataset}{'_full' if args.full else ''}"
    merged_results["model"] = "Llama2"
    merged_results["method"] = args.method
    if args.cross_entropy:
        merged_results["method"] += "_ce"
    df.append(merged_results)
    df = pd.DataFrame(df).to_csv("results/truth.csv", index=False)


if __name__ == "__main__":
    # parse arguments with argparse
    parser = argparse.ArgumentParser(description="Run the concept learning experiments")

    parser.add_argument("--data_dir", type=str)
    parser.add_argument(
        "--method",
        type=str,
        default="ours",
        help="Method to use for concept learning",
        choices=[
            "ours",
            "ace",
            "pca",
            "dictlearn",
            "seminmf",
            "gt",
            "ct",
            "random",
            "raw",
        ],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="truth_topics",
        help="Dataset to use",
        choices=["truth_topics", "obqa"],
    )
    parser.add_argument("--full", action="store_true", help="Use the full dataset")
    parser.add_argument(
        "--acc_vs_concepts",
        action="store_true",
        help="Evaluate accuracy vs number of concepts",
    )
    parser.add_argument(
        "--cross_entropy",
        action="store_true",
        help="Use cross-entropy loss for attribute learning",
    )
    parser.add_argument(
        "--n_attr", type=int, default=3, help="Number of attributes to learn"
    )
    parser.add_argument(
        "--n_concepts_per_attr",
        type=int,
        default=3,
        help="Number of concepts per attribute to learn",
    )
    parser.add_argument(
        "--subspace_dim",
        type=int,
        default=200,
        help="Subspace dimension for attribute learning",
    )
    args = parser.parse_args()

    if args.acc_vs_concepts:
        # data_train, data_test, labels_train, labels_test, target_names = load_newsgroups(args.data_dir)

        if args.dataset == "truth_topics":
            data = truth_topics_dataset(full=args.full)
        else:
            raise NotImplementedError

        for k, v in data.items():
            print(k, len(v["data"]))
        labels_train = torch.tensor(data["train"]["labels"])
        labels_test = torch.tensor(data["val"]["labels"])

        if not os.path.exists(
            f"{args.data_dir}/{args.dataset}_{'full_' if args.full else ''}hiddens.pkl"
        ):
            print("ERROR")
        else:
            train_emb, test_emb = pickle.load(
                open(
                    f"{args.data_dir}/{args.dataset}_{'full_' if args.full else ''}hiddens.pkl",
                    "rb",
                )
            )

        accs = {"ours": [], "ace": [], "pca": [], "nmf": [], "ct": [], "dictlearn": []}
        durs = {"ours": [], "ace": [], "pca": [], "nmf": [], "ct": [], "dictlearn": []}
        cl = ConceptLearner(
            samples=[None], input_to_latent=None, input_processor=None, device="cuda"
        )
        # train_emb = load(f"output/train_embeddings_{args.dataset}_{args.model}_{args.learn_method}.pkl").float()
        # test_emb = load(f"output/test_embeddings_{args.dataset}_{args.model}.pkl").float()
        # test_emb = test_emb - torch.mean(train_emb, dim=0)
        # train_emb = train_emb - torch.mean(train_emb, dim=0)
        labels_train = labels_train[:, 0]
        labels_test = labels_test[:, 0]

        penalty = "l2"
        for seed in range(3):
            our_concepts, duration = pickle.load(
                open(
                    f"{args.dataset}_{'full_' if args.full else ''}{args.method}_concepts_120_10_{seed}.pkl",
                    "rb",
                )
            )
            durs["ours"].append(duration)
            # pca_concepts = load(f"output/attr_concepts_{args.dataset}_{args.model}_pca_center.pkl")
            start = time.time()
            pca_concepts = cl.learn_pca_concepts(120, train_emb)
            duration = time.time() - start
            durs["pca"].append(duration)

            for c_num in range(len(our_concepts), len(our_concepts) + 10, 10):
                lr = LogisticRegression(penalty=penalty, max_iter=10000).fit(
                    cosim(train_emb, our_concepts[:c_num]), labels_train
                )
                accs["ours"].append(
                    lr.score(cosim(test_emb, our_concepts[:c_num]), labels_test)
                )

                lr = LogisticRegression(penalty=penalty, max_iter=10000).fit(
                    cosim(train_emb, pca_concepts[:c_num]), labels_train
                )
                accs["pca"].append(
                    lr.score(cosim(test_emb, pca_concepts[:c_num]), labels_test)
                )

                start = time.time()
                ace_concepts = cl.learn_ace_concepts(c_num, train_emb)
                duration = time.time() - start
                lr = LogisticRegression(penalty=penalty, max_iter=10000).fit(
                    cosim(train_emb, ace_concepts), labels_train
                )
                accs["ace"].append(lr.score(cosim(test_emb, ace_concepts), labels_test))
                durs["ace"].append(duration)

                start = time.time()
                nmf_concepts = cl.learn_seminmf_concepts(c_num, train_emb)
                duration = time.time() - start
                lr = LogisticRegression(penalty=penalty, max_iter=10000).fit(
                    cosim(train_emb, nmf_concepts), labels_train
                )
                accs["nmf"].append(lr.score(cosim(test_emb, nmf_concepts), labels_test))
                durs["nmf"] = duration

                start = time.time()
                dictlearn_concepts = cl.learn_dictlearn_concepts(c_num, train_emb)
                duration = time.time() - start
                lr = LogisticRegression(penalty=penalty, max_iter=10000).fit(
                    cosim(train_emb, dictlearn_concepts), labels_train
                )
                accs["dictlearn"].append(
                    lr.score(cosim(test_emb, dictlearn_concepts), labels_test)
                )
                durs["dictlearn"] = duration

                start = time.time()
                ct = concept_transformer_learner(
                    c_num,
                    20,
                    device="cuda",
                    epochs=100,
                    batch_size=32,
                    log_dir=f"{args.data_dir}/ct",
                    embedding_dim=5120,
                )
                Z_train, Z_val, y_train, y_val = train_test_split(
                    train_emb, labels_train, test_size=0.2, random_state=42
                )
                ct.training(Z_train, y_train, Z_val, y_val, test_emb, labels_test)
                duration = time.time() - start
                concepts = normalize(ct.return_concepts().detach().cpu()[0])

                lr = LogisticRegression(penalty=penalty, max_iter=10000).fit(
                    cosim(train_emb, concepts), labels_train
                )
                accs["ct"].append(lr.score(cosim(test_emb, concepts), labels_test))
                durs["ct"].append(duration)

        with open(f"results/accs_seeds_{args.dataset}.csv", "a") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "n_concepts",
                    "method",
                    "dataset",
                    "acc",
                    "acc_std",
                    "duration",
                    "duration_std",
                ],
            )
            writer.writeheader()

            for method in accs:
                writer.writerow(
                    {
                        "method": method,
                        "n_concepts": c_num,
                        "dataset": args.dataset,
                        "acc": np.mean(accs[method]),
                        "acc_std": np.std(accs[method]),
                        "duration": np.mean(durs[method]),
                        "duration_std": np.std(durs[method]),
                    }
                )

    main(args)
