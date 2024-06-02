from __future__ import annotations

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from utils import cosim, normalize


def compositionality_eval(data, concepts, labels, return_coeffs=False, lr=0.1):
    coeffs = torch.nn.Parameter(
        torch.randn(len(data), labels.shape[1]), requires_grad=True
    )
    with torch.no_grad():
        coeffs.abs_()
        coeffs *= labels
    optimizer = torch.optim.Adam([coeffs], lr=lr)
    last_coeffs = coeffs.detach().clone()

    dataset = torch.utils.data.TensorDataset(data, labels, coeffs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=True)
    for _ in range(300):
        for batch, batch_labels, batch_coeffs in loader:
            optimizer.zero_grad()
            loss = torch.tensor(0.0)
            for i in range(batch.shape[0]):
                if torch.sum(batch_labels[i] * batch_coeffs) == 0:
                    loss += torch.norm(batch[i][None, :])
                    # loss += 0
                    continue
                # loss += 1 - cosim(batch[i][None, :], ((batch_labels[i] * batch_coeffs[i])[None, :] @ concepts))[0, 0]
                loss += torch.norm(
                    batch[i][None, :]
                    - ((batch_labels[i] * batch_coeffs[i])[None, :] @ concepts)
                )
            loss /= batch.shape[0]
            loss.backward()
            # print(loss.item())
            # clip gradient
            torch.nn.utils.clip_grad_norm_([coeffs], 5)
            optimizer.step()
            with torch.no_grad():
                coeffs.clamp_(0)
            if torch.norm(coeffs - last_coeffs) < 1e-5:
                break
            last_coeffs = coeffs.detach().clone()
        # print(loss.item())

    # eval
    with torch.no_grad():
        loss = torch.tensor(0.0)
        for i in range(data.shape[0]):
            if torch.sum(labels[i] * coeffs) == 0:
                loss += torch.norm(data[i][None, :])
                # loss += 0
                continue
            loss += torch.norm(
                data[i][None, :] - ((labels[i] * coeffs[i])[None, :] @ concepts)
            )
            # loss += 1 - cosim(data[i][None, :], ((labels[i] * coeffs[i])[None, :] @ concepts))[0, 0]
        loss /= data.shape[0]
    if return_coeffs:
        return loss.item(), coeffs.detach()
    return loss.item()


def compositional_f1(data, concepts, labels, compositional_labels):
    (
        data_train,
        data_test,
        labels_train,
        labels_test,
        compositional_labels_train,
        compositional_labels_test,
    ) = train_test_split(
        data, labels, compositional_labels, test_size=0.2, random_state=42
    )
    f1s = []
    for i in range(compositional_labels.shape[1]):
        if torch.sum(compositional_labels[:, i]) == 0:
            continue
        composed = torch.prod(labels[compositional_labels[:, i] == 1], dim=0) == 1
        component_scores_train = cosim(data_train, concepts[composed])
        component_scores_test = cosim(data_test, concepts[composed])
        lr = LogisticRegression(penalty=None, max_iter=10000).fit(
            component_scores_train, compositional_labels_train[:, i] == 1
        )
        # lr = LogisticRegression(penalty=None, max_iter=10000).fit(component_scores_train, compositional_labels_train[:, i] == 1)
        f1s.append(
            average_precision_score(
                compositional_labels_test[:, i],
                lr.predict_proba(component_scores_test)[:, 1],
            )
        )
    return f1s


def concept_match(
    concepts,
    base_labels,
    compositional_labels,
    compositional_labels_test,
    train_emb,
    test_emb,
    complete_concept_ids=None,
    pseudo=False,
):
    # Get GT base labels or use pseudo labels
    base_labels_binary = (base_labels == 1).int()
    if pseudo:
        for i in range(compositional_labels.shape[1]):
            if torch.sum(compositional_labels[:, i]) == 0:
                continue
            # lr = LogisticRegression(C=100, max_iter=10000).fit(base_labels, compositional_labels[:, i] == 1)
            # top_soft = torch.argsort(torch.tensor(lr.coef_[0]), descending=True)[:5]
            soft_gt = torch.sum(
                base_labels[compositional_labels[:, i] == 1], dim=0
            ) / len(torch.nonzero(compositional_labels[:, i] == 1))
            top_soft = torch.argsort(soft_gt, descending=True)[:2]
            composed = torch.zeros(base_labels.shape[1]).int()
            composed[top_soft] = 1
            base_labels_binary[compositional_labels[:, i] == 1] = composed
        # for i in range(base_labels_binary.shape[0]):
        #     labels = torch.zeros_like(base_labels[i]).int()
        #     # for j in range(5):
        #     #     top_soft = torch.argsort(base_labels[i][20*j:20*j + 20], descending=True)[0] + 20*j
        #     #     labels[top_soft] = 1
        #     # base_labels_binary[i] = labels
        #     top_soft = torch.argsort(base_labels[i], descending=True)[:5]
        #     labels[top_soft] = 1
        #     base_labels_binary[i] = labels

    aucs = []
    cos = []
    gt_composed_concepts = []
    gt_labels = []
    idxs = []
    for i in range(compositional_labels.shape[1]):
        if torch.sum(compositional_labels[:, i]) == 0:
            continue

        gt_composed_concepts.append(
            normalize(
                train_emb[compositional_labels[:, i] == 1].mean(dim=0, keepdim=True)
            )
        )
        # Use soft labels if given the pseudo labels as base_labels
        if pseudo:
            soft_gt = torch.sum(
                base_labels_binary[compositional_labels[:, i] == 1], dim=0
            ) / len(torch.nonzero(compositional_labels[:, i] == 1))
            top_soft = torch.argsort(soft_gt, descending=True)[:2]
            # print(top_soft)
            composed = torch.zeros_like(soft_gt).int()
            composed[top_soft] = 1
        else:
            composed = (
                torch.prod(base_labels[compositional_labels[:, i] == 1], dim=0) == 1
            )

        gt_labels.append(composed)
        idxs.append(i)

    gt_composed_concepts = torch.cat(gt_composed_concepts)
    gt_labels = torch.stack(gt_labels)

    if True or complete_concept_ids is None:
        subset_complete = torch.ones(compositional_labels.shape[0]).bool()
    else:
        subset_complete = (
            torch.sum(compositional_labels[:, complete_concept_ids] == 1, dim=1) > 0
        ).flatten()
    # subset_compositional = np.array([0, 4, 8])
    # subset = (torch.sum(compositional_labels[:, [0, 4, 8]] == 1, dim=1) > 0).flatten()
    composed_concepts = []
    comp_score, coeffs = compositionality_eval(
        train_emb[subset_complete],
        concepts,
        base_labels_binary[subset_complete],
        return_coeffs=True,
    )
    # print(coeffs)
    for i, idx in enumerate(idxs):
        coeffs_curr = torch.mean(
            coeffs[compositional_labels[subset_complete][:, idx] == 1],
            dim=0,
            keepdim=True,
        )
        if torch.sum(coeffs_curr.T[gt_labels[i]]) == 0:
            composed_concepts.append(torch.zeros_like(gt_composed_concepts[i][None, :]))
            aucs.append(0)
            cos.append(0)
            continue
        composed = normalize(coeffs_curr.T[gt_labels[i]].T @ concepts[gt_labels[i]])
        composed_concepts.append(composed)
        aucs.append(
            average_precision_score(
                compositional_labels_test[:, idx], cosim(test_emb, composed)
            )
        )
        cos.append(composed @ gt_composed_concepts[i])

    return torch.cat(composed_concepts), aucs, cos, coeffs, gt_labels, comp_score


def find_coeffs(concepts, labels, embeddings):
    _, coeffs = compositionality_eval(embeddings, concepts, labels, return_coeffs=True)
    return coeffs
