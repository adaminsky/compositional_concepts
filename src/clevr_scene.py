# Parse a CLEVR scene file into a set of features.
# The features are either representative of entity concept presence or relation
# and attribute concept presence.
from __future__ import annotations

import glob
import json
import math
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class CLEVRSample:
    def __init__(self, root, sample_data, scene_name):
        self.sample_data = sample_data
        self.objects = []
        self.relations = []
        try:
            self.img = Image.open(root + sample_data["image_filename"]).convert("RGB")
        except:
            print("Image not found: ", scene_name)

        for object in sample_data["objects"]:
            self.objects.append(
                (object["shape"], object["size"], object["material"], object["color"])
            )

        for relation in ["left", "right", "front", "behind"]:
            for obj_id in range(len(self.objects)):
                if len(sample_data["relationships"][relation][obj_id]) > 0:
                    self.relations.append(
                        (
                            relation,
                            obj_id,
                            sample_data["relationships"][relation][obj_id][0],
                        )
                    )


def standard(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def create_relation_features(samples: list[CLEVRSample], max_objects=3):
    relation_features = []
    for sample in samples:
        relation_feature = []
        relation_feature.append(1)
        for obj1_id in range(max_objects):
            for obj2_id in range(max_objects):
                if obj1_id == obj2_id:
                    continue
                for relation in ["left", "right", "front", "behind"]:
                    if (relation, obj1_id, obj2_id) in sample.relations:
                        relation_feature.append(1)
                    else:
                        relation_feature.append(0)
        relation_features.append(relation_feature)

    return np.array(relation_features)


def create_entity_features(samples: list[CLEVRSample], max_objects=3):
    entity_features = []
    for _ in samples:
        entity_features.append([1])
    return np.array(entity_features)


def get_count_labels(samples: list[CLEVRSample]):
    count_labels = []
    for sample in samples:
        count_labels.append(len(sample.objects))
    return np.array(count_labels)


def get_images(samples: list[CLEVRSample]):
    images = []
    for sample in samples:
        images.append(sample.img)
    return images


def batch_inference(model, dataset, batch_size=128, resize=None, device="cuda"):
    nb_batchs = math.ceil(len(dataset) / batch_size)
    start_ids = [i * batch_size for i in range(nb_batchs)]

    results = []

    with torch.no_grad():
        for i in tqdm(start_ids):
            x = torch.tensor(dataset[i : i + batch_size])
            x = x.to(device)

            if resize:
                x = torch.nn.functional.interpolate(
                    x, size=resize, mode="bilinear", align_corners=False
                )

            results.append(model(x).cpu())

    results = torch.cat(results)
    return results


def counting_experiment():
    scene_dir = "../clevr-dataset-gen/counting_data/scenes/*.json"
    all_scenes = glob.glob(scene_dir)
    samples = []
    for scene in all_scenes:
        with open(scene) as f:
            samples.append(
                CLEVRSample(
                    "../clevr-dataset-gen/counting_data/images/", json.load(f), scene
                )
            )
    print(len(samples))

    entity_features = create_entity_features(samples)
    relation_features = create_relation_features(samples)
    count_labels = get_count_labels(samples)
    images = get_images(samples)

    # Train a logistic regression model to predict the number of objects in a scene
    train_e, test_e, train_labels_e, test_labels_e, train_imgs_e, test_imgs_e = (
        train_test_split(entity_features, count_labels, images, test_size=0.2)
    )
    train_r, test_r, train_labels_r, test_labels_r = train_test_split(
        relation_features, count_labels, test_size=0.2
    )

    model_e = LogisticRegression().fit(train_e, train_labels_e)
    model_r = LogisticRegression().fit(train_r, train_labels_r)

    # ViT
    from transformers import AutoProcessor, CLIPModel

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = model.eval()

    def clip_embed(imgs, masks=None):
        with torch.no_grad():
            # Select the CLS token embedding from the last hidden layer
            return model.get_image_features(pixel_values=imgs)

    train_emb_e = batch_inference(
        clip_embed,
        processor(images=train_imgs_e, return_tensors="pt")["pixel_values"],
        batch_size=128,
        resize=(224, 224),
        device=device,
    )
    test_emb_e = batch_inference(
        clip_embed,
        processor(images=test_imgs_e, return_tensors="pt")["pixel_values"],
        batch_size=128,
        resize=(224, 224),
        device=device,
    )

    concept_text = [
        "An image with one gray cube",
        "An image with two gray cubes",
        "An image with three gray cubes",
    ]
    with torch.no_grad():
        clip_zs_concepts = torch.cat(
            [
                model.get_text_features(
                    **processor(text=[concept], return_tensors="pt").to(device)
                ).cpu()
                for concept in concept_text
            ]
        )
    clip_zs_concepts = clip_zs_concepts.cpu()
    clip_zs_concepts = clip_zs_concepts / clip_zs_concepts.norm(dim=-1, keepdim=True)

    # Train a logistic regression model to predict the number of objects in a scene based on CLIP embeddings
    clip_probe = LogisticRegression().fit(train_emb_e, train_labels_e)
    print(
        "CLIP Probe accuracy: ",
        accuracy_score(test_labels_e, clip_probe.predict(test_emb_e)),
    )

    # Evaluation zero-shot clip accuracy
    clip_zs_act = test_emb_e @ clip_zs_concepts.T
    clip_zs_pred = clip_zs_act.argmax(dim=1) + 1
    print("CLIP Zero-shot accuracy: ", accuracy_score(test_labels_e, clip_zs_pred))
    print(test_labels_e)
    print(clip_zs_pred)

    print(
        "Entity model accuracy: ",
        accuracy_score(test_labels_e, model_e.predict(test_e)),
    )
    print(
        "Relation model accuracy: ",
        accuracy_score(test_labels_r, model_r.predict(test_r)),
    )


def get_grammar_labels(samples: list[CLEVRSample], depth=1):
    # S -> base_concepts_i | bind base_concepts_i S | rel_concepts_i S_1 S_2 | bind (rel_concepts_i O1) S
    base_concepts = {
        "object": lambda obj, sample: True,
        "red": lambda obj, sample: sample.objects[obj][3] == "red",  # red
        "green": lambda obj, sample: sample.objects[obj][3] == "green",  # green
        "blue": lambda obj, sample: sample.objects[obj][3] == "blue",  # blue
        "sphere": lambda obj, sample: sample.objects[obj][0] == "sphere",  # sphere
        "cube": lambda obj, sample: sample.objects[obj][0] == "cube",  # cube
        "cylinder": lambda obj, sample: sample.objects[obj][0]
        == "cylinder",  # cylinder
    }
    bind = lambda c1, c2: lambda obj, sample: c1(obj, sample) and c2(obj, sample)
    rel_concepts = {
        "right of": lambda c1, c2: lambda object, sample: any(
            [
                (relation[0] == "right")
                and c1(relation[1], sample)
                and c2(relation[2], sample)
                for relation in sample.relations
            ]
        ),
        "left of": lambda c1, c2: lambda object, sample: any(
            [
                (relation[0] == "left")
                and c1(relation[1], sample)
                and c2(relation[2], sample)
                for relation in sample.relations
            ]
        ),
        "infront of": lambda c1, c2: lambda object, sample: any(
            [
                (relation[0] == "front")
                and c1(relation[1], sample)
                and c2(relation[2], sample)
                for relation in sample.relations
            ]
        ),
        "behind of": lambda c1, c2: lambda object, sample: any(
            [
                (relation[0] == "behind")
                and c1(relation[1], sample)
                and c2(relation[2], sample)
                for relation in sample.relations
            ]
        ),
    }

    # pass in a conjunction of concepts [c1, c2] and a sample to evaluate
    eval = lambda conj, sample: all(
        [
            any([concept(obj, sample) for obj in range(len(sample.objects))])
            for concept in conj
        ]
    )

    # generate all possible conjunctions of concepts of depth
    def rec_gen_concepts(depth, simple=False):
        if depth == 0:
            subconcepts = base_concepts.copy()
            if not simple:
                subconcepts.update(
                    {
                        f"({name1} and {name2}) object": bind(c1, c2)
                        for name1, c1 in subconcepts.items()
                        for name2, c2 in subconcepts.items()
                    }
                )
            return subconcepts
        else:
            new_concepts = {}
            subconcepts = rec_gen_concepts(depth - 1, simple=True)
            # new_concepts.update({f"({name1} and {name2}) object": bind(c1, c2) for name1, c1 in subconcepts.items() for name2, c2 in subconcepts.items()})
            new_concepts.update(subconcepts)
            for rel_name, rel_concept in rel_concepts.items():
                new_concepts.update(
                    {
                        f"({name1}) {rel_name} ({name2})": rel_concept(c1, c2)
                        for name1, c1 in subconcepts.items()
                        for name2, c2 in subconcepts.items()
                    }
                )
            return new_concepts

    concepts_attr = [rec_gen_concepts(0)]
    concepts_rel = [rec_gen_concepts(d - 1) for d in range(2, depth + 1)]
    concepts_all = concepts_attr + concepts_rel
    relations_mask = np.concatenate(
        [
            np.zeros(len(concepts_attr[0]), dtype=bool),
            np.ones(
                sum([len(concepts_rel[i]) for i in range(len(concepts_rel))]),
                dtype=bool,
            ),
        ],
        dtype=bool,
    )
    concepts = OrderedDict([(k, v) for d in concepts_all for k, v in d.items()])
    # subset = np.random.choice(len(concepts), min(1000, len(concepts)), replace=False)
    # subset = list(range(min(len(concepts), 1000)))
    # concept_subset = [sorted(list(concepts.keys()))[i] for i in subset]
    # concepts = {k: concepts[k] for k in concept_subset}
    concept_subset = list(concepts.keys())
    print(concept_subset[:10])
    print(len(concepts))
    # concepts = []
    # seen = set()
    # for i in range(depth):
    #     if i == 0:
    #         concepts = base_concepts
    #         seen = set(concepts)
    #     else:
    #         new_concepts = []
    #         seen = set()
    #         for c1 in concepts:
    #             for c2 in base_concepts:
    #                 if c1 != c2 and (c1, c2) not in seen:
    #                     new_concepts.append(lambda obj, c1=c1, c2=c2: bind(c1, c2, obj))
    #                     seen.add((c1, c2))
    #                     seen.add((c2, c1))
    #         concepts = new_concepts

    # # add conjunctions
    # for c1 in concepts:
    #     for c2 in concepts:
    #         concepts.append([c1, c2])
    labels = []
    for sample in samples:
        label = []
        for concept_name in concept_subset:
            label.append(eval([concepts[concept_name]], sample))
        labels.append(np.array(label))
    print("finished generating labels")

    labels = np.stack(labels).astype(int)
    if "green" in concept_subset and "(green and green) object" in concept_subset:
        # check if they are equal
        print("checking if green and object is equal to (green and green) object")
        assert np.all(
            labels[:, concept_subset.index("green")]
            == labels[:, concept_subset.index("(green and green) object")]
        )

    uniq = np.sort(np.unique(labels, axis=1, return_index=True)[1])
    return labels[:, uniq], np.array(concept_subset)[uniq], relations_mask[uniq]


def get_bow_labels(samples: list[CLEVRSample]):
    labels = []
    for sample in samples:
        label = np.zeros(6)  # red, green, blue, sphere, cube, cylinder
        for obj in sample.objects:
            if obj[3] == "red":
                label[0] = 1
            elif obj[3] == "green":
                label[1] = 1
            elif obj[3] == "blue":
                label[2] = 1
            if obj[0] == "sphere":
                label[3] = 1
            elif obj[0] == "cube":
                label[4] = 1
            elif obj[0] == "cylinder":
                label[5] = 1
        labels.append(label)
    return np.stack(labels)


def get_better_bow_labels(samples: list[CLEVRSample]):
    labels = []
    for sample in samples:
        label = np.zeros(
            9
        )  # red sphere, red cube, red cylinder, green sphere, green cube, green cylinder, blue sphere, blue cube, blue cylinder
        for obj in sample.objects:
            if obj[3] == "red":
                if obj[0] == "sphere":
                    label[0] = 1
                elif obj[0] == "cube":
                    label[1] = 1
                elif obj[0] == "cylinder":
                    label[2] = 1
            elif obj[3] == "green":
                if obj[0] == "sphere":
                    label[3] = 1
                elif obj[0] == "cube":
                    label[4] = 1
                elif obj[0] == "cylinder":
                    label[5] = 1
            elif obj[3] == "blue":
                if obj[0] == "sphere":
                    label[6] = 1
                elif obj[0] == "cube":
                    label[7] = 1
                elif obj[0] == "cylinder":
                    label[8] = 1
        labels.append(label)
    return np.stack(labels)


if __name__ == "__main__":
    # load data
    scene_dir = "../clevr-dataset-gen/output/scenes/*.json"
    all_scenes = glob.glob(scene_dir)
    samples = []
    for scene in all_scenes:
        with open(scene) as f:
            samples.append(
                CLEVRSample("../clevr-dataset-gen/output/images/", json.load(f), scene)
            )
    print(len(samples))

    left_color_label = []
    left_shape_label = []
    cube_color_label = []
    cube_images = []
    for i, sample in enumerate(samples):
        for obj in sample.objects:
            if sum([1 for obj in sample.objects if obj[0] == "cube"]) > 1:
                break
            if obj[0] == "cube":
                cube_color_label.append(obj[3])
                cube_images.append(sample.img)
                break
        for relation in sample.relations:
            if relation[0] == "left":
                left_color_label.append(sample.objects[relation[1]][3])
                left_shape_label.append(sample.objects[relation[1]][0])
                break

    left_color_label = np.array(left_color_label)
    left_shape_label = np.array(left_shape_label)
    images = get_images(samples)

    # Printing dataset stats
    for l in np.unique(left_color_label):
        print(l, np.sum(left_color_label == l))
    for l in np.unique(left_shape_label):
        print(l, np.sum(left_shape_label == l))

    both_label = np.array(
        [
            f"{z[0]} {z[1]}"
            for z in zip(left_color_label.tolist(), left_shape_label.tolist())
        ]
    )
    for l in np.unique(both_label):
        print(l, np.sum(both_label == l))

    # Train a logistic regression model to predict the color of the object to the left
    (
        train_imgs,
        test_imgs,
        train_labels_color,
        test_labels_color,
        train_labels_shape,
        test_labels_shape,
    ) = train_test_split(images, left_color_label, left_shape_label, test_size=0.2)

    # ViT
    from transformers import AutoProcessor, CLIPModel

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = model.eval()

    def clip_embed(imgs, masks=None):
        with torch.no_grad():
            # Select the CLS token embedding from the last hidden layer
            return model.get_image_features(pixel_values=imgs)

    train_emb_e = standard(
        batch_inference(
            clip_embed,
            processor(images=train_imgs, return_tensors="pt")["pixel_values"],
            batch_size=128,
            resize=(224, 224),
            device=device,
        )
    )
    test_emb_e = standard(
        batch_inference(
            clip_embed,
            processor(images=test_imgs, return_tensors="pt")["pixel_values"],
            batch_size=128,
            resize=(224, 224),
            device=device,
        )
    )

    clf = LogisticRegression().fit(train_emb_e, train_labels_color)
    color_concepts = standard(clf.coef_)
    print(
        "CLIP Probe accuracy for predicting color on left: ",
        accuracy_score(test_labels_color, clf.predict(test_emb_e)),
    )
    print("Random accuracy: ", 1 / len(np.unique(left_color_label)))

    clf = LogisticRegression().fit(train_emb_e, train_labels_shape)
    shape_concepts = standard(clf.coef_)
    print(
        "CLIP Probe accuracy for predicting shape on left: ",
        accuracy_score(test_labels_shape, clf.predict(test_emb_e)),
    )
    print("Random accuracy: ", 1 / len(np.unique(left_shape_label)))

    train_imgs, test_imgs, train_labels_ccolor, test_labels_ccolor = train_test_split(
        cube_images, cube_color_label, test_size=0.2
    )
    train_emb_e = standard(
        batch_inference(
            clip_embed,
            processor(images=train_imgs, return_tensors="pt")["pixel_values"],
            batch_size=128,
            resize=(224, 224),
            device=device,
        )
    )
    test_emb_e = standard(
        batch_inference(
            clip_embed,
            processor(images=test_imgs, return_tensors="pt")["pixel_values"],
            batch_size=128,
            resize=(224, 224),
            device=device,
        )
    )

    clf = LogisticRegression().fit(train_emb_e, train_labels_ccolor)
    cube_color_concepts = standard(clf.coef_)
    print(
        "CLIP Probe accuracy for predicting color on cube: ",
        accuracy_score(test_labels_ccolor, clf.predict(test_emb_e)),
    )

    print(color_concepts @ color_concepts.T)
    print(shape_concepts @ shape_concepts.T)
    print(color_concepts @ shape_concepts.T)
    print(color_concepts @ cube_color_concepts.T)
