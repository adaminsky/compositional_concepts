from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from concept_transformer.vit import ConceptTransformer_model
from CUB_dataset_joint import CUB2011Parts_dataset
from torch.nn import functional as F

# import albumentations as A
# from albumentations.augmentations.geometric.resize import Resize
# from albumentations.augmentations.geometric.rotate import Rotate
# from albumentations.augmentations.transforms import HorizontalFlip, Normalize
# from albumentations.pytorch.transforms import ToTensorV2


class unsupervised_cl_baselines:
    def __init__(self, num_concepts, num_classes, **kwargs):
        self.num_concepts = num_concepts
        self.num_classes = num_classes

    def train_models_with_concepts(self, X):
        pass

    def compute_sample_activations(self, X):
        pass

    def return_concepts(self):
        pass


def eval_models_with_concepts(model, dataloader, device=torch.device("cuda")):
    model.eval()
    with torch.no_grad():
        logits_ls = []
        labels_ls = []
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            logits_ls.append(logits.detach().cpu())
            labels_ls.append(y)

        logits = torch.cat(logits_ls, dim=0)
        labels = torch.cat(labels_ls, dim=0)
        # compute accuracy
        acc = (logits.argmax(dim=1) == labels).float().mean().item()
        # acc = (logits.argmax(dim=1) == Y).float().mean()
        # print(f"Accuracy: {acc:.4f}")

    return acc


def train_models_with_concepts(
    log_dir,
    model,
    train_loader,
    valid_loader,
    test_loader,
    epochs=100,
    device=torch.device("cuda"),
):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    best_val_acc = 0
    best_test_acc = 0
    best_test_acc2 = 0
    for epoch in range(epochs):
        model.train()
        average_loss = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            optimizer.zero_grad()
            loss = F.nll_loss(F.log_softmax(logits, dim=-1), y)
            loss.backward()
            optimizer.step()
            average_loss += loss.item()

        average_loss /= len(train_loader)
        print("average training loss::", average_loss)

        val_acc = eval_models_with_concepts(model, valid_loader, device=device)
        test_acc = eval_models_with_concepts(model, test_loader, device=device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(log_dir, "model_best.pt"))
        if best_test_acc2 < test_acc:
            best_test_acc2 = test_acc

        print(
            f"Epoch {epoch} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | Best Val Acc: {best_val_acc:.4f} | Best Test Acc: {best_test_acc:.4f}"
        )
        print(f"Epoch {epoch} | Best Test Acc: {best_test_acc2:.4f}")

    model.load_state_dict(torch.load(os.path.join(log_dir, "model_best.pt")))


class concept_transformer_learner(unsupervised_cl_baselines):
    def __init__(
        self,
        num_concepts,
        num_classes,
        device=torch.device("cuda"),
        epochs=100,
        batch_size=32,
        seed=0,
        log_dir=None,
        embedding_dim=768,
    ):
        super().__init__(num_concepts, num_classes)
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = ConceptTransformer_model(
            num_classes=num_classes,
            n_concepts=num_concepts,
            embedding_dim=embedding_dim,
        )
        self.device = device
        self.seed = seed
        self.model.to(device)
        self.train_transform = None
        # A.Compose([Resize(224, 224),
        #                                   HorizontalFlip(p=0.5),
        #                                   Rotate(limit=(-30,30),p=1.0),
        #                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #                                   ToTensorV2()])
        # ,
        #                                  keypoint_params = A.KeypointParams(format='xy', remove_invisible=False))

        self.test_transform = None
        # A.Compose([Resize(224, 224),
        #                                  Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #                                  ToTensorV2()])
        # ,
        #                                 keypoint_params = A.KeypointParams(format='xy', remove_invisible=False))

        self.log_dir = log_dir

    def training(
        self,
        train_images,
        train_targets,
        valid_images,
        valid_targets,
        test_images,
        test_targets,
    ):
        self.train_dataset = CUB2011Parts_dataset(
            train_images, train_targets, transform=self.train_transform
        )
        self.valid_dataset = CUB2011Parts_dataset(
            valid_images, valid_targets, transform=self.test_transform
        )
        self.test_dataset = CUB2011Parts_dataset(
            test_images, test_targets, transform=self.test_transform
        )

        torch.manual_seed(self.seed)
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        train_models_with_concepts(
            self.log_dir,
            self.model,
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader,
            epochs=self.epochs,
            device=self.device,
        )

    def testing(self, test_images, test_targets):
        self.test_dataset = CUB2011Parts_dataset(
            test_images, test_targets, transform=self.test_transform
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )
        acc = eval_models_with_concepts(
            self.model, self.test_dataloader, device=self.device
        )
        return acc
        # self.model = self.model.fit(X, concepts)

    # def compute_sample_activations(self, X):
    #     sample_activation = self.model.transform(X)
    #     sample_activation = torch.from_numpy(sample_activation).float()
    #     return sample_activation

    def return_concepts(self):
        return self.model.concepts
