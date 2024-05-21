import os
from typing import TYPE_CHECKING

from numpy import average

import evaluate
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.nn import BCELoss
from mlflow import log_metric, log_param
import mlflow
from mlflow.models import infer_signature
import lightning.pytorch as pl

from siamese.dataset import PersonsImages
from siamese.model import SiameseNN

if TYPE_CHECKING:
    from torch import Tensor
    from torch.optim import Optimizer


START_BATCH_SIZE = 112

DATASET_PATH = os.getenv("DATASET_PATH")
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.001
CLS_THRESHOLD = 0.5

accuracy = evaluate.load("accuracy")


def get_accuracy(logit: 'Tensor', labels: 'Tensor') -> (float, float, float):
    print("logit ", logit.tolist())
    print("label ", labels.tolist())
    pred = (logit > CLS_THRESHOLD).float()
    print(f"predicted {pred.tolist()}")
    acc_similar = (pred == labels).sum().item() / len(labels)
    print(f'acc of acc_similar {acc_similar}')

    return acc_similar


class ModelTrainingWrapper(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone', 'dataset', 'save_chpt_path'])
        self.backbone = SiameseNN()
        self.batch_size = START_BATCH_SIZE
        self.dataset = PersonsImages(DATASET_PATH)
        self.train_ds, self.valid_ds, self.test_ds = (
            random_split(self.dataset, [0.7, 0.18, 0.12])
        )  # type: PersonsImages
        log_param('starting learning rate', LEARNING_RATE)
        log_param('weight decay', WEIGHT_DECAY)
        log_param('Loss function', 'BCELoss')
        self.criterion = BCELoss()
        self.learning_rate = LEARNING_RATE
        self.weight_decay = WEIGHT_DECAY
        self.eval_loss = 0.0
        self.eval_accuracy_similarity = []
        self.test_loss = 0.0
        self.test_accuracy_similarity = []

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, shuffle=False, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False, batch_size=12, num_workers=8)

    def training_step(self, batch, batch_idx):
        lbl_images, target_imgs, labels = batch
        logits = self.backbone(lbl_images, target_imgs)

        labels = labels.view_as(logits)

        loss = self.criterion(logits, labels.float())
        loss_item = loss.item()
        log_metric('train loss', loss_item, batch_idx)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> 'Optimizer':
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        log_param('Optimizer', 'AdamW')
        return optimizer

    def validation_step(self, batch, batch_idx):
        lbl_images, target_imgs, labels = batch
        logits = self.backbone(lbl_images, target_imgs)

        labels = labels.view_as(logits)

        loss = self.criterion(logits, labels.float())
        loss_item = loss.item()
        log_metric('eval loss', loss, batch_idx)
        self.eval_loss += loss_item
        acc = get_accuracy(logits, labels)
        self.eval_accuracy_similarity.append(acc)
        self.log("eval_loss", loss, prog_bar=True)

    def on_validation_start(self) -> None:
        self.eval_loss = 0
        self.eval_accuracy_similarity = []

    def on_validation_epoch_end(self) -> None:
        eval_loss = self.eval_loss / len(self.val_dataloader())
        self.log("Validation loss", eval_loss)
        self.log("Validation accuracy for similarity", average(self.eval_accuracy_similarity))

        # if eval(os.getenv("LOG_MODEL_TO_MLFLOW")):
        #     predictions = trainer.predict()
        #     signature = infer_signature(X_test, predictions)
        #     mlflow.pytorch.log_model(rf, "model", signature=signature)

    def test_step(self, batch, batch_idx):
        lbl_images, target_imgs, labels = batch
        logits = self.backbone(lbl_images, target_imgs)
        labels = labels.view_as(logits)

        loss = self.criterion(logits, labels.float())
        log_metric('test loss', loss, batch_idx)

        self.test_loss += loss.item()
        acc = get_accuracy(logits, labels)
        self.test_accuracy_similarity.append(acc)

    def on_test_start(self) -> None:
        self.test_loss = 0
        self.test_accuracy_similarity = []

    def on_test_epoch_end(self) -> None:
        test_loss = self.test_loss / len(self.test_dataloader())
        self.log("Test loss", test_loss)
        self.log("Test accuracy for similarity", average(self.test_accuracy_similarity))

    def forward(self, x1, x2):
        return self.backbone(x1, x2)
