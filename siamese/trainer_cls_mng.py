import os
from typing import TYPE_CHECKING, Dict, Any

import numpy as np

import evaluate
from transformers import AutoImageProcessor, DefaultDataCollator, AutoModelForImageClassification
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from mlflow import log_metric, log_param
import lightning.pytorch as pl

from siamese.dataset import HasHumanImages
from siamese.custom_types import HasFace

if TYPE_CHECKING:
    from torch.optim import Optimizer


START_BATCH_SIZE = 144

DATASET_PATH = os.getenv("DATASET_CSL_PATH")
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
CLS_THRESHOLD = 0.49
MODEL_SAVE_PATH = os.getenv('SAVE_CLS_MODEL_PATH')

accuracy = evaluate.load("accuracy")
checkpoint = os.getenv('CLS_MODEL_CHECKPOINT')


def compute_metrics(predictions, labels):
    predictions = torch.argmax(predictions, dim=1)
    res = accuracy.compute(predictions=predictions, references=labels)
    return res.get('accuracy', 0)


class ClsModelTrainingWrapper(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone', 'dataset', 'save_chpt_path'])
        self.image_processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.data_collator = DefaultDataCollator()
        labels = (HasFace.HAS_HUM_FACE.value, HasFace.IS_OTHER.value)

        label2id = {
            'HAS_HUM_FACE': str(HasFace.HAS_HUM_FACE.value),
            'OTHERS': str(HasFace.IS_OTHER.value),
        }
        id2label = {
            str(HasFace.HAS_HUM_FACE.value): 'HAS_HUM_FACE',
            str(HasFace.IS_OTHER.value): 'OTHERS',
        }

        self.backbone = AutoModelForImageClassification.from_pretrained(
            checkpoint,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )
        for child in list(self.backbone.children())[:-1]:
            for param in child.parameters():
                param.requires_grad = False

        self.batch_size = START_BATCH_SIZE
        self.dataset = HasHumanImages(DATASET_PATH, self.image_processor)

        self.train_ds, self.valid_ds, self.test_ds = (
            random_split(self.dataset, [0.7, 0.18, 0.12])
        )  # type: HasHumanImages
        log_param('starting learning rate', LEARNING_RATE)
        log_param('weight decay', WEIGHT_DECAY)
        log_param('Loss function', 'CrossEntropyLoss')
        self.criterion = CrossEntropyLoss()
        self.learning_rate = LEARNING_RATE
        self.weight_decay = WEIGHT_DECAY
        self.eval_loss = 0.0
        self.best_prev_eval_loss = 100.0
        self.eval_accuracy = []
        self.test_loss = 0.0
        self.test_accuracy = []

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, shuffle=False, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False, batch_size=12, num_workers=8)

    def _handle_batch_input(self, batch):
        images, labels = batch
        # images, labels = images.to(self.device), labels.to(self.device)
        logits = self.backbone(images)
        logits = logits.logits

        labels = labels
        return logits, labels

    def training_step(self, batch, batch_idx):
        logits, labels = self._handle_batch_input(batch)

        loss = self.criterion(logits, labels)
        loss_item = loss.item()
        log_metric('train loss', loss_item, batch_idx)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> 'Optimizer':
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        log_param('Optimizer', 'AdamW')
        return optimizer

    def validation_step(self, batch, batch_idx):
        logits, labels = self._handle_batch_input(batch)
        loss = self.criterion(logits, labels)
        loss_item = loss.item()
        log_metric('eval loss', loss, batch_idx)
        self.eval_loss += loss_item
        acc = compute_metrics(logits, labels)
        self.eval_accuracy.append(acc)

        self.log("eval_loss", loss, prog_bar=True)

    def on_validation_start(self) -> None:
        self.eval_loss = 0
        self.eval_accuracy = []

    def on_validation_epoch_end(self) -> None:
        eval_loss = self.eval_loss / len(self.val_dataloader())
        self.log("Validation loss", eval_loss)
        self.log("Validation accuracy", np.average(self.eval_accuracy))

        if eval_loss < self.best_prev_eval_loss:
            self.best_prev_eval_loss = eval_loss
            self.backbone.config.to_json_file(f"{MODEL_SAVE_PATH}/model_best_config.json")
            self.backbone.save_pretrained(f"{MODEL_SAVE_PATH}/model_best")
            self.image_processor.save_pretrained(f"{MODEL_SAVE_PATH}/model_best")

    def test_step(self, batch, batch_idx):
        logits, labels = self._handle_batch_input(batch)

        loss = self.criterion(logits, labels)
        log_metric('test loss', loss, batch_idx)

        self.test_loss += loss.item()
        acc = compute_metrics(logits, labels)
        self.test_accuracy.append(acc)

    def on_test_start(self) -> None:
        self.test_loss = 0
        self.test_accuracy = []

    def on_test_epoch_end(self) -> None:
        test_loss = self.test_loss / len(self.test_dataloader())
        self.log("Test loss", test_loss)
        self.log("Test accuracy", np.average(self.test_accuracy))
        self.backbone.config.to_json_file(f"{MODEL_SAVE_PATH}/config.json")
        self.backbone.save_pretrained(f"{MODEL_SAVE_PATH}/model")
        self.image_processor.save_pretrained(f"{MODEL_SAVE_PATH}/model")

    def forward(self, x):
        return self.backbone(x)
