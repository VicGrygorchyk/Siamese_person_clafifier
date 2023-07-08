from typing import TYPE_CHECKING

from numpy import average
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.nn import Module, BCELoss
from mlflow import log_metric, log_param
import lightning.pytorch as pl

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module
    from torch.optim import Optimizer
    from dataset import PersonsImages

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
CLS_THRESHOLD = 0.5


def get_accuracy(logit: 'Tensor', labels: 'Tensor'):
    print("logit ", logit)
    print("label ", labels)

    pred = (logit > CLS_THRESHOLD).float()
    print(f"predicted {pred}")
    return (pred == labels).sum().item() / len(labels)


class ModelTrainingWrapper(pl.LightningModule):

    def __init__(self,
                 backbone: 'Module',
                 batch_size: int,
                 dataset: 'PersonsImages',
                 save_chpt_path: str,
                 learning_rate: float = LEARNING_RATE,
                 weight_decay: float = WEIGHT_DECAY):
        super().__init__()
        # self.save_hyperparameters(ignore=['backbone', 'dataset'])
        self.backbone = backbone
        self.batch_size = batch_size
        self.dataset = dataset
        self.train_ds, self.valid_ds, self.test_ds = random_split(dataset, [0.7, 0.15, 0.15])  # type: PersonsImages
        log_param('starting learning rate', LEARNING_RATE)
        log_param('weight decay', WEIGHT_DECAY)
        log_param('Loss function', 'BCELoss')
        self.criterion = BCELoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_chpt_path = save_chpt_path
        self.eval_loss = 0.0
        self.eval_accuracy = []
        self.test_loss = 0.0
        self.test_accuracy = []

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, shuffle=False, batch_size=12, num_workers=10)

    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False, batch_size=12, num_workers=10)

    def training_step(self, batch, batch_idx):
        lbl_images, target_imgs, labels = batch
        logits = self.backbone(lbl_images, target_imgs)

        logits = logits.squeeze(dim=1)

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
        logits = logits.squeeze(dim=1)

        loss = self.criterion(logits, labels.float())
        loss_item = loss.item()
        log_metric('eval loss', loss, batch_idx)
        self.eval_loss += loss_item
        self.eval_accuracy.append(get_accuracy(logits, labels))
        self.log("eval_loss", loss, prog_bar=True)

    def on_validation_start(self) -> None:
        self.eval_loss = 0
        self.eval_accuracy = []

    def on_validation_epoch_end(self) -> None:
        eval_loss = self.eval_loss / len(self.val_dataloader())
        self.log("Validation loss", eval_loss)
        self.log("Validation accuracy", average(self.eval_accuracy))
        print("Validation accuracy ", average(self.eval_accuracy))

    def test_step(self, batch, batch_idx):
        lbl_images, target_imgs, labels = batch
        logits = self.backbone(lbl_images, target_imgs)
        logits = logits.squeeze(dim=1)
        labels = torch.squeeze(labels, dim=1)
        loss = self.criterion(logits, labels.float())
        log_metric('test loss', loss, batch_idx)

        self.test_loss += loss.item()
        self.test_accuracy.append(get_accuracy(logits, labels))
        self.log("test_loss", loss, prog_bar=True)

    def on_test_start(self) -> None:
        self.test_loss = 0
        self.test_accuracy = []

    def on_test_epoch_end(self) -> None:
        test_loss = self.test_loss / len(self.test_dataloader())
        self.log("Test loss", test_loss)
        self.log("Test accuracy", average(self.test_accuracy))
        print("Test accuracy ", average(self.test_accuracy))
