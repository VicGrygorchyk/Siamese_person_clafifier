from typing import TYPE_CHECKING

import torch
from accelerate import Accelerator
from torch.optim import AdamW, lr_scheduler
from torch.nn import BCELoss
from mlflow import log_metric, log_param
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module
    from torch.utils.data import DataLoader
    from torch.optim import Optimizer

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01


def get_accuracy(pred: 'Tensor', label: 'Tensor'):
    return pred.eq(label).sum().item() * 100 / len(label)


class TrainerManager:

    def __init__(self, model: 'Module', save_dir: str,
                 train_dl: 'DataLoader', valid_dl: 'DataLoader', test_dl: 'DataLoader',
                 num_epochs: int = 20):
        self.model = model
        self.output_dir = save_dir
        # log_param('Optimizer', 'AdamW')
        # log_param('learning rate', LEARNING_RATE)
        # log_param('weight decay', WEIGHT_DECAY)
        self.optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.criterion = BCELoss()
        self.train_dataloader = train_dl
        self.eval_dataloader = valid_dl
        self.test_dataloader = test_dl

        self.accelerator = Accelerator()
        # override model, optim and dataloaders to allow Accelerator to autohandle `device`
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, \
            self.test_dataloader, self.criterion = \
            self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.eval_dataloader,
                self.test_dataloader,
                self.criterion
            )  # type: Module, Optimizer, DataLoader, DataLoader, DataLoader, BCELoss
        len_train_dataloader = len(self.train_dataloader)
        # log_metric('Length of training dataloader', len_train_dataloader)
        num_update_steps_per_epoch = len_train_dataloader
        self.num_epochs = num_epochs
        self.num_training_steps = self.num_epochs * num_update_steps_per_epoch
        # create scheduler with changing learning rate
        # self.lr_scheduler = lr_scheduler.StepLR(
        #     "linear",
        #     optimizer=self.optimizer,
        #     num_warmup_steps=100,
        #     num_training_steps=self.num_training_steps,
        # )
        # self.progress_bar = tqdm(range(self.num_training_steps))

    def run(self):
        for epoch in range(self.num_epochs):
            self.train()
            self.evaluate()

    def train(self):
        self.model.train(True)
        train_loss = 0.0
        for data in tqdm(self.train_dataloader):
            lbl_image, trgt_image, label = data
            logits = self.model(lbl_image, trgt_image)

            self.optimizer.zero_grad()

            # squeeze as logits are of shape (batch, 1) labels (batch, )
            logits = torch.squeeze(logits).float()
            label = label.float()
            loss = self.criterion(logits, label)
            train_loss += loss.item()

            loss.backward()
            self.optimizer.step()
            # self.progress_bar.update(1)
        fin_loss = train_loss / len(self.train_dataloader)
        print(f"Total train loss is {fin_loss}")

    def evaluate(self):
        eval_loss = 0
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.eval_dataloader):
                lbl_image, trgt_image, label = data
                logits = self.model(lbl_image, trgt_image)
                logits = torch.squeeze(logits).float()
                label = label.float()
                loss = self.criterion(logits, label)
                eval_loss += loss.item()
                pred = torch.where(logits > 0.5, 1, 0)
                correct = (correct + get_accuracy(pred, label)) / 2
        eval_loss = eval_loss / len(self.eval_dataloader)
        print(f"Eval loss {eval_loss}. Accuracy: {correct}%")

    def test(self):
        test_loss = 0.0
        correct = 0.0
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.test_dataloader):
                lbl_image, trgt_image, label = data
                logits = self.model(lbl_image, trgt_image)
                logits = torch.squeeze(logits).float()
                label = label.float()
                loss = self.criterion(logits, label)
                test_loss += loss.item()
                pred = torch.where(logits > 0.5, 1, 0)
                correct = (correct + get_accuracy(pred, label)) / 2
        test_loss = test_loss / len(self.test_dataloader)
        print(f"Test loss {test_loss}. Accuracy: {correct}%")
