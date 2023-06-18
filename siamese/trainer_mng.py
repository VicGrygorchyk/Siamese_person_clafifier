from typing import TYPE_CHECKING

import torch
from accelerate import Accelerator
from torch.optim import AdamW, lr_scheduler
from torch.nn import BCEWithLogitsLoss, Module, MSELoss
from mlflow import log_metric, log_param
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module
    from torch.utils.data import DataLoader
    from torch.optim import Optimizer

LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
CLS_THRESHOLD = 0.5


def get_accuracy(logit1: 'Tensor', logit2: 'Tensor', label: 'Tensor'):

    # if label is 0 - get item from of similar items, else different
    inverted = torch.where(label == 0, 1, 0)
    print("label ", label)

    merged = logit1 * inverted[:, None] + logit2 * label[:, None]
    print("merged sum ", merged.sum(1))

    pred = torch.where(merged > CLS_THRESHOLD, 1, 0)
    pred = torch.squeeze(pred)
    print(f"predicted {pred}")
    return pred.eq(label).sum().item() / len(label)


class TrainerManager:

    def __init__(self, model: 'Module', save_dir: str,
                 train_dl: 'DataLoader', valid_dl: 'DataLoader', test_dl: 'DataLoader',
                 num_epochs: int = 20):
        self.model = model
        self.save_dir = save_dir
        log_param('starting learning rate', LEARNING_RATE)
        log_param('weight decay', WEIGHT_DECAY)
        self.optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        log_param('Optimizer', 'AdamW')
        self.criterion = BCEWithLogitsLoss()
        log_param('Loss function', 'BCEWithLogitsLoss')
        self.train_dataloader = train_dl
        self.eval_dataloader = valid_dl
        self.test_dataloader = test_dl

        self.lr_scheduler = lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=5,
            gamma=0.1
        )
        log_param('LR scheduler ', 'lr_scheduler.StepLR')

        self.accelerator = Accelerator()
        # override model, optim and dataloaders to allow Accelerator to autohandle `device`
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, \
            self.test_dataloader, self.criterion, self.lr_scheduler = \
            self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.eval_dataloader,
                self.test_dataloader,
                self.criterion,
                self.lr_scheduler
            )  # type: Module, Optimizer, DataLoader, DataLoader, DataLoader, BCEWithLogitsLoss, lr_scheduler.StepLR
        len_train_dataloader = len(self.train_dataloader)
        # log_metric('Length of training dataloader', len_train_dataloader)
        num_update_steps_per_epoch = len_train_dataloader
        self.num_epochs = num_epochs
        self.num_training_steps = self.num_epochs * num_update_steps_per_epoch
        # create scheduler with changing learning rate

    def run(self):
        start_eval_loss = 100
        for epoch in range(self.num_epochs):
            print(f'EPOCH {epoch}')
            self.train(epoch)
            eval_loss = self.evaluate()
            # save the model if current eval loss is better than prev
            if eval_loss < start_eval_loss:
                torch.save(self.model.state_dict(), self.save_dir)
                start_eval_loss = eval_loss

    def train(self, epoch):
        self.model.train(True)
        train_loss = 0.0
        step = 1
        for data in tqdm(self.train_dataloader):
            lbl_image, same_img, diff_img, label = data

            logits1 = self.model(lbl_image, same_img)
            logits2 = self.model(lbl_image, diff_img)

            logits_combined = torch.concat([logits1, logits2], dim=0)
            # squeeze as logits are of shape (batch, 1) labels (batch, )
            logits_combined = torch.squeeze(logits_combined).float()

            labels_combined = torch.concat(
                [torch.zeros((logits1.shape[0], )), torch.ones((logits2.shape[0]), )],
                dim=0
            ).to('cuda')
            labels_combined = labels_combined.float()

            self.optimizer.zero_grad()

            loss = self.criterion(logits_combined, labels_combined)
            loss_item = loss.item()
            log_metric('train loss', loss_item, step)
            train_loss += loss_item

            loss.backward()
            self.optimizer.step()
            # self.progress_bar.update(1)
            step += 1
        self.lr_scheduler.step(epoch)
        fin_loss = train_loss / len(self.train_dataloader)
        print(f"Total train loss is {fin_loss}")

    def evaluate(self):
        eval_loss = 0
        correct = 0
        step = 1
        devider = 1

        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.eval_dataloader):
                lbl_image, similar_image, diff_image, label = data

                logits1 = self.model(lbl_image, similar_image)
                logits2 = self.model(lbl_image, diff_image)

                loss = self.criterion(logits1, logits2)
                loss_item = loss.item()
                log_metric('eval loss', loss, step)
                eval_loss += loss_item

                correct = (correct + get_accuracy(logits1, logits2, label)) / devider
                devider = 2

                log_metric('eval accuracy', correct, step)
                step += 1
        eval_loss = eval_loss / len(self.eval_dataloader)
        print(f"Eval loss {eval_loss}. Accuracy: {correct}%")
        return eval_loss

    def test(self):
        test_loss = 0.0
        correct = 0.0
        devider = 1
        step = 1
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.test_dataloader):
                lbl_image, similar_image, diff_image, label = data

                logits1 = self.model(lbl_image, similar_image)
                logits2 = self.model(lbl_image, diff_image)

                loss = self.criterion(logits1, logits2)
                log_metric('eval loss', loss, step)

                test_loss += loss.item()
                correct = (correct + get_accuracy(logits1, logits2, label)) / devider
                devider = 2
                step += 1
        test_loss = test_loss / len(self.test_dataloader)
        print(f"Test loss {test_loss}. Accuracy: {correct}%")
