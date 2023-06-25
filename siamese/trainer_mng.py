from typing import TYPE_CHECKING

import torch
from numpy import average
from accelerate import Accelerator
from torch.optim import AdamW, lr_scheduler
from torch.nn import Module, BCELoss
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


def get_accuracy(logit: 'Tensor', label: 'Tensor'):
    label = torch.squeeze(label)
    print("label ", label)

    pred = torch.where(logit > CLS_THRESHOLD, 1, 0)
    pred = torch.squeeze(pred)
    print(f"predicted {pred}")
    return (pred == label).sum().item() / len(label)


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
        self.criterion = BCELoss()
        log_param('Loss function', 'BCELoss')
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
            )  # type: Module, Optimizer, DataLoader, DataLoader, DataLoader, BCELoss, lr_scheduler.StepLR
        len_train_dataloader = len(self.train_dataloader)
        # log_metric('Length of training dataloader', len_train_dataloader)
        num_update_steps_per_epoch = len_train_dataloader
        self.num_epochs = num_epochs
        self.num_training_steps = self.num_epochs * num_update_steps_per_epoch

    def run(self):
        start_eval_loss = torch.inf
        for epoch in range(self.num_epochs):
            print(f'EPOCH {epoch}')
            self.train(epoch)
            eval_loss = self.evaluate()
            # save the model if current eval loss is better than prev
            if eval_loss < start_eval_loss:
                self.accelerator.wait_for_everyone()
                unwrapped_model = self.accelerator.unwrap_model(self.model)

                print(f'Saving a new version of the unwrapped_model {self.save_dir}')
                torch.save(unwrapped_model.state_dict(), self.save_dir)
                start_eval_loss = eval_loss

    def train(self, epoch):
        self.model.train(True)
        train_loss = 0.0
        step = 1
        for data in tqdm(self.train_dataloader):
            with self.accelerator.accumulate(self.model):
                lbl_images, target_imgs, labels = data
                labels = torch.unsqueeze(labels, 1)
                logits = self.model(lbl_images, target_imgs)

                self.optimizer.zero_grad()

                loss = self.criterion(logits, labels.float())
                loss_item = loss.item()
                log_metric('train loss', loss_item, step)
                train_loss += loss_item

                self.accelerator.backward(loss)
                self.optimizer.step()

                step += 1
        self.lr_scheduler.step(epoch)
        fin_loss = train_loss / len(self.train_dataloader)
        print(f"Total train loss is {fin_loss}")

    def evaluate(self):
        eval_loss = 0
        acc = []
        step = 1

        self.model.eval()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        with torch.no_grad():
            for data in tqdm(self.eval_dataloader):
                lbl_images, target_imgs, labels = data
                labels = torch.unsqueeze(labels, 1)
                logits = unwrapped_model(lbl_images, target_imgs)

                loss = self.criterion(logits, labels.float())
                loss_item = loss.item()
                log_metric('eval loss', loss, step)
                eval_loss += loss_item

                acc.append(get_accuracy(logits, labels))

                log_metric('eval accuracy', average(acc), step)
                step += 1
        eval_loss = eval_loss / len(self.eval_dataloader)
        print(f"Eval loss {eval_loss}. Accuracy: {average(acc)}")
        return eval_loss

    def test(self):
        test_loss = 0.0
        acc = []
        step = 1
        self.model.eval()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        with torch.no_grad():
            for data in tqdm(self.test_dataloader):
                lbl_image, target_imgs, labels = data
                labels = torch.unsqueeze(labels, 1)
                logits = unwrapped_model(lbl_image, target_imgs)

                loss = self.criterion(logits, labels.float())
                log_metric('eval loss', loss, step)

                test_loss += loss.item()
                acc.append(get_accuracy(logits, labels))

                step += 1
        test_loss = test_loss / len(self.test_dataloader)
        print(f"Test loss {test_loss}. Accuracy: {average(acc)}")
