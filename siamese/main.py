import os

from torch.utils.data import DataLoader, random_split
from torch import cuda
from mlflow import start_run

from model import SiameseNN
from dataset import PersonsImages
from trainer_mng import TrainerManager

EPOCH = 1


DATASET_PATH = os.getenv("DATASET_PATH")
SAVE_MODEL_PATH = os.getenv("SAVE_MODEL_PATH")


if __name__ == "__main__":
    cuda.empty_cache()
    # dataset
    dataset = PersonsImages(DATASET_PATH)

    with start_run(description=f"Run with epoch ${EPOCH} and Custom Triplet loss, small dataset"):
        train_ds, valid_ds, test_ds = random_split(dataset, [0.7, 0.15, 0.15])  # type: PersonsImages

        # dataloader
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=4)
        valid_dl = DataLoader(valid_ds, shuffle=False, batch_size=8)
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=8)

        # model
        model = SiameseNN()
        # train
        trainer = TrainerManager(
            model,
            SAVE_MODEL_PATH,
            train_dl, valid_dl, test_dl, num_epochs=EPOCH
        )
        trainer.run()
        trainer.test()
