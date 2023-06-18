from torch.utils.data import DataLoader, random_split
from mlflow import start_run

from model import SiameseNN
from dataset import CelebImages
from trainer_mng import TrainerManager

EPOCH = 25


if __name__ == "__main__":
    # dataset
    dataset = CelebImages('/home/mudro/Documents/Projects/siamese/labels_data.json')

    with start_run(description=f"Run with epoch ${EPOCH} and Custom Triplet loss, small dataset"):
        train_ds, valid_ds, test_ds = random_split(dataset, [0.8, 0.12, 0.08])  # type: CelebImages

        # dataloader
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=12)
        valid_dl = DataLoader(valid_ds, shuffle=False, batch_size=18)
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=18)

        # model
        model = SiameseNN()
        # train
        trainer = TrainerManager(
            model,
            '/home/mudro/Documents/Projects/siamese/saved_model/v5_with_dot_attn2_simplified.pt',
            train_dl, valid_dl, test_dl, num_epochs=EPOCH
        )
        trainer.run()
        trainer.test()
