from torch.utils.data import DataLoader, random_split

from model import SiameseNN
from dataset import CelebImages
from trainer_mng import TrainerManager

EPOCH = 20


if __name__ == "__main__":
    # dataset
    dataset = CelebImages('/home/mudro/Documents/Projects/siamese/data/labeling/')
    train_ds, valid_ds, test_ds = random_split(dataset, [0.8, 0.1, 0.1])  # type: CelebImages

    # dataloader
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=16)
    valid_dl = DataLoader(valid_ds, shuffle=False, batch_size=32)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=32)

    # model
    model = SiameseNN()
    # train
    trainer = TrainerManager(model, '', train_dl, valid_dl, test_dl, num_epochs=3)
    trainer.run()
    trainer.test()
