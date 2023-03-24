from torch.utils.data import DataLoader, random_split

from model import SiameseNN
from dataset import CelebImages
from trainer_mng import TrainerManager

EPOCH = 10


if __name__ == "__main__":
    # dataset
    dataset = CelebImages('/home/mudro/Documents/Projects/siamese/data/labels_data.json')
    train_ds, valid_ds, test_ds = random_split(dataset, [0.8, 0.15, 0.05])  # type: CelebImages

    # dataloader
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=18)
    valid_dl = DataLoader(valid_ds, shuffle=False, batch_size=24)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=24)

    # model
    model = SiameseNN()
    # train
    trainer = TrainerManager(
        model,
        '/home/mudro/Documents/Projects/siamese/saved_model/siamese3.pt',
        train_dl, valid_dl, test_dl, num_epochs=50
    )
    trainer.run()
    trainer.test()
