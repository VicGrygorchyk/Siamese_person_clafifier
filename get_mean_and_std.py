import os

import torch
from siamese.dataset import PersonsImages

EPOCH = 50


DATASET_PATH = os.getenv("DATASET_PATH")


if __name__ == "__main__":
    # dataset
    dataset = PersonsImages(DATASET_PATH)

    means1 = torch.tensor([0], dtype=torch.float)
    std1 = torch.tensor([0], dtype=torch.float)
    means2 = torch.tensor([0], dtype=torch.float)
    std2 = torch.tensor([0], dtype=torch.float)
    means3 = torch.tensor([0], dtype=torch.float)
    std3 = torch.tensor([0], dtype=torch.float)

    for img_label, img_target, _, in dataset:
        img_label = img_label.unsqueeze(dim=1)
        img_target = img_target.unsqueeze(dim=1)
        means1 += img_label[0].mean()
        means1 += img_target[0].mean()
        std1 += img_label[0].std()
        std1 += img_target[0].std()

        means2 += img_label[1].mean()
        means2 += img_target[1].mean()
        std2 += img_label[1].std()
        std2 += img_target[1].std()

        means3 += img_label[2].mean()
        means3 += img_target[2].mean()
        std3 += img_label[2].std()
        std3 += img_target[2].std()

    print(means1 / len(dataset))
    print(std1 / len(dataset))
    print(means2 / len(dataset))
    print(std2 / len(dataset))
    print(means3 / len(dataset))
    print(std3 / len(dataset))
