from typing import List, Tuple, TYPE_CHECKING
import json
from random import random

from torch.utils.data import Dataset

from preprocess import torch_transform, image_helper
from custom_types import ImageItem, Category

if TYPE_CHECKING:
    from torch import TensorType

LABEL_IMG = "user"
SIMILAR_CATEGORY_FILE_ENDINGS = ['_true', '_t']
DIFF_CATEGORY_FILE_ENDINGS = ['_false', '_n']


class CelebImages(Dataset):
    def __init__(self, json_file: str = "/home/mudro/Documents/Projects/siamese/data/train.json"):
        super().__init__()
        self.root = json_file
        with open(self.root, 'r+') as json_file:
            self._data_paths: List[ImageItem] = [ImageItem(**item) for item in json.load(json_file)]
        # transform
        self.transformation = torch_transform.TransformHelper()

    def __len__(self):
        return len(self._data_paths)

    def __getitem__(self, index) -> Tuple['TensorType', 'TensorType', int]:
        """
        For every example, we will select two images: label and target, and label_category aka class
        """
        item_path = self._data_paths[index]
        # print(item_path)
        label_img = image_helper.load_image(item_path.label_img)
        category = item_path.label_category
        r = random()

        if category == Category.SIMILAR:
            target_img = image_helper.load_image(item_path.target_img)
            if 0.6 < r < 0.8:
                target_img = image_helper.flip_img(target_img)
            if 0.8 < r:
                target_img = image_helper.rotate(target_img, 10)
        else:
            target_img = image_helper.load_image(item_path.target_img)

        label_img, target_img = image_helper.resize_2_images(label_img, target_img)
        label_img, target_img = self.transformation.transform_2_imgs(label_img, target_img)

        return label_img, target_img, category
