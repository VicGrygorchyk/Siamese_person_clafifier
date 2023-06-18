from typing import List, Tuple, TYPE_CHECKING
import json
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

    def __getitem__(self, index) -> Tuple['TensorType', 'TensorType', 'TensorType', int]:
        """
        For every example, we will select two images: label and target, and label_category aka class
        """
        item_path = self._data_paths[index]
        # print(item_path)
        label_img = image_helper.load_image(item_path.label_img)
        category = item_path.label_category

        if category == Category.SIMILAR:
            same_img = image_helper.load_image(item_path.target_img)
            # get from another directory
            idx = index + 1
            if idx >= len(self._data_paths):
                idx = 0
            diff_item_path = self._data_paths[idx]
            diff_img = image_helper.load_image(diff_item_path.target_img)
        else:
            # get users image, but crop black borders and make mirror img
            same_img = image_helper.load_image(item_path.label_img)
            same_img = image_helper.flip_img(same_img)
            # same_img = image_helper.crop_black_border(same_img)
            diff_img = image_helper.load_image(item_path.target_img)

        label_img, same_img, diff_img = image_helper.resize_3_images(label_img, same_img, diff_img)
        label_img, same_img, diff_img = self.transformation.transform(label_img, same_img, diff_img)

        return label_img, same_img, diff_img, category
