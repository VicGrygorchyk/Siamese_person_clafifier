from typing import List, Tuple, TYPE_CHECKING
import json

from torch import tensor
from torch.utils.data import Dataset
from torch.nn import functional as F

from siamese.preprocess import torch_transform, image_helper
from siamese.custom_types import ImageItem

if TYPE_CHECKING:
    from torch import TensorType

LABEL_IMG = "user"
SIMILAR_CATEGORY_FILE_ENDINGS = ['_true', '_t']
DIFF_CATEGORY_FILE_ENDINGS = ['_false', '_n']


class PersonsImages(Dataset):
    def __init__(self, json_file: str):
        super().__init__()
        self.root = json_file
        with open(self.root, 'r+') as json_file:
            self._data_paths: List[ImageItem] = [ImageItem(**item) for item in json.load(json_file)]
        # transform
        self.transformation = torch_transform.TransformHelper()

    def __len__(self):
        return len(self._data_paths)

    def __getitem__(self, index) -> Tuple['TensorType', 'TensorType', int, 'TensorType']:
        """
        For every example, we will select two images: label and target, and label_category aka class
        """
        item_path = self._data_paths[index]
        label_img = image_helper.load_image(item_path.label_img)
        category = item_path.label_category
        labels_onehot = F.one_hot(tensor(category), 2)

        target_img = image_helper.load_image(item_path.target_img)
        label_img, target_img = image_helper.resize_2_images(label_img, target_img)

        label_img, target_img = self.transformation.transform_2_imgs(label_img, target_img)

        return label_img, target_img, category, labels_onehot
