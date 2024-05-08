from typing import List, Tuple, TYPE_CHECKING
import json

from torch import Tensor
from torch.utils.data import Dataset
from retinaface.pre_trained_models import get_model

from siamese.preprocess import torch_transform, image_helper, get_face
from siamese.custom_types import ImageItem, CLSImageItem

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
        self.transformation = torch_transform.TransformTrainHelper()
        self.face_detect_model = get_model("resnet50_2020-07-20", max_size=512)

    def __len__(self):
        return len(self._data_paths)

    def __getitem__(self, index) -> Tuple['TensorType', 'TensorType', 'Tensor']:
        """
        For every example, we will select two images: label and target, and label_category aka class
        """
        item_path = self._data_paths[index]
        label_img = image_helper.load_image(item_path.label_img)
        category = item_path.label_category

        label_similar = float(category.label_similar)

        target_img = image_helper.load_image(item_path.target_img)
        label_img, target_img = image_helper.scale_images(label_img, target_img)

        label_img, target_img = self.transformation.transform_2_imgs(label_img, target_img)

        return label_img, target_img, label_similar


class HasHumanImages(Dataset):
    def __init__(self, json_file: str, image_processor):
        super().__init__()
        self.root = json_file
        with open(self.root, 'r+') as json_file:
            self._data_paths: List[CLSImageItem] = [CLSImageItem(**item) for item in json.load(json_file)]
        # transform
        self.image_processor = image_processor
        self.transformation = torch_transform.TransformCLSTrainHelper()

    def __len__(self):
        return len(self._data_paths)

    def __getitem__(self, index):
        item_path = self._data_paths[index]
        category: int = item_path.label_category

        label = category
        img = image_helper.load_image(item_path.label_img_path)
        try:
            img = self.image_processor(img)
        except Exception as exc:
            print(f"Error {item_path.label_img_path}")
            raise exc

        img = img.data['pixel_values'][0]

        return img, label
