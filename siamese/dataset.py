from typing import List, Tuple
from glob import glob
from enum import Enum
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from siamese.preprocess.image_helper import load_image

LABEL_IMG = "user"


# aka class for classification
class Category(Enum):
    SIMILAR = 0
    DIFFERENT = 1


@dataclass(slots=True)
class ImageItem:
    label_category: Category
    label_img: str
    target_img: str


SIMILAR_CATEGORY_FILE_ENDINGS = ['_true', '_t']
DIFF_CATEGORY_FILE_ENDINGS = ['_false', '_n']


class CelebImages(Dataset):
    def __init__(self, root: str = "/home/mudro/Documents/Projects/siamese/data/train/"):
        super().__init__()
        self.root = root
        self._data_paths: List[ImageItem] = self.init_data_paths()

    def init_data_paths(self) -> List[ImageItem]:
        results: List[ImageItem] = []
        if not self.root.endswith('/'):
            self.root = f"{self.root}/"
        for folder in glob(f'{self.root}*'):
            # get all files
            files_path = glob(f'{folder}/*')
            label_path_ls = list(filter(lambda item: LABEL_IMG in item, files_path))
            if not label_path_ls:
                raise Exception(f"Cannot find {LABEL_IMG} for folder {folder}")
            label_path = label_path_ls.pop()
            # there might be more than one target in the folder
            category = self._get_category(folder)
            for file_path in files_path:
                results.append(
                    ImageItem(
                        label_category=category,
                        label_img=label_path,
                        target_img=file_path
                    )
                )
        return results

    @staticmethod
    def _get_category(folder_path: str):
        ending = folder_path.split('/')[-1]
        if any([end in ending for end in SIMILAR_CATEGORY_FILE_ENDINGS]):
            return Category.SIMILAR
        elif any([end in ending for end in DIFF_CATEGORY_FILE_ENDINGS]):
            return Category.DIFFERENT
        else:
            raise Exception(f'Cannot detect the category for folder {folder_path}')

    def __len__(self):
        return len(self._data_paths)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, Category]:
        """
        For every example, we will select two images: label and target, and label_category aka class
        """
        item_path = self._data_paths[index]
        label_img = torch.tensor(load_image(item_path.label_img))
        target_img = torch.tensor(load_image(item_path.target_img))
        category = item_path.label_category

        return label_img, target_img, category
