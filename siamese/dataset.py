from typing import List, Tuple, TYPE_CHECKING
from glob import glob

from torch.utils.data import Dataset
from torchvision import transforms

from preprocess import image_helper
from custom_types import Category, ImageItem

if TYPE_CHECKING:
    from torch import TensorType

LABEL_IMG = "user"
SIMILAR_CATEGORY_FILE_ENDINGS = ['_true', '_t']
DIFF_CATEGORY_FILE_ENDINGS = ['_false', '_n']
NORMALIZE_COEF = 0.5


class CelebImages(Dataset):
    def __init__(self, root: str = "/home/mudro/Documents/Projects/siamese/data/train/"):
        super().__init__()
        self.root = root
        self._data_paths: List[ImageItem] = self.init_data_paths()
        # transform
        self.transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((450, 450)),
            transforms.Normalize(mean=[NORMALIZE_COEF, NORMALIZE_COEF, NORMALIZE_COEF],
                                 std=[NORMALIZE_COEF, NORMALIZE_COEF, NORMALIZE_COEF])
        ])

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

    def __getitem__(self, index) -> Tuple[TensorType, TensorType, int]:
        """
        For every example, we will select two images: label and target, and label_category aka class
        """
        item_path = self._data_paths[index]
        # print(item_path)
        label_img = image_helper.load_image(item_path.label_img)
        target_img = image_helper.load_image(item_path.target_img)
        label_img, target_img = image_helper.resize_images(label_img, target_img)

        category = item_path.label_category
        label_img = self.transformation(label_img)
        target_img = self.transformation(target_img)

        return label_img, target_img, category.value
