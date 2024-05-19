import os
import sys
from typing import List
from glob import glob
import json
from dataclasses import asdict

sys.path.append('./')
from siamese.custom_types import Category, ImageItem


LABEL_IMG = "user"
OTHER_CATEGORY_ENDING = 'other'
SIMILAR_CATEGORY_FILE_ENDINGS = ['_true', '_t']
DIFF_CATEGORY_FILE_ENDINGS = ['_false', '_n']


class DatasetJSONCreator:
    def __init__(self, root: str):
        super().__init__()
        self.root = root
        self.data_paths: List[ImageItem] = self.init_data_paths()

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
            label_index = files_path.index(label_path_ls.pop())
            label_path = files_path.pop(label_index)
            # there might be more than one target in the folder
            category = self._get_category(folder)

            for file_path in files_path:
                results.append(
                    ImageItem(
                        label_similar=category,
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

    def save_to_json(self, json_path):
        with open(json_path, "w+") as json_file:
            json.dump([item.dict() for item in self.data_paths], json_file, indent=4)


if __name__ == "__main__":
    dataset_labeled = os.getenv('LABELED_DS_TRAIN')
    json_creator = DatasetJSONCreator(dataset_labeled)
    dataset_json = os.getenv('DATASET_PATH')
    json_creator.save_to_json(dataset_json)
