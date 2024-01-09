import os
import sys
from typing import List
from glob import glob
import json
from dataclasses import asdict

sys.path.append('./')
from siamese.custom_types import CLSImageItem, HasFace


CATEGORY_OTHERS = "others"
CATEGORY_WITH_HUM = 'photo_w_hum'


class DatasetCLSCreator:
    def __init__(self, root: str):
        super().__init__()
        self.root = root
        self.data_paths: List[CLSImageItem] = self.get_data_paths()

    def get_data_paths(self) -> List[CLSImageItem]:
        results: List[CLSImageItem] = []
        if not self.root.endswith('/'):
            self.root = f"{self.root}/"
        for folder in glob(f'{self.root}*'):
            # get all files
            files_path = glob(f'{folder}/*')

            has_hum = HasFace.HAS_HUM_FACE if folder.endswith(CATEGORY_WITH_HUM) else HasFace.IS_OTHER

            for file_path in files_path:

                results.append(
                    CLSImageItem(
                        label_category=has_hum,
                        label_img_path=file_path
                    )
                )
        return results

    def save_to_json(self, json_path):
        with open(json_path, "w+") as json_file:
            json.dump([item.dict() for item in self.data_paths], json_file, indent=4)


if __name__ == "__main__":
    dataset_cls_labeled = os.getenv('LABELED_CSL_DS_TRAIN')
    json_creator = DatasetCLSCreator(dataset_cls_labeled)
    dataset_cls_json = os.getenv('DATASET_CSL_PATH')
    json_creator.save_to_json(dataset_cls_json)
