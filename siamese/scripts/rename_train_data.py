from typing import List, TypedDict
import json
import os


class LabelItem(TypedDict):
    folder: str
    is_same_img: int
    confidence: float
    label_img: str
    label_img_has_face: int
    target_img: str
    target_img_has_face: int


def main(path_to_file):
    with open(path_to_file) as f:
        json_data: List[LabelItem] = json.load(f)

        for item in json_data:
            label = item['is_same_img']
            folder = item['folder']

            if label == 0:
                new_path_dir = f"{folder}_true"
            else:
                new_path_dir = f"{folder}_false"
            try:
                os.rename(folder, new_path_dir)

            except FileNotFoundError as exc:
                # folder have been already renamed
                # FIXME: make iterator which removes already renamed examples
                print(exc)
                pass


if __name__ == "__main__":
    labels_path = os.getenv('DIRTY_LABELS_PATH')
    main(labels_path)
