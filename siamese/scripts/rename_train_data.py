from typing import List, TypedDict
import json
import os


class LabelItem(TypedDict):
    folder: str
    label_category: int


def main(path_to_file):
    with open(path_to_file) as f:
        json_data: List[LabelItem] = json.load(f)

        for item in json_data:
            label = item['label_category']
            folder = item['folder']

            if label == 0:
                new_path_dir = f"{folder}_true"
            else:
                new_path_dir = f"{folder}_false"
            try:
                os.rename(folder, new_path_dir)
            except FileNotFoundError:
                # folder have been already renamed
                # FIXME: make iterator which removes already renamed examples
                pass


if __name__ == "__main__":
    labels_path = os.getenv('LABELS_PATH')
    main(labels_path)
