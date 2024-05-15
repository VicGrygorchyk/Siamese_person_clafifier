from typing import List, TypedDict
import json
import os


class LabelItem(TypedDict):
    folder: str
    label_category: int
    label_img: str
    label_img_has_face: int
    target_img: str
    target_img_has_face: int


def main(path_to_file):
    with open(path_to_file) as f:
        json_data: List[LabelItem] = json.load(f)

        for item in json_data:
            label = item['label_category']
            folder = item['folder']
            label_img_path = item['label_img']
            label_img_has_face = item['label_img_has_face']
            target_img_path = item['target_img']
            target_img_has_face = item['target_img_has_face']

            if label == 0:
                new_path_dir = f"{folder}"
            else:
                new_path_dir = f"{folder}"
            try:
                # os.rename(folder, new_path_dir)

                label_img_name = label_img_path.split('/')[-1]
                new_path_label = f"{new_path_dir}/{'other' if label_img_has_face == 1 else ''}{label_img_name}"
                os.rename(label_img_path, new_path_label)

                target_img_name = target_img_path.split('/')[-1]
                new_path_trgt = f"{new_path_dir}/{'other' if target_img_has_face == 1 else ''}{target_img_name}"
                os.rename(target_img_path, new_path_trgt)

            except FileNotFoundError as exc:
                # folder have been already renamed
                # FIXME: make iterator which removes already renamed examples
                print(exc)
                pass


if __name__ == "__main__":
    labels_path = os.getenv('LABELS_PATH')
    main(labels_path)
