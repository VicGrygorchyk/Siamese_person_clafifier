import os
import sys
from glob import glob
from random import random

sys.path.append('./')
from siamese.preprocess import image_helper


def transform_image(path_to_img_folder: str):
    if not path_to_img_folder.endswith('/'):
        path_to_img_folder = f'{path_to_img_folder}/'

    images = []

    files_path = glob(f'{path_to_img_folder}*')
    image_with_label_path = [file for file in files_path if '/user' in file].pop()
    print(f"image_with_label_path ${image_with_label_path}")

    for file_path in files_path:
        if file_path == image_with_label_path:
            continue
        img = image_helper.load_image(file_path)
        print(f"file_path ${file_path}")
        images.append((img, file_path))

    for img_, path_ in images:
        new_img_path = f'{path_to_img_folder}aug_{path_.split("/")[-1]}.jpg'
        print(new_img_path)
        r = random()
        if r < 0.4:
            img_ = image_helper.flip_img(img_)
        elif 0.4 < r < 0.6:
            img_ = image_helper.rotate(img_, 5)
        elif 0.6 < r < 0.8:
            img_ = image_helper.rotate(img_, 15)
        elif 0.8 < r:
            img_ = image_helper.rotate(img_, 10)
            img_ = image_helper.flip_img(img_)
        image_helper.save_img(new_img_path, img_)


if __name__ == "__main__":
    dataset_labeled = os.getenv('LABELED_DS_TRAIN')

    for path_dir in glob(f'{dataset_labeled}/*_false'):
        transform_image(path_dir)
