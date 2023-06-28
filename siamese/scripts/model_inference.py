"""
Use Siamese model
"""
import os
import sys
from typing import List, Tuple
from glob import glob

import torch

sys.path.append('./')
from siamese.preprocess import image_helper, torch_transform
from siamese.custom_types import Category
from siamese.model import SiameseNN


transformation = torch_transform.TransformHelper()
with torch.no_grad():
    siamese = SiameseNN()
    siamese.load_state_dict(
        torch.load(os.getenv("SAVE_MODEL_PATH"))
    )


def use_model(imgs: 'torch.Tensor', other_imgs: 'torch.Tensor') -> torch.FloatTensor:
    with torch.no_grad():
        activations = siamese(imgs, other_imgs)
        print(activations)

    return activations


def predict_one(img_target_path, img_label_path) -> int:
    img_label = image_helper.load_image(img_label_path)
    img_target = image_helper.load_image(img_target_path)
    label_img, target_img = image_helper.resize_2_images(img_label, img_target)

    label_img, target_img = transformation.transform_2_imgs(label_img, target_img)
    label_img = label_img.unsqueeze(dim=0)
    target_img = target_img.unsqueeze(dim=0)

    predicted = use_model(label_img, target_img)

    result = Category.SIMILAR.value if predicted.squeeze().item() <= 0.5 else Category.DIFFERENT.value
    print(result)
    return result


def predict_batch(path_to_img_folder: str) -> Tuple['torch.Tensor', List, str]:
    if not path_to_img_folder.endswith('/'):
        path_to_img_folder = f'{path_to_img_folder}/'
    images_labels = []  # user
    images_targets = []
    files_path = glob(f'{path_to_img_folder}*')
    image_with_label_path = [file for file in files_path if '/user' in file].pop()
    print(f"image_with_label_path ${image_with_label_path}")
    image_with_label = image_helper.load_image(image_with_label_path)

    for file_path in files_path:
        if file_path == image_with_label_path:
            continue
        img = image_helper.load_image(file_path)
        print(f"file_path ${file_path}")
        image_with_label_copy = image_with_label.copy()

        image_label_resized, img = image_helper.resize_2_images(image_with_label_copy, img)
        image_label_transformed, img = transformation.transform_2_imgs(image_label_resized, img)
        images_labels.append(image_label_transformed)
        images_targets.append(img)

    label_imgs = torch.stack(images_labels, dim=0)
    target_imgs = torch.stack(images_targets, dim=0)

    predicted = use_model(label_imgs, target_imgs)

    result = torch.where(predicted > 0.5, Category.DIFFERENT.value, Category.SIMILAR.value)
    result = result.squeeze()
    return result, [file for file in files_path if '/user' not in file], image_with_label_path


if __name__ == "__main__":
    # predict_one(
    #     '/media/mudro/0B8CDB8D01D869D6/VICTOR_MY_LOVE/datasets/siamese/data/yandex/1514_false/user',
    #     '/media/mudro/0B8CDB8D01D869D6/VICTOR_MY_LOVE/datasets/siamese/data/yandex/1514_false/google_img0'
    # )
    saved_results = []
    accuracy = []

    for path_dir in glob('/media/mudro/0B8CDB8D01D869D6/VICTOR_MY_LOVE/datasets/siamese/data/New_data/*'):
        predicted_tensor, images_pathes, label_image_path = predict_batch(path_dir)
        predicted_res = predicted_tensor.tolist()
        print(predicted_res)
        label_category = predicted_res
        if isinstance(predicted_res, list):
            zeros = predicted_res.count(0)
            ones = predicted_res.count(1)
            label_category = 0 if zeros > ones else 1
        for img_path in images_pathes:
            expected = 0 if path_dir.endswith("true") else 1
            accuracy.append(expected == label_category)
            saved_results.append(
                {
                    "folder": path_dir,
                    "label_category": label_category,
                    "label_img": label_image_path,
                    "target_img": img_path
                }
            )
    print(len(accuracy))
    print(accuracy.count(True))
    with open('/home/mudro/Documents/Projects/siamese/dirty_label.json', "w+") as json_file:
        json.dump(saved_results, json_file, indent=4)
