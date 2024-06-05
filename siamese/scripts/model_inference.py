"""
Use Siamese model
"""
import os
import sys
from typing import List, Tuple
from glob import glob
import json

import torch

sys.path.append('./')
from siamese.preprocess import image_helper, torch_transform
from siamese.custom_types import Category
from siamese.model3 import SiameseNN
from siamese.trainer_mng import ModelTrainingWrapper


THRESHOLD = 0.5
MIN = 0.35
MAX = 0.65

transformation = torch_transform.TransformInferHelper()
# load pure Pytorch model
model_checkpoint = ModelTrainingWrapper.load_from_checkpoint(
    checkpoint_path=f'{os.getenv("SAVE_MODEL_PATH")}{os.getenv("BEST_MODEL_CHECKPOINT")}'
)
siamese = SiameseNN()
siamese.load_state_dict(model_checkpoint.backbone.state_dict())
TEST = False


def use_model(imgs: 'torch.Tensor', other_imgs: 'torch.Tensor', diff=None) -> torch.FloatTensor:
    siamese.eval()
    with torch.no_grad():
        activations = siamese(imgs, other_imgs)
        print(activations)

    return activations


def predict_one(img_target_path, img_label_path) -> int:
    img_label = image_helper.load_image(img_label_path)
    img_target = image_helper.load_image(img_target_path)
    label_img, target_img = image_helper.scale_images(img_label, img_target)

    label_img, target_img = transformation.transform_2_imgs(label_img, target_img)
    label_img = label_img.unsqueeze(dim=0)
    target_img = target_img.unsqueeze(dim=0)

    diff = image_helper.get_image_difference(label_img, target_img)
    predicted = use_model(label_img, target_img, diff)

    result = Category.SIMILAR.value if predicted.item() <= THRESHOLD else Category.DIFFERENT.value
    print(result)
    return result


def predict_batch(path_to_img_folder: str) -> Tuple['torch.Tensor', List, List, str]:
    if not path_to_img_folder.endswith('/'):
        path_to_img_folder = f'{path_to_img_folder}/'
    images_labels = []  # user
    images_targets = []
    img_diff = []
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

        image_with_label_copy = image_helper.make_square(image_with_label_copy)
        img = image_helper.make_square(img)
        diff = image_helper.get_image_difference(image_with_label_copy, img)
        img_diff.append(diff)

        image_label_transformed, img = transformation.transform_2_imgs(image_with_label_copy, img)
        images_labels.append(image_label_transformed)
        images_targets.append(img)

    label_imgs = torch.stack(images_labels, dim=0)
    target_imgs = torch.stack(images_targets, dim=0)
    diff_stack = torch.tensor(img_diff)

    predicted = use_model(label_imgs, target_imgs, diff_stack)

    return predicted, [file for file in files_path if '/user' not in file], img_diff, image_with_label_path


if __name__ == "__main__":
    # predict_one(
    #     '/media/mudro/0B8CDB8D01D869D6/VICTOR_MY_LOVE/datasets/siamese/data/yandex/1514_false/user',
    #     '/media/mudro/0B8CDB8D01D869D6/VICTOR_MY_LOVE/datasets/siamese/data/yandex/1514_false/google_img0'
    # )
    saved_results = []
    if TEST:
        accuracy = []
        tp = 0
        tn = 0
        fp = 0
        fn = 0
    dataset_path = os.getenv('DATA_FOR_INFER')

    for path_dir in glob(f'{dataset_path}/*'):
        if x := len(glob(f'{path_dir}/*')) <= 1:
            print("========= ", x)
            continue

        predicted_tensor, images_pathes, img_diffs, label_image_path = predict_batch(path_dir)
        predicted = predicted_tensor

        should_skipped = torch.where((predicted > MIN) & (predicted < MAX), 3, -1)
        print(f"predicted_res should be skipped {should_skipped}")

        is_similar = torch.where(predicted > THRESHOLD, Category.DIFFERENT.value, Category.SIMILAR.value)
        print(f"predicted_res is_similar {is_similar}")

        should_skipped_list = should_skipped.tolist()
        is_similar_list = is_similar.tolist()
        predicted_list = predicted.tolist()

        for idx, img_path in enumerate(images_pathes):
            if TEST:
                expected = 0 if path_dir.endswith("true") else 1
                # accuracy.append(expected == label_category)
                # if expected == label_category:
                #     if expected == 0:
                #         tp += 1
                #     else:
                #         tn += 1
                # else:
                #     if expected == 0:
                #         fp += 1
                #     else:
                #         fn += 1
            should_skipped_list_item = should_skipped_list[idx][0]
            is_similar_list_item = is_similar_list[idx][0]
            predicted_list_item = predicted_list[idx][0]
            saved_results.append(
                {
                    "folder": path_dir,
                    "is_same_img": is_similar_list_item,
                    "confidence": predicted_list_item,
                    "skip_because_confidence": should_skipped_list_item,
                    "label_img": label_image_path,
                    "target_img": img_path,
                    "diff": img_diffs[idx],
                }
            )
    # if TEST:
    #     print(len(accuracy))
    #     print(accuracy.count(True))
    #     print(f'TP {tp}, TN {tn}, FP {fp}, FN {fn}')

    labels_path = os.getenv('DIRTY_LABELS_PATH')
    with open(labels_path, "w+") as json_file:
        json.dump(saved_results, json_file, indent=4)
