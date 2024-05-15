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
from siamese.custom_types import Category, HasFace
from siamese.model import SiameseNN
from siamese.trainer_mng import ModelTrainingWrapper


THRESHOLD = 0.37
# 0.45 TP 609, TN 158, FP 46, FN 39
# 0.3 TP 607, TN 162, FP 48, FN 35

transformation = torch_transform.TransformInferHelper()
# load pure Pytorch model
model_checkpoint = ModelTrainingWrapper.load_from_checkpoint(checkpoint_path=f'{os.getenv("SAVE_MODEL_PATH")}epoch=41-step=210.ckpt')
siamese = SiameseNN()
siamese.load_state_dict(model_checkpoint.backbone.state_dict())
TEST = False


def use_model(imgs: 'torch.Tensor', other_imgs: 'torch.Tensor') -> torch.FloatTensor:
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

    predicted = use_model(label_img, target_img)

    result = Category.SIMILAR.value if predicted.item() <= THRESHOLD else Category.DIFFERENT.value
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

        image_label_resized, img = image_helper.scale_images(image_with_label_copy, img)
        image_label_transformed, img = transformation.transform_2_imgs(image_label_resized, img)
        images_labels.append(image_label_transformed)
        images_targets.append(img)

    label_imgs = torch.stack(images_labels, dim=0)
    target_imgs = torch.stack(images_targets, dim=0)

    predicted = use_model(label_imgs, target_imgs)

    return predicted, [file for file in files_path if '/user' not in file], image_with_label_path


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

        predicted_tensor, images_pathes, label_image_path = predict_batch(path_dir)
        predicted = predicted_tensor

        # if predicted[0].shape[0] > 1:

        has_face_1 = torch.where(predicted[:, 0] < THRESHOLD, HasFace.HAS_FACE.value, HasFace.IS_OTHER.value)
        print(f"predicted_res has_face_1 {has_face_1}")
        has_face_1_list = has_face_1.tolist()
        has_face_2 = torch.where(predicted[:, 1] < THRESHOLD, HasFace.HAS_FACE.value, HasFace.IS_OTHER.value)
        print(f"predicted_res has_face_2 {has_face_2}")
        has_face_2_list = has_face_2.tolist()
        is_similar = torch.where(predicted[:, 2] > THRESHOLD, Category.DIFFERENT.value, Category.SIMILAR.value)
        print(f"predicted_res is_similar {is_similar}")

        is_similar_list = is_similar.tolist()
        zeros = is_similar_list.count([0])
        ones = is_similar_list.count([1])
        is_similar_label = 0 if zeros > ones else 1

        for img_path, has_face in zip(images_pathes, has_face_2_list):
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

            saved_results.append(
                {
                    "folder": path_dir,
                    "label_category": is_similar_label,
                    "label_img": label_image_path,
                    "label_img_has_face": has_face_1_list[0],
                    "target_img": img_path,
                    "target_img_has_face": has_face
                }
            )
    # if TEST:
    #     print(len(accuracy))
    #     print(accuracy.count(True))
    #     print(f'TP {tp}, TN {tn}, FP {fp}, FN {fn}')

    labels_path = os.getenv('LABELS_PATH')
    with open(labels_path, "w+") as json_file:
        json.dump(saved_results, json_file, indent=4)
