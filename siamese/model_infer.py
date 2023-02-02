"""
Use Siamese model
"""
from glob import glob

import torch
from preprocess import image_helper, torch_transform
from custom_types import Category

from model import SiameseNN


transformation = torch_transform.TransformHelper()


def use_model(saved_model_path: str, label_imgs: 'torch.Tensor', target_imgs: 'torch.Tensor') -> 'torch.Tensor':
    with torch.no_grad():
        siamese = SiameseNN()
        siamese.load_state_dict(torch.load(saved_model_path))
        predicted = siamese(label_imgs, target_imgs)
        print(predicted)

    return predicted


def predict_one(saved_model_path, img_target_path, img_label_path) -> int:
    img_label = image_helper.load_image(img_label_path)
    img_target = image_helper.load_image(img_target_path)
    label_img, target_img = image_helper.resize_images(img_label, img_target)

    label_img, target_img = transformation.transform(label_img, target_img)
    label_img = label_img.unsqueeze(dim=0)
    target_img = target_img.unsqueeze(dim=0)

    predicted = use_model(saved_model_path, label_img, target_img)

    result = Category.SIMILAR.value if predicted.squeeze().item() < 0.5 else Category.DIFFERENT.value
    print(result)
    return result


def predict_batch(saved_model_path: str, path_to_img_folder: str) -> 'torch.Tensor':
    if not path_to_img_folder.endswith('/'):
        path_to_img_folder = f'{path_to_img_folder}/'
    images_labels = []  # user
    images_targets = []
    files_path = glob(f'{path_to_img_folder}*')
    image_with_label_path = [file for file in files_path if '/user' in file].pop()
    image_with_label = image_helper.load_image(image_with_label_path)

    for file_path in files_path:
        if file_path == image_with_label_path:
            continue
        img = image_helper.load_image(file_path)
        # FIXME no need to resize and transform label img as it is the same for each iter, but how to resize?
        image_label, img = image_helper.resize_images(image_with_label, img)
        image_label, img = transformation.transform(image_label, img)
        images_targets.append(img)
        images_labels.append(image_label)

    label_imgs = torch.stack(images_labels, dim=0)
    target_imgs = torch.stack(images_targets, dim=0)

    predicted = use_model(saved_model_path, label_imgs, target_imgs)

    result = torch.where(predicted < 0.5, Category.SIMILAR.value, Category.DIFFERENT.value)
    result = result.squeeze()
    print(result)
    return result


if __name__ == "__main__":
    predict_one(
        '/home/mudro/Documents/Projects/siamese/saved_model/siamese.pt',
        '/home/mudro/Documents/Projects/siamese/data/train/1500/user',
        '/home/mudro/Documents/Projects/siamese/data/train/1467/google_img0'
    )
    # predict_batch(
    #     '/home/mudro/Documents/Projects/siamese/saved_model/siamese.pt',
    #     '/home/mudro/Documents/Projects/siamese/data/train/1465/'
    # )
