from typing import List

import cv2
from retinaface.pre_trained_models import get_model
import numpy as np
import torch

model = get_model("resnet50_2020-07-20", max_size=1024)


def add_bbox(image, annotation):
    vis_image = image.copy()
    bbox = annotation.get('bbox')

    if not bbox:
        return None

    x_min, y_min, x_max, y_max = bbox
    x_min = np.clip(x_min, 0, x_max - 1)
    y_min = np.clip(y_min, 0, y_max - 1)

    vis_image = cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

    return vis_image


def minimize_face_to_bbox(image, annotation):
    bbox = annotation.get('bbox')
    if not bbox:
        return None

    x_min, y_min, x_max, y_max = bbox

    x_min = np.clip(x_min, 0, x_max - 1)
    y_min = np.clip(y_min, 0, y_max - 1)
    img = image[y_min: y_max, x_min: x_max, ]
    return img


def get_faces(img_path) -> List[List]:
    image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        model.eval()
        annotations = model.predict_jsons(image)
        print(annotations)
        return [add_bbox(image, a) for a in annotations]


if __name__ == '__main__':
    from glob import glob

    path_dir = '/home/mudro/Documents/Projects/siamese/test'
    for path in glob(f'{path_dir}/*'):
        faces = get_faces(path)
        for idx, face in enumerate(faces):
            if not isinstance(face, np.ndarray) and not face:
                print('no face')
                continue

            cv2.imwrite(f'{path.split("/")[-1]}{idx}.png', face)
