from typing import Dict, List
import random
import os

import cv2
import numpy as np
import torch


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


def minimize_face_to_bbox(image, annotation, min_size=(50, 50)):
    bbox = annotation.get('bbox')
    if not bbox:
        return None

    x_min, y_min, x_max, y_max = bbox
    if min_size and (abs(x_min - x_max) < min_size[0] or abs(y_min - y_max) < min_size[1]):
        return None

    x_min = np.clip(x_min, 0, x_max - 1)
    y_min = np.clip(y_min, 0, y_max - 1)
    img = image[y_min: y_max, x_min: x_max, ]
    return img


def get_faces_annotations(model, img) -> List[Dict]:
    with torch.no_grad():
        model.eval()
        annotations = model.predict_jsons(img)
        return annotations


if __name__ == '__main__':
    from glob import glob
    from retinaface.pre_trained_models import get_model

    model = get_model("resnet50_2020-07-20", max_size=1024)

    path_dir = '/media/mudro/Disk120Linux/datasets/siamese/data/faces'
    for folder in glob(f'{path_dir}/*'):
        for path in glob(f'{folder}/*'):
            print(path)
            image = cv2.imread(path, cv2.COLOR_BGR2RGB)
            if image is not None:
                # Get image height and width
                height, width, _ = image.shape

                # Check if height or width is less than 50 pixels
                if height < 50 or width < 50:
                    print(f"Deleting {path} as it has dimensions less than 50 pixels.")
                    os.remove(path)

            else:
                print(f"Failed to read {path}. Skipping...")

            annotations = get_faces_annotations(model, image)
            print(annotations)
            faces = [minimize_face_to_bbox(image, annotation) for annotation in annotations]
            for idx, face in enumerate(faces):
                if not isinstance(face, np.ndarray) and not face:
                    continue

                name_prefix = ''.join((chr(random.randrange(97, 123)) for i in range(10)))
                cv2.imwrite(
                    f'/media/mudro/Disk120Linux/datasets/siamese/data/faces/{name_prefix}{path.split("/")[-1]}{idx}.png',
                    face
                )
