import os
import json
import cv2

from transformers import AutoImageProcessor


path = os.getenv('DATASET_CSL_PATH')
checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)


with open(path) as file:
    cls_file = json.load(file)

    for item in cls_file:
        img_path = item['label_img_path']
        img = cv2.imread(img_path)
        # img_shape = img.shape
        #
        # if img_shape[2] < 3:
        #     print(img_shape)
        #     print(img_path)

        try:
            image_processor(img)
        except Exception as exc:
            print(exc)
            print(img_path)
