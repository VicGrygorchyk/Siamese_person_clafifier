from glob import glob

from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification
import cv2


import torch

checkpoints_path = "/home/mudro/PycharmProjects/siamese/saved_model/cls_v10/model_best"
images_hum = glob("/home/mudro/PycharmProjects/siamese/test/HUMAN/*")
images_others = glob("/home/mudro/PycharmProjects/siamese/test/OTHER/*")

image_processor = AutoImageProcessor.from_pretrained(
    checkpoints_path,
    local_files_only=False
)
model = AutoModelForImageClassification.from_pretrained(
    checkpoints_path,
    local_files_only=True
)

with torch.no_grad():
    hum_tp = 0
    hum_fp = 0
    other_tp = 0
    other_fp = 0

    for im_path in images_others:
        img = cv2.imread(im_path)
        inputs = image_processor(img, return_tensors="pt")
        logits = model(**inputs).logits
        predicted_label = logits.argmax(-1).item()
        if predicted_label == 0:
            other_fp += 1
        else:
            other_tp += 1
        print(f"predicted_label {predicted_label}")
        label = model.config.id2label[predicted_label]

        print(im_path)
        print(f"logits {logits}")
        print(f"label {label}\n\n")

    for im_path in images_hum:
        img = cv2.imread(im_path)
        inputs = image_processor(img, return_tensors="pt")
        logits = model(**inputs).logits
        predicted_label = logits.argmax(-1).item()
        if predicted_label == 0:
            hum_tp += 1
        else:
            hum_fp += 1
        print(f"predicted_label {predicted_label}")
        label = model.config.id2label[predicted_label]

        print(im_path)
        print(f"logits {logits}")
        print(f"label {label}\n\n")

    print(f"Humans face TP {hum_tp}, FP {hum_fp}, Others TP {other_tp}, FP {other_fp}")
