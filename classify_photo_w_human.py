from glob import glob

from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification
import cv2


import torch

checkpoints_path = "/home/mudro/PycharmProjects/siamese/saved_model/cls_v7/model_best"
images = glob("/home/mudro/PycharmProjects/siamese/test/*")
image_processor = AutoImageProcessor.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    local_files_only=False
)
model = AutoModelForImageClassification.from_pretrained(
    checkpoints_path,
    local_files_only=True
)

with torch.no_grad():
    for im_path in images:
        img = cv2.imread(im_path)
        inputs = image_processor(img, return_tensors="pt")
        logits = model(**inputs).logits
        predicted_label = logits.argmax(-1).item()
        print(f"predicted_label {predicted_label}")
        label = model.config.id2label[predicted_label]

        print(im_path)
        print(f"logits {logits}")
        print(f"label {label}\n\n")
