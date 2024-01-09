from glob import glob

from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification
import cv2


import torch

checkpoints_path = "/home/mudro/Documents/Projects/siamese/saved_model/cls_v3/model"
images = glob("/home/mudro/Documents/Projects/siamese/test/*")
image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k', local_files_only=True)
model = AutoModelForImageClassification.from_pretrained(checkpoints_path, local_files_only=True)

with torch.no_grad():
    for im_path in images:
        img = cv2.imread(im_path)
        inputs = image_processor(img, return_tensors="pt")
        logits = model(**inputs).logits
        predicted_label = logits.argmax(-1).item()
        print(predicted_label)
        label = model.config.id2label[predicted_label]
        if label == 'HAS_HUM_FACE':
            print(im_path)
            print(logits)
            print(label)
