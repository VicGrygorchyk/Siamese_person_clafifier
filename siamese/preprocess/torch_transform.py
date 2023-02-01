from typing import Tuple

from torchvision import transforms

from siamese.preprocess import image_helper

NORMALIZE_COEF = 0.5


class TransformHelper:
    def __init__(self):
        self.transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((450, 450)),
            transforms.Normalize(mean=[NORMALIZE_COEF, NORMALIZE_COEF, NORMALIZE_COEF],
                                 std=[NORMALIZE_COEF, NORMALIZE_COEF, NORMALIZE_COEF])
        ])

    def transform(self, label_img, target_img) -> Tuple['Tensor', 'Tensor']:
        label_img, target_img = image_helper.resize_images(label_img, target_img)
        label_img = self.transformation(label_img)
        target_img = self.transformation(target_img)
        return label_img, target_img
