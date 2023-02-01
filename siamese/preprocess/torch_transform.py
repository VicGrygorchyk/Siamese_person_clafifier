from typing import Tuple

from torchvision import transforms

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
        label_img = self.transformation(label_img)
        target_img = self.transformation(target_img)
        return label_img, target_img
