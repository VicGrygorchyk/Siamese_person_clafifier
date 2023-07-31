from typing import Tuple, TYPE_CHECKING

from torchvision import transforms

if TYPE_CHECKING:
    from torch import TensorType

MEAN_COEF = 0.85
STD_COEF = 0.5


class TransformHelper:
    def __init__(self):
        self.transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((400, 400), antialias=True),
            transforms.Normalize(mean=[MEAN_COEF, MEAN_COEF, MEAN_COEF],
                                 std=[STD_COEF, STD_COEF, STD_COEF])
        ])

    def transform(self, label_img, target_1_img, target_2_img) -> Tuple['Tensor', 'Tensor', 'Tensor']:
        label_img = self.transformation(label_img)
        target_1_img = self.transformation(target_1_img)
        target_2_img = self.transformation(target_2_img)

        return label_img, target_1_img, target_2_img

    def transform_2_imgs(self, label_img, target_1_img) -> Tuple['TensorType', 'TensorType']:
        label_img = self.transformation(label_img)
        target_1_img = self.transformation(target_1_img)

        return label_img, target_1_img
