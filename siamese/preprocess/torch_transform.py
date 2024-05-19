from typing import Tuple, TYPE_CHECKING

from torchvision import transforms

if TYPE_CHECKING:
    from torch import TensorType

MEAN_COEF = 0.5
STD_COEF = 0.5


class TransformCLSTrainHelper:
    def __init__(self):
        self.transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=(1, 25)),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

    def transform(self, img) -> 'TensorType':
        _img = self.transformation(img)
        return _img


class TransformTrainHelper:
    def __init__(self, size=(350, 350)):
        self.transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size, antialias=True),
            transforms.RandomRotation(degrees=(0, 25)),
            transforms.RandomAdjustSharpness(sharpness_factor=0, p=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.3),
            transforms.Normalize(mean=[MEAN_COEF, MEAN_COEF, MEAN_COEF],
                                 std=[STD_COEF, STD_COEF, STD_COEF])
        ])

    def transform(self, img) -> 'TensorType':
        _img = self.transformation(img)
        return _img

    def transform_2_imgs(self, label_img, target_1_img) -> Tuple['TensorType', 'TensorType']:
        label_img = self.transformation(label_img)
        target_1_img = self.transformation(target_1_img)

        return label_img, target_1_img


class TransformInferHelper:
    def __init__(self):
        self.transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((100, 100), antialias=True),
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
