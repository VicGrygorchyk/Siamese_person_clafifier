from enum import Enum
from dataclasses import dataclass


# aka class for classification
class Category(Enum):
    SIMILAR = 0
    DIFFERENT = 1


@dataclass(slots=True)
class ImageItem:
    label_category: Category
    label_img: str
    target_img: str
