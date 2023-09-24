from typing import List
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel


# aka class for classification
class Category(Enum):
    SIMILAR = 0
    DIFFERENT = 1


class HasFace(Enum):
    HAS_FACE = 0
    IS_OTHER = 1


class Label(BaseModel):
    label_has_face_source: HasFace
    label_has_face_target: HasFace
    label_similar: Category

    class Config:
        use_enum_values = True


class ImageItem(BaseModel):
    label_category: Label
    label_img: str
    target_img: str

    class Config:
        use_enum_values = True
