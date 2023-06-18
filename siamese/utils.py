from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import TensorType, FloatTensor


CATEGORY_THRESHOLD = 0.21


def calc_euclidean(x1: 'TensorType', x2: 'TensorType'):
    return (x1 - x2).pow(2).sum(1)


def get_category(logit1: 'TensorType', logit2: 'TensorType') -> 'FloatTensor':
    diff = calc_euclidean(logit1, logit2) + 0.1
    diff = diff / 100
    print("diff ", diff)
    category = (diff > CATEGORY_THRESHOLD).float()
    print(f'Category is {category}')
    return category
