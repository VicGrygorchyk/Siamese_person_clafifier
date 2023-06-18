from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import TensorType, FloatTensor


CATEGORY_THRESHOLD = 0.21


def calc_euclidean(x1: 'TensorType', x2: 'TensorType'):
    return (x1 - x2).pow(2).sum(1)


def get_category(activations: 'TensorType') -> 'FloatTensor':
    print(f'Activation sum {activations.sum(1)}')
    category = (activations.sum(1) > CATEGORY_THRESHOLD).float()
    print(f'Category is {category}')
    return category
