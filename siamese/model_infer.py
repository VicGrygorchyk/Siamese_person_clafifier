"""
Use Siamese model
"""
from torch import load as torch_load

from model import SiameseNN


def use_model(saved_model_path):
    model = SiameseNN()
    model.load_state_dict(torch_load(saved_model_path))
