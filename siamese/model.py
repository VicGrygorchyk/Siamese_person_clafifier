import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class SiameseNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.fc_in_features = self.resnet.fc.in_features

        # remove the last layer of resnet18 (linear layer which is before last layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    @staticmethod
    def cosine_distance(tensor1, tensor2):
        """
        Calculates the cosine distance between two tensors.

        Args:
            tensor1 (torch.Tensor): The first input tensor.
            tensor2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The cosine distance between the two tensors.
        """
        # Normalize the input tensors
        tensor1_normalized = F.normalize(tensor1, dim=-1)
        tensor2_normalized = F.normalize(tensor2, dim=-1)

        # Compute the dot product of the normalized tensors
        dot_product = torch.sum(tensor1_normalized * tensor2_normalized, dim=-1)

        # Calculate the cosine distance
        cosine_distance = 1.0 - dot_product

        return cosine_distance

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        output = self.cosine_distance(output1, output2).pow(2)

        # pass the difference to the linear layers
        output = self.fc(output)

        return output
