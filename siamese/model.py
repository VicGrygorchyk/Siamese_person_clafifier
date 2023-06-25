import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

from preprocess.image_helper import get_frames


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)  # 1x1 convolution to reduce channels
        self.softmax = nn.Softmax(dim=2)  # softmax along spatial dimensions

    def forward(self, x):
        attention = self.conv(x)
        attention = self.softmax(attention)
        return attention


class ScaledDotAttnModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.scaled_dot_attn = nn.functional.scaled_dot_product_attention

    def forward(self, query, key, value, dropout_p=0.2):
        return self.scaled_dot_attn(query, key, value, dropout_p=dropout_p)


class FrameDetector(nn.Module):

    def __init__(self):
        super().__init__()
        # self.cnn = nn.Conv2d(3, 16, 3)
        # self.self_attn = SelfAttention(16)
        # self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        # self.fc = nn.Linear(256 * 256 * 49, 64)
        self.resnet = resnet34()

    @staticmethod
    def _get_image_borders(image):
        return get_frames(image)

    def forward(self, image1):
        output = self.resnet(image1)
        output = output.view(output.size()[0], -1)
        return output


class SiameseNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.fc_in_features = self.resnet.fc.in_features

        # remove the last layer of resnet18 (linear layer which is before last layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.frame_detector = FrameDetector()

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(3024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

        self.scaled_dot_attm = ScaledDotAttnModule()
        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        frame1 = self.frame_detector(input1)
        frame2 = self.frame_detector(input2)
        combined_output1 = torch.cat((output1, frame1), dim=1)
        combined_output2 = torch.cat((output2, frame2), dim=1)

        attention_output = self.scaled_dot_attm(combined_output1, combined_output2, combined_output2)
        output = (combined_output1 - combined_output1).pow(2)
        combined_output = torch.cat((attention_output, output), dim=1)

        # pass the difference to the linear layers
        output = self.fc(combined_output)
        output = self.sigmoid(output)
        return output
