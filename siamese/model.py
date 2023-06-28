import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


class ScaledDotAttnModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.scaled_dot_attn = nn.functional.scaled_dot_product_attention

    def forward(self, query, key, value, dropout_p=0.2):
        return self.scaled_dot_attn(query, key, value, dropout_p=dropout_p)


class SiameseNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.resnet_org = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.fc_in_features = self.resnet_org.fc.in_features

        # remove the last layer of resnet (linear layer which is before last layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet_org.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2),
        )
        self.scaled_dot_attn = ScaledDotAttnModule()

        self.softmax = nn.Softmax(dim=1)

        # self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.05)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)

        return output

    def forward_frame_once(self, x):
        output = self.resnet2(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = (output1 - output2).pow(2)

        attention_output = self.scaled_dot_attn(output1, output2, output2)

        combined_output = torch.cat((attention_output, output), dim=1)

        # pass the difference to the linear layers
        output = self.fc(combined_output)
        output = self.softmax(output)

        return output
