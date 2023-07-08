import torch
import torch.nn as nn
from torchvision.models import regnet_x_800mf, RegNet_X_800MF_Weights


class ScaledDotAttnModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.scaled_dot_attn = nn.functional.scaled_dot_product_attention

    def forward(self, query, key, value):
        return self.scaled_dot_attn(query, key, value)


class SiameseNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.net_org = regnet_x_800mf(weights=RegNet_X_800MF_Weights.DEFAULT)

        # remove the last layer of backbone (linear layer which is before last layer)
        backbone_layers = list(self.net_org.children())
        self.backbone = torch.nn.Sequential(*backbone_layers[:-1])

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(672, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self.scaled_dot_attn = ScaledDotAttnModule()

        self.sigmoid = nn.Sigmoid()

        self.backbone.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.backbone(x)
        output = output.view(output.size()[0], -1)
        output = nn.functional.normalize(output, dim=1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        attention_output = torch.pow(self.scaled_dot_attn(output1, output2, output1), 2)
        attention_output = nn.functional.normalize(attention_output, dim=1)
        # print("===== attention_output ", attention_output.shape)
        #
        # print("===== output1 ", output1.shape)
        # print("===== output2 ", output2.shape)

        output = 1 - torch.pow(
            nn.functional.cosine_similarity(output1, output2),
            2
        )
        # print("===== after cosine output ", output)
        # print("===== output shape ", output.shape)

        # output = nn.functional.normalize(output, dim=0)
        # print("output shape", output.shape)

        # combined_output = torch.cat((attention_output, output), dim=1)

        # output = self.fc(output)

        # output = self.sigmoid(output)
        # print("===after sigmoid output ", output)

        return output
