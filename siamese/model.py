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

        self.scaled_dot_attn = ScaledDotAttnModule()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(672, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 1)
        )

        self.fc_final = torch.nn.Linear(2, 1)

        self.fc_final.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        self.backbone.apply(self.init_weights)

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

        output = 1 - torch.pow(
            nn.functional.cosine_similarity(output1, output2),
            2
        )
        output = output.unsqueeze(dim=1)
        # print("===== after cosine output ", output.shape)
        # print("===== after cosine output ", output.shape)

        attention_output = self.scaled_dot_attn(output1, output2, output1)
        attention_output = nn.functional.normalize(attention_output, dim=1)
        attention_output = self.fc(attention_output)
        # print("===== attention_output ", attention_output)
        # print("===== attention_output shape ", attention_output.shape)
        output = attention_output + output
        # print("===== output ", output)
        # print("===== output shape ", output.shape)

        return output
