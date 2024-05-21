import torch
import torch.nn as nn
from torchvision.models import regnet_x_800mf, regnet_y_400mf, RegNet_X_800MF_Weights, RegNet_Y_400MF_Weights


class ScaledDotAttnModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.scaled_dot_attn = nn.functional.scaled_dot_product_attention

    def forward(self, query, key, value):
        return self.scaled_dot_attn(query, key, value)


class SiameseNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.net_org_x = regnet_y_400mf(weights=RegNet_Y_400MF_Weights)

        # remove the last layer of backbone (linear layer which is before last layer)
        backbone_layers1 = list(self.net_org_x.children())
        self.backbone_x = torch.nn.Sequential(*backbone_layers1[:-1])

        self.scaled_dot_attn1 = ScaledDotAttnModule()
        self.scaled_dot_attn2 = ScaledDotAttnModule()
        self.scaled_dot_attn3 = ScaledDotAttnModule()
        self.scaled_dot_attn4 = ScaledDotAttnModule()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(440, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 1)
        )
        self.fc.apply(self.init_weights)
        self.backbone_x.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.backbone_x(x)
        output = output.view(output.size()[0], -1)
        output = nn.functional.normalize(output, dim=1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # if it is similar
        output = torch.pow(
            (1 - nn.functional.cosine_similarity(output1, output2)),
            2
        )
        output = output.unsqueeze(dim=1)

        # attention
        attention_output = self.scaled_dot_attn1(output1, output2, output1)
        attention_output = self.scaled_dot_attn2(attention_output, output2, attention_output)
        attention_output = self.scaled_dot_attn3(attention_output, output2, attention_output)
        attention_output = self.scaled_dot_attn4(attention_output, output2, attention_output)

        # print(f"attention_output {attention_output}")
        # print(f"attention_output {attention_output.shape}")
        attention_output = nn.functional.normalize(attention_output, dim=1)
        # print("===== attention_output ", attention_output)
        # print("===== attention_output shape ", attention_output.shape)

        final_output = attention_output + output
        final_output = self.fc(final_output)

        # print("===== final_output ", final_output)
        # print("===== final_output shape ", final_output.shape)
        final_output = torch.sigmoid(final_output)
        return final_output
