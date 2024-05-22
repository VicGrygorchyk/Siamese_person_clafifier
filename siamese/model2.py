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
        self.net_org_x = regnet_x_800mf(weights=RegNet_X_800MF_Weights)

        # remove the last layer of backbone (linear layer which is before last layer)
        backbone_layers1 = list(self.net_org_x.children())
        self.backbone_x = torch.nn.Sequential(*backbone_layers1[:-1])

        self.scaled_dot_attn1 = ScaledDotAttnModule()
        self.scaled_dot_attn2 = ScaledDotAttnModule()
        self.scaled_dot_attn3 = ScaledDotAttnModule()
        self.scaled_dot_attn4 = ScaledDotAttnModule()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(672, 128),  # 440 for x_400
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 1)
        )

        self.fc_attn = torch.nn.Sequential(
            torch.nn.Linear(672, 128),  # 440 for x_400
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 1)
        )

        self.final_fc = torch.nn.Sequential(
            torch.nn.Linear(4, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
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

    def forward(self, input1, input2, diff):
        diff = diff.unsqueeze(dim=1)
        diff = diff.type_as(input1)

        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # if it is similar
        cos_similarity = torch.pow(
            (1 - nn.functional.cosine_similarity(output1, output2)),
            2
        )
        cos_similarity = cos_similarity.unsqueeze(dim=1)
        feat_difference = torch.pow((output1 - output2), 2)

        # attention
        attention_output = self.scaled_dot_attn1(output1, output2, output1)
        attention_output = self.scaled_dot_attn2(attention_output, output2, attention_output)
        attention_output = self.scaled_dot_attn3(attention_output, output2, attention_output)
        attention_output = self.scaled_dot_attn4(attention_output, output2, attention_output)
        fc_attn_output = self.fc_attn(attention_output)

        # attention_output = nn.functional.normalize(attention_output, dim=1)
        # print("===== attention_output ", attention_output)
        # print("===== attention_output shape ", attention_output.shape)
        # print("===== cos_similarity ", cos_similarity)
        # print("===== cos_similarity shape ", cos_similarity.shape)
        # print("===== feature diff ", feat_difference)
        # print("===== feature diff shape ", feat_difference.shape)
        # print("===== diff ", diff)
        # print("===== diff shape ", diff.shape)

        # att_and_diff = attention_output + feat_difference
        # print("===== att_and_diff shape ", att_and_diff.shape)

        fc_output = self.fc(feat_difference)
        # print("===== fc_output ", fc_output)
        # print("===== fc_output ", fc_output.shape)

        final_input = torch.cat((fc_output, fc_attn_output, diff, cos_similarity), dim=1)
        # print("===== final ", final_input)
        # print("===== final_input shape ", final_input.shape)

        final_output = self.final_fc(final_input)

        # print("===== final_output ", final_output)
        # print("===== final_output shape ", final_output.shape)
        final_output = torch.sigmoid(final_output)
        return final_output
