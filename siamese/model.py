import torch
import torch.nn as nn
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights


class ScaledDotAttnModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.scaled_dot_attn = nn.functional.scaled_dot_product_attention

    def forward(self, query, key, value, dropout_p=0.3):
        return self.scaled_dot_attn(query, key, value, dropout_p=dropout_p)


class SiameseNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.net_org = efficientnet_b7(weights=EfficientNet_B7_Weights)

        # remove the last layer of backbone (linear layer which is before last layer)
        self.backbone = torch.nn.Sequential(*(list(self.net_org.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(512 * 10, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
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

        attention_output = self.scaled_dot_attn(output1, output2, output1)
        attention_output = nn.functional.normalize(attention_output, dim=1)

        output = torch.pow((output1 - output2), 2)
        output = nn.functional.normalize(output, dim=1)

        combined_output = torch.cat((attention_output, output), dim=1)

        # pass the difference to the linear layers
        output = self.fc(combined_output)
        output = self.sigmoid(output)

        return output
