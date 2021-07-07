import torch
import torch.nn as nn
import torch.nn.functional as F


class PokeCNN(nn.Module):
    def __init__(self, class_count=1):
        super().__init__()
        conv_dimen = 6
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=2)
        self.conv1_2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=2)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(in_channels=conv_dimen, out_channels=conv_dimen * 2, kernel_size=3, stride=2)
        self.conv2_2 = nn.Conv2d(in_channels=conv_dimen * 2, out_channels=conv_dimen * 2, kernel_size=3, stride=2)


        self.fc1 = nn.Linear(in_features=15376, out_features=5000)
        self.fc2 = nn.Linear(in_features=5000, out_features=500)
        self.fc_final = nn.Linear(in_features=500, out_features=class_count)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
#        x = self.pool(x)
        x = F.relu(self.conv1_2(x))
 #       x = self.pool(x)

        x = torch.flatten(x, 1)


        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_final(x)
        x = F.softmax(x, dim=1)

        return x
