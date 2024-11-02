import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torchvision.models as models


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
class RotNetModel(BaseModel):
    def __init__(self, pretrained=True, fine_tune=True, num_classes=360):
        super().__init__()
        if pretrained:
            print('[INFO]: Loading pre-trained weights')
        else:
            print('[INFO]: Not loading pre-trained weights')
        self.model = models.efficientnet_b1(pretrained=pretrained)
        if fine_tune:
            print('[INFO]: Fine-tuning all layers...')
            for params in self.model.parameters():
                params.requires_grad = True
        elif not fine_tune:
            print('[INFO]: Freezing hidden layers...')
            for params in self.model.parameters():
                params.requires_grad = False

        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
