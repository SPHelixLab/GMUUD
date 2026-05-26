from Resnet_deep import ResNet18,ResNet34,ResNet50
from Resnet_shallow import ResNet8
import torch.nn as nn
import torch.nn.functional as F

''' MLP '''
class MLP(nn.Module):
    def __init__(self,  num_classes, channel=1):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(28*28*1 if channel==1 else 32*32*3, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out

def get_model(model_name,numclass):
    if model_name == 'ResNet50':
        model = ResNet50(num_classes = numclass)
    elif model_name == 'ResNet34':
        model = ResNet34(num_classes=numclass)
    elif model_name == 'ResNet18':
        model = ResNet18(num_classes=numclass)
    elif model_name == 'ResNet8':
        model = ResNet8(num_classes=numclass)
    elif model_name == 'MLP':
        model = MLP(num_classes=numclass)
    return model