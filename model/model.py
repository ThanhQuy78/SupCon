import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetEncoder(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super().__init__()
        if backbone=='resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            self.feature_dim = 2048
        elif backbone=='resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.feature_dim = 512
        elif backbone=='resnet101':
            self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
            self.feature_dim = 2048
        else:
            raise ValueError("Unsupported backbone")
        self.model.fc = nn.Identity()
    def forward(self,x):
        return self.model(x)


class ProjectionHead(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim,output_dim)
        )
    def forward(self,x):
        x = self.net(x)
        x = F.normalize(x,dim=1)  
        return x


class ClassifierHead(nn.Module):
    def __init__(self,input_dim,num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim,num_classes)
        
    def forward(self,x):
        return self.fc(x)