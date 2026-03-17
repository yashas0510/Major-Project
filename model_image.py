import torch.nn as nn
import pretrainedmodels

class DeepfakeImgNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.xception = pretrainedmodels.__dict__['xception'](pretrained='imagenet')
        self.xception.last_linear = nn.Identity()
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(2048, 2)
    def forward(self, x):
        feats = self.xception.features(x)
        feats = nn.AdaptiveAvgPool2d(1)(feats)
        feats = feats.view(feats.size(0), -1)
        out = self.dropout(feats)
        out = self.classifier(out)
        return out
