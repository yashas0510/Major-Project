import torch
import torch.nn as nn
import pretrainedmodels

class DeepfakeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.xception = pretrainedmodels.__dict__['xception'](pretrained='imagenet')
        self.xception.last_linear = nn.Identity()
        self.lstm = nn.LSTM(2048, 1024, num_layers=2, batch_first=True, dropout=0.3)   # MUST MATCH TRAINING
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(1024, 2)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        feats = self.xception.features(x)
        feats = nn.AdaptiveAvgPool2d(1)(feats).view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(feats)
        out = self.dropout(lstm_out[:, -1])
        out = self.classifier(out)
        return out

def load_model(model_path, device):
    model = DeepfakeNet().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
