import torch
import torch.nn as nn
import torch.nn.functional as F

class TICLNet_layerPool(nn.Module):
    def __init__(self):
        super(TICLNet_layerPool, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d( 1, 4, 5, stride=1, padding=0),# b, 4, 48
            nn.ReLU(True),
            nn.Conv1d(4, 4, 4, stride=2, padding=0), # b, 4, 23
            nn.ReLU(True),
            nn.Conv1d(4, 4, 4, stride=1, padding=0), # b, 4, 20
            nn.ReLU(True)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(4*20,16),
            nn.ReLU(True),
            nn.Linear(16,4),
            # nn.LogSoftmax(dim=1) this is combined in loss
        )

    def forward(self, x):
        x = x.view(-1,1,52)
        x = self.conv(x)
        x = x.view(-1,4*20)
        x = self.fc(x)
        return x