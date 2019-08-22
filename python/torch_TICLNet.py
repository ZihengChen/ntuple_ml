import torch
import torch.nn as nn
import torch.nn.functional as F

class TICLNet(nn.Module):
    def __init__(self):
        super(TICLNet, self).__init__()
        self.rnn = LayerSummarizeNet()
        self.cnn = ClassifyNet()
        
    def forward(self, batchEvents):
        batchSummary = self.summarizeBatchEvents(batchEvents)
        output = self.cnn(batchSummary)
        return output # (batch_size, nClasses=4)
        
    def summarizeBatchEvents(self, batchEvents):
        batchSummary = []
        for event in batchEvents:
            eventSummary = self.sumarizeEvent(event)
            batchSummary.append(eventSummary)
        batchSummary = torch.cat(batchSummary)
        return batchSummary # (batch_size * nChannel=16, Length=50)
    
    def sumarizeEvent(self, event):
        eventSummary = []
        for layer in event:
            layerSummary = self.summarizeLayer(layer)
            eventSummary.append(layerSummary)
        eventSummary = torch.cat(eventSummary).transpose(1,0)
        return eventSummary # (nChannel=16, Length=50)
    
    def summarizeLayer(self, layer):
        if layer.shape[0] > 0:
            layer = torch.tensor(layer)
            layerSummary = self.rnn(layer)
        else:
            layerSummary = torch.zeros(1,16)
        return layerSummary # (1, nChannel=16)
    
    
class LayerSummarizeNet(nn.Module):
    
    def __init__(self):
        super(LayerSummarizeNet, self).__init__()
        self.input_size = 3
        self.hidden_size = 20
        self.output_size = 16
    
        self.lstm1 = nn.LSTMCell(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):
        batch_size = 1
        h_t = torch.zeros(batch_size, self.hidden_size)
        c_t = torch.zeros(batch_size, self.hidden_size) 

        for i, input_t in enumerate(input.chunk(input.size(0), dim=0)):
            # shape of input_t is (1,3) = (batch_size, input_size)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
        
        output = torch.sigmoid(self.linear(h_t))
        return output
    
class ClassifyNet(nn.Module):
    
    def __init__(self):
        super(ClassifyNet, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(16, 32, 5, stride=1, padding=0), # b, 32, 46
            nn.ReLU(True),
            nn.Conv1d(32, 32, 4, stride=2, padding=0), # b, 32, 21
            nn.ReLU(True),
            nn.Conv1d(32, 32, 4, stride=1, padding=0), # b, 32, 19
            nn.ReLU(True)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(32*19,16),
            nn.ReLU(True),
            nn.Linear(16,4),
            # nn.LogSoftmax(dim=1) this is combined in loss
        )

    def forward(self, x):
        x = x.view(-1,16,50)
        x = self.conv(x)
        x = x.view(-1,32*19)
        x = self.fc(x)
        return x

