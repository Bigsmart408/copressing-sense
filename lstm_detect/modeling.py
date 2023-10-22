import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, num_classes):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes).to('cuda')
        )
        self.device = 'cuda'



    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        self.to(x.device)
        x = x.to('cuda')
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out