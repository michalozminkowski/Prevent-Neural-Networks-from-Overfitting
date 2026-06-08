import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(MLP, self).__init__()
        
        layers = []
        layers.append(nn.Flatten())
        
        layers.append(nn.Linear(28 * 28, 1024))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))
            
        layers.append(nn.Linear(1024, 1024))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))
            
        layers.append(nn.Linear(1024, 10))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
