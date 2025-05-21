import torch.nn as nn

class SketchNet(nn.Module):
    def __init__(self):
        super(SketchNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))