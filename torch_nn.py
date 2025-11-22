import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = self.downsampling_conv()
        self.dense = self.dense_layers()
        self.flatten = nn.Flatten()

    def downsampling_conv(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2),   
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),  
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),   
            nn.ReLU()
        )

    def dense_layers(self):
        return nn.Sequential(
            nn.Linear(in_features=12800, out_features=64),   
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=5),  
            nn.Softmax(dim=1) 
        )

    def forward(self,obs):

        out = self.down(obs)
        out = self.flatten(out)
        out = self.dense(out)

        return out


















    