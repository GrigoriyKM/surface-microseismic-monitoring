import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels,),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x): 
        return self.conv(x)

    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
                # 40 40
        self.common_layer = nn.Sequential(DoubleConv(1,16),
                                          nn.MaxPool2d(kernel_size=(2,2)), #20х20
                                          DoubleConv(16,32), 
                                          nn.MaxPool2d(kernel_size=(2,2)), #10х10
                                          DoubleConv(32,64),
                                          nn.MaxPool2d(kernel_size=(2,2))) #5х5 
                                          

        self.layer1 = nn.Sequential(DoubleConv(64,128),
                                    nn.Flatten(),
                                    nn.Linear(5*5*128, 500),
                                    nn.ReLU(),
                                    nn.Linear(500, 6),
                                    nn.Sigmoid())
        
        self.layer2 = nn.Sequential(DoubleConv(64,128),
                                    nn.Flatten(),
                                    nn.Linear(5*5*128, 500),
                                    nn.ReLU(),
                                    nn.Linear(500, 2),
                                    nn.Sigmoid()
                                   )
        self.loss_fn = nn.MSELoss()

    def forward(self,x):
        x = self.common_layer(x)
        X1 = self.layer1(x)
        X2 = self.layer2(x)
        output = torch.cat([X1, X2], dim=1)
        return output