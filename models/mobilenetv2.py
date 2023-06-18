import torch
import torch.nn as nn
from torchinfo import summary
# https://arxiv.org/pdf/1801.04381

class InvertedResidualBlock(nn.Module):
    expansion = 4
    def __init__(self, input_channels, output_channels, expansion, stride=1):
        super(InvertedResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=input_channels*expansion,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(input_channels*expansion),
            nn.ReLU(),
            nn.Conv2d(in_channels=input_channels*expansion,
                      out_channels=input_channels*expansion,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      groups=input_channels*expansion,
                      bias=False),
            nn.BatchNorm2d(input_channels*expansion),
            nn.ReLU(),
            nn.Conv2d(in_channels=input_channels*expansion,
                      out_channels=output_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(output_channels)
        )
        self.relu = nn.ReLU()
        
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=input_channels,
                          out_channels=output_channels,
                          kernel_size=1,
                          stride=stride,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(output_channels)
            )
        else:
            self.shortcut = nn.Sequential()
    
    def forward(self, x):
        z = self.block(x)
        z += self.shortcut(x)
        z = self.relu(z)
        return z

class MobileNetV2(nn.Module):
    configuration = [
#       t (expansion), c (output channels), n(repetition num), s(stride)
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1]
    ]
    def __init__(self, input_width, output_num, input_channels=3):
        super(MobileNetV2, self).__init__()
        self.current_channels = 32

        # preliminary layer
        self.preliminary = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=self.current_channels,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(self.current_channels),
            nn.ReLU()
        )
        
        self.layer1 = self.make_layer(self.configuration[0])
        self.layer2 = self.make_layer(self.configuration[1])
        self.layer3 = self.make_layer(self.configuration[2])
        self.layer4 = self.make_layer(self.configuration[3])
        self.layer5 = self.make_layer(self.configuration[4])
        self.layer6 = self.make_layer(self.configuration[5])
        self.layer7 = self.make_layer(self.configuration[6])

        # final layer
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=self.current_channels,
                      out_channels=1280,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.Flatten(),
            nn.Linear(in_features=1280*int(input_width/(2**5))**2, out_features=output_num),
            nn.Softmax(dim=1)
        )


    def forward(self,x):
        z = self.preliminary(x)
        
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)
        z = self.layer5(z)
        z = self.layer6(z)
        z = self.layer7(z)

        z = self.final(z)
        return z

    def make_layer(self, layer_cofig):
        expansion, output_channels, repetition_num, stride = layer_cofig

        layers = []
        for _ in range(repetition_num):
            layers.append(InvertedResidualBlock(self.current_channels, output_channels, expansion, stride=stride))
            self.current_channels = output_channels
            stride = 1
        return nn.Sequential(*layers)

if __name__ == "__main__":
    img = torch.randn(1,3,224,224)    
    # Testing MobileNetV2
    test_model = MobileNetV2(224,10)
    # print(test_model(img).size())
    summary(test_model, input_size=(1,3,224,224), col_names=["input_size","output_size","num_params"])