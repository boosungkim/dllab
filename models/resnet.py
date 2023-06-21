"""
This is a Pytorch implementation of the [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) paper.

The researchers behind ResNet aimed to address the common "degradation problem" encountered in deep convolutional networks.

When the depth of a CNN model is increased, it initially shows an improvement in performance 
but eventually degrades rapidly as the training accuracy plateus or even worsens over time. 

The common misconception is that this rapid degradation in accuracy is caused by overfitting. 
While overfitting due to exploding/vanishing gradients is expected in very deep 
networks, it is accounted for by nomalized initializations of the dataset and the 
intermediate Batch Normalization layers.

The degradation is definitely not caused by overfitting, as adding more layers actually causes 
the training error to increase. While the researchers in the paper are not sure, 
their conclusion is that “deep plain nets may have exponentially low convergence rates,” which 
can be prevented with Residual Learning.

[Accompanying blog post](https://boosungkim.com/blog/2023/resnet34-implementation/)
"""

import torch
import torch.nn as nn
from torchinfo import summary

# === Residual Block with no Bottleneck ===

class ResidualBlockNoBottleneck(nn.Module):
    expansion = 1
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlockNoBottleneck, self).__init__()
        self.expansion = 1

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_channels)
        )
        self.relu = nn.ReLU()
        
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm2d(output_channels)
            )
        else:
            self.shortcut = nn.Sequential()
    
    def forward(self, x):
        z = self.block(x)
        z += self.shortcut(x)
        z = self.relu(z)
        return z

# === Residual Block with Bottleneck ===


class ResidualBlockBottleneck(nn.Module):
    expansion = 4
    def __init__(self, input_channels, in_channels, stride=1):
        super(ResidualBlockBottleneck, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels*4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels*4)
        )
        self.relu = nn.ReLU()
        
        if stride != 1 or input_channels != self.expansion*in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=in_channels*4, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(in_channels*4)
            )
        else:
            self.shortcut = nn.Sequential()
    
    def forward(self, x):
        z = self.block(x)
        z += self.shortcut(x)
        z = self.relu(z)
        return z

# === ResNet ===


class ResNet(nn.Module):
    def __init__(self, architecture, input_width, output_num,  block_type=ResidualBlockNoBottleneck, input_channels=3):
        super(ResNet, self).__init__()
        self.architecture = architecture
        self.input_width = input_width
        self.output_num = output_num
        self.current_channels = 64

        # preliminary layer
        self.preliminary = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.make_layer(block_type, 64, self.architecture[0])
        self.layer2 = self.make_layer(block_type, 128, self.architecture[1], 2)
        self.layer3 = self.make_layer(block_type, 256, self.architecture[2], 2)
        self.layer4 = self.make_layer(block_type, 512, self.architecture[3], 2)

        # final layer
        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.current_channels*int(input_width/(2**5))**2, out_features=output_num),
            nn.Softmax(dim=1)
        )


    def forward(self,x):
        z = self.preliminary(x)
        z = self.maxpool1(z)
        
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)

        z = self.final(z)
        return z

    def make_layer(self, block_type, in_channels, repetition_num, stride=1):
        layers = []
        for _ in range(repetition_num):
            layers.append(block_type(self.current_channels, in_channels, stride=stride))
            self.current_channels = in_channels*block_type.expansion
            stride = 1
        return nn.Sequential(*layers)

# === Tensor Test ===


if __name__ == "__main__":
    # Testing the No Bottleneck Residual Block
    testing_residual_block_input = torch.randn(1,64,56,56)
    # testing_residual_block = ResidualBlockNoBottleneck(64, 64, stride=2)
    # print(testing_residual_block(testing_residual_block_input).size())

    # Testing the Bottleneck Residual Block
    # testing_residual_block = ResidualBlockBottleneck(64, 64, stride=2)
    # print(testing_residual_block(testing_residual_block_input).size())

    img = torch.randn(1,3,224,224)    
    # Testing ResNet
    test_model = ResNet([3,4,6,3],224,10, ResidualBlockBottleneck)
    # print(test_model(img).size())
    summary(test_model, input_size=(1,3,224,224), col_names=["input_size","output_size","num_params"])