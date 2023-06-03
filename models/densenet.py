import torch
import torch.nn as nn
from torchinfo import summary

class DenseLayer(nn.Module):
    def __init__(self, input_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.growth_rate = growth_rate

        self.layer = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=input_channels, out_channels=4*input_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4*input_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=4*input_channels, out_channels=growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
    
    def forward(self,x):
        return self._forward_implementation(x)
    
    def _forward_implementation(self,x):
        z = self.layer(x)
        z = torch.cat([z, x], 1)
        return z

    
class TransitionBlock(nn.Module):
    def __init__(self, input_channels):
        super(TransitionBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = input_channels // 2

        self.layer = nn.Sequential(
            nn.BatchNorm2d(self.input_channels),
            nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
    
    def forward(self,x):
        return self.layer(x)
    
    def output_channels_num(self):
        return self.output_channels


class DenseNet(nn.Module):
    def __init__(self, architecture, input_width, output_num, growth_rate=32):
        super(DenseNet, self).__init__()
        self.architecture = architecture
        self.input_width = input_width
        self.output_num = output_num
        self.growth_rate = growth_rate

        # Initial layers
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.growth_rate*2, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.growth_rate*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Main section
        input_channels = self.growth_rate*2
        self.dense_block1, input_channels = self.create_dense_block(input_channels, architecture[0])
        self.transition_layer1 = TransitionBlock(input_channels)
        input_channels = self.transition_layer1.output_channels_num()

        self.dense_block2, input_channels = self.create_dense_block(input_channels, architecture[1])
        self.transition_layer2 = TransitionBlock(input_channels)
        input_channels = self.transition_layer2.output_channels_num()

        self.dense_block3, input_channels = self.create_dense_block(input_channels, architecture[2])
        self.transition_layer3 = TransitionBlock(input_channels)
        input_channels = self.transition_layer3.output_channels_num()

        self.dense_block4, input_channels = self.create_dense_block(input_channels, architecture[3])

        # Classification layer
        self.avgpool = nn.AvgPool2d(kernel_size=int(input_width/(2**5)))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=1*1*input_channels, out_features=output_num, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        z = self.initial(x)
        z = self.dense_block1(z)
        z = self.transition_layer1(z)

        z = self.dense_block2(z)
        z = self.transition_layer2(z)

        z = self.dense_block3(z)
        z = self.transition_layer3(z)

        z = self.dense_block4(z)
        print(z.size())
        z = self.avgpool(z)
        z = self.flatten(z)
        z = self.fc(z)
        return self.softmax(z)
    
    def create_dense_block(self, input_channels, repetition_num):
        layers = []
        in_channels = input_channels
        for _ in range(repetition_num):
            layers.append(DenseLayer(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        
        return nn.Sequential(*layers), in_channels

    

if __name__ == "__main__":
    model = DenseNet([6,12,24,16],224,10,12)
    test = torch.rand(1,3,224,224)
    print(model(test).size())
    # summary(model, input_size=(1,3,224,224), col_names=["input_size","output_size","num_params"])