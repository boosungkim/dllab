import torch
import torch.nn as nn
from torchinfo import summary

class DenseBlock(nn.Module):
    def __init__(self, input_channels, growth_rate, repetition_num):
        super(DenseBlock, self).__init__()
        self.growth_rate = growth_rate

        self.layers = []
        for i in range(repetition_num):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(input_channels),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=input_channels, out_channels=4*input_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(4*input_channels),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=4*input_channels, out_channels=growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
                )
            )
    
    def forward(self,x):
        return self._forward_implementation(x)
    
    def _forward_implementation(self,x):
        z = x
        identities_concat = None
        # print(self.layers)
        for layer in self.layers:
            identities_concat = z
            if identities_concat:
                z = torch.cat([identities_concat, z],1)
            identities_concat = z
            print(z.size())
            z = layer(z)
            
        return z
    
    # def create_1x1(self, input_channels, output_channels):
    #     return nn.Sequential(
    #         nn.BatchNorm2d(input_channels),
    #         nn.ReLU(),
    #         nn.Conv2d(in_channels=input_channels, out_channels=4*input_channels, kernel_size=1, stride=1, padding=0, bias=False)
    #     )

    # def create_3x3(self, input_channels, output_channels):
    #     return nn.Sequential(
    #         nn.BatchNorm2d(input_channels),
    #         nn.ReLU(),
    #         nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1, bias=False)
    #     )
        

    
class TransitionBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(TransitionBlock, self).__init__()
        # batch norm
        # 1by1 conv2d
        # 2by2 avg pool
    
    def forward(self):
        pass

# architecture = [6,12,32,32]
class DenseNet(nn.Module):
    def __init__(self, architecture, input_width, output_num, growth_rate=32):
        super(DenseNet, self).__init__()
        self.architecture = architecture
        self.input_width = input_width
        self.output_num = output_num
        self.growth_rate = growth_rate

        # Initial layers
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.growth_rate, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.growth_rate),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.dense = DenseBlock(self.growth_rate, self.growth_rate, 6)

    def forward(self, x):
        z = self.initial(x)
        z = self.dense(z)
        return z
    
    # def create_dense_block(self, )
    

if __name__ == "__main__":
    test = torch.rand(1,3,224,224)
    model = DenseNet([5,12,32,32],224,10)
    print(model(test).size())