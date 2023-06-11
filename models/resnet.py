import torch
import torch.nn as nn
from torchinfo import summary


class ResidualBlockNoBottleneck(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlockNoBottleneck, self).__init__()

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


class ResidualBlockBottleneck(nn.Module):
    def __init__(self):
        pass

class ResNet(nn.Module):
    # 
    # Pytorch implementation of the ResNet model from Deep Residual Learning for Image Recognition
    # References:
            # @article{DBLP:journals/corr/HeZRS15,
            # author    = {Kaiming He and
            #         Xiangyu Zhang and
            #         Shaoqing Ren and
            #         Jian Sun},
            # title     = {Deep Residual Learning for Image Recognition},
            # journal   = {CoRR},
            # volume    = {abs/1512.03385},
            # year      = {2015},
            # url       = {http://arxiv.org/abs/1512.03385},
            # archivePrefix = {arXiv},
            # eprint    = {1512.03385},
            # timestamp = {Wed, 17 Apr 2019 17:23:45 +0200},
            # biburl    = {https://dblp.org/rec/journals/corr/HeZRS15.bib},
            # bibsource = {dblp computer science bibliography, https://dblp.org}
            # }

    def __init__(self, architecture, input_width, output_num, input_channels=3, block_type=ResidualBlockNoBottleneck):
        super(ResNet, self).__init__()
        self.architecture = architecture
        self.input_width = input_width
        self.output_num = output_num
        self.block_type = block_type

        # preliminary layer
        self.preliminary = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.make_layer(64, 64, self.architecture[0])
        self.layer2 = self.make_layer(64, 128, self.architecture[1], 2)
        self.layer3 = self.make_layer(128, 256, self.architecture[2], 2)
        self.layer4 = self.make_layer(256, 512, self.architecture[3], 2)

        # final layer
        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512*int(input_width/(2**5))**2, out_features=output_num),
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

    def make_layer(self, input_channels, output_channels, repetition_num, stride=1):
        layers = []
        input_channels = input_channels
        output_channels = output_channels
        for _ in range(repetition_num):
            layers.append(self.block_type(input_channels, output_channels, stride=stride))
            input_channels = output_channels
            stride = 1
        return nn.Sequential(*layers)

if __name__ == "__main__":
    # Testing the No Bottleneck Residual Block
    # testing_residual_block_input = torch.randn(1,64,112,112)
    # testing_residual_block = ResidualBlockNoBottleneck(64, 64, stride=2)
    # print(testing_residual_block(testing_residual_block_input).size())

    img = torch.randn(1,3,224,224)    
    # Testing ResNet
    test_model = ResNet([3,4,6,3],224,10)
    # print(test_model(img).size())
    summary(test_model, input_size=(1,3,224,224), col_names=["input_size","output_size","num_params"])