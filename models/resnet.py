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

class CustomResNet34(nn.Module):
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

    def __init__(self, architecture, input_width, input_channels, output_num):
        super(CustomResNet34, self).__init__()
        self.architecture = architecture
        output_channels = 64
        # preliminary layer
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        input_channels = output_channels
        output_channels = 64
        # first layer
        # self.resblock1 = ResidualBlockNoBottleneck(input_channels, output_channels)

        self.conv2_1 = self.create_layer(input_channels, output_channels)
        self.relu2_1 = nn.ReLU()

        self.conv2_2 = self.create_layer(output_channels, output_channels)
        self.relu2_2 = nn.ReLU()
        
        self.conv2_3 = self.create_layer(output_channels, output_channels)
        self.relu2_3 = nn.ReLU()

        # second layer
        input_channels = output_channels
        output_channels = 128

        self.dim_reduction1 = self.skip_connection(input_channels, output_channels)

        self.conv3_1 = self.create_layer(input_channels, output_channels, 2)
        self.relu3_1 = nn.ReLU()

        self.conv3_2 = self.create_layer(output_channels, output_channels)
        self.relu3_2 = nn.ReLU()

        self.conv3_3 = self.create_layer(output_channels, output_channels)
        self.relu3_3 = nn.ReLU()

        self.conv3_4 = self.create_layer(output_channels, output_channels)
        self.relu3_4 = nn.ReLU()


        # third layer
        input_channels = output_channels
        output_channels = 256

        self.dim_reduction2 = self.skip_connection(input_channels, output_channels)

        self.conv4_1 = self.create_layer(input_channels, output_channels, 2)
        self.relu4_1 = nn.ReLU()

        self.conv4_2 = self.create_layer(output_channels, output_channels)
        self.relu4_2 = nn.ReLU()

        self.conv4_3 = self.create_layer(output_channels, output_channels)
        self.relu4_3 = nn.ReLU()

        self.conv4_4 = self.create_layer(output_channels, output_channels)
        self.relu4_4 = nn.ReLU()

        self.conv4_5 = self.create_layer(output_channels, output_channels)
        self.relu4_5 = nn.ReLU()

        self.conv4_6 = self.create_layer(output_channels, output_channels)
        self.relu4_6 = nn.ReLU()

        # fourth layer
        input_channels = output_channels
        output_channels = 512

        self.dim_reduction3 = self.skip_connection(input_channels, output_channels)

        self.conv5_1 = self.create_layer(input_channels, output_channels, 2)
        self.relu5_1 = nn.ReLU()

        self.conv5_2 = self.create_layer(output_channels, output_channels)
        self.relu5_2 = nn.ReLU()

        self.conv5_3 = self.create_layer(output_channels, output_channels)
        self.relu5_3 = nn.ReLU()

        # final layer
        self.flat = nn.Flatten()
        self.fc = nn.Linear(in_features=512*int(input_width/(2**5))**2, out_features=output_num)
        self.softmax = nn.Softmax(dim=1)


    def forward(self,x):
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.relu1(z)
        z = self.maxpool1(z)
        
        identity = z
        # first layer
        # z = self.resblock1(z)

        z = self.conv2_1(z)
        z = z + identity
        z = self.relu2_1(z)
        identity = z
        z = self.conv2_2(z)
        z = z + identity
        z = self.relu2_2(z)
        identity = z
        z = self.conv2_3(z)
        z = z + identity
        z = self.relu2_3(z)
        
        # second layer
        identity = z
        z = self.conv3_1(z)
        z = z + self.dim_reduction1(identity)
        self.relu3_1(z)

        identity = z
        z = self.conv3_2(z)
        z = z + identity
        self.relu3_2(z)

        identity = z
        z = self.conv3_3(z)
        z = z + identity
        self.relu3_3(z)

        z = self.conv3_4(z)
        z = z + identity
        self.relu3_4(z)

        # third layer
        identity = z
        z = self.conv4_1(z)
        z = z + self.dim_reduction2(identity)
        z= self.relu4_1(z)

        identity = z
        z = self.conv4_2(z)
        z = z + identity
        z = self.relu4_2(z)

        identity = z
        z = self.conv4_3(z)
        z = z + identity
        z= self.relu4_3(z)

        z = self.conv4_4(z)
        z = z + identity
        z= self.relu4_4(z)

        z = self.conv4_5(z)
        z = z + identity
        z= self.relu4_5(z)

        z = self.conv4_6(z)
        z = z + identity
        z = self.relu4_6(z)

        # fourth layer
        identity = z
        z = self.conv5_1(z)
        z = z + self.dim_reduction3(identity)
        z = self.relu5_1(z)

        identity = z
        z = self.conv5_2(z)
        z = z + identity
        z = self.relu5_2(z)

        identity = z
        z = self.conv5_3(z)
        z = z + identity
        z = self.relu5_3(z)

        # final layer
        z = self.flat(z)
        z = self.fc(z)
        z = self.softmax(z)

        return z

    def create_layer(self, input_channels, output_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_channels)
        )
    
    def skip_connection(self, input_channels, output_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(output_channels)
        )

if __name__ == "__main__":
    img = torch.randn(1,3,224,224)

    testing_residual_block_input = torch.randn(1,64,112,112)
    testing_residual_block = ResidualBlockNoBottleneck(64, 64, stride=2)
    print(testing_residual_block(testing_residual_block_input).size())


    # test_model = CustomResNet34([],32,3,1000)
    # t = torch.randn(1,3,32,32)
    # print(test_model(t).size())
    # summary(test_model, input_size=(1,3,224,224), col_names=["input_size","output_size","num_params"])