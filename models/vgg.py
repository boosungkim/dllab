import torch
import torch.nn as nn
from torchinfo import summary

CONFIGURATION = {
    "VGG11": [
        [64,'M'],
        [128,'M'],
        [256,256,'M'],
        [512,512,'M'],
        [512,512,'M']
    ],
    "VGG13": [
        [64,64,'M'],
        [128,128,'M'],
        [256,256,'M'],
        [512,512,'M'],
        [512,512,'M']
    ],
    "VGG16": [
        [64,64,'M'],
        [128,128,'M'],
        [256,256,256,'M'],
        [512,512,512,'M'],
        [512,512,512,'M']
    ],
    "VGG19": [
        [64,64,'M'],
        [128,128,'M'],
        [256,256,256,256,'M'],
        [512,512,512,512,'M'],
        [512,512,512,512,'M']
    ]
}

class VGGModel(nn.Module):
    # 
    # Pytorch implementation of the various VGG models from Very Deep Convulutional Networks For Large-Scale Image Recognition
    # 
    def __init__(self, architecture_name, input_width, num_output):
        super(VGGModel, self).__init__()
        self.architecture = self.create_architecture(CONFIGURATION.get(architecture_name), input_width, num_output)
        self.block1 = self.architecture[0]
        self.block2 = self.architecture[1]
        self.block3 = self.architecture[2]
        self.block4 = self.architecture[3]
        self.block5 = self.architecture[4]
        self.block6 = self.architecture[5]
        self.flat = nn.Flatten()


    def forward(self,x):
        z = self.block1(x)
        # print(z.size())     # torch.Size([64, 112, 112])
        z = self.block2(z)
        # print(z.size())     # torch.Size([128, 56, 56])
        z = self.block3(z)
        # print(z.size())     # torch.Size([256, 28, 28])
        z = self.block4(z)
        # print(z.size())     # torch.Size([512, 14, 14])
        z = self.block5(z)
        # print(z.size())     # torch.Size([512, 7, 7])
        z = self.flat(z)      # Flatten Conv2d result for FC layers
        # print(z.size())
        z = self.block6(z)
        return z

    
    def create_architecture(self, architecture, input_width, num_outputs):
        """
        Create the CNN architecture with num_outputs outputs in the end.
        
        Parameters
        -------------
        architecture  :   2D list of int and string
              Each entry is either the number of filters in the Conv2d layer or an indication
              of MaxPool2d
        num_outputs   :   int
              Number of output classes in the end of the network
        
        Returns
        -------------
        blocks_list   :   List of nn.Sequential
            List of PyTorch NN sequences, each representing one block
        """
        blocks_list = []
        num_next_input_channels = 3
        
        for block in architecture:
            num_next_input_channels, layers = self.create_block(block, num_next_input_channels)
            blocks_list.append(layers)
        
        # Final layer for all VGG models
        blocks_list.append(
            nn.Sequential(
                        nn.Linear(512*int(input_width / 32)**2, 4096), # 32 = 2
                        nn.ReLU(),
                        nn.Linear(4096,4096),
                        nn.ReLU(),
                        nn.Linear(4096, num_outputs),
                        nn.Dropout(p=0.5),
                        nn.Softmax(dim=1))
        )
        return blocks_list


    def create_block(self, block, num_next_input_channels):
        """
        Create a singular CNN block.
        
        Parameters
        -------------
        block   :   1D list of int and string of block
            Each entry is either the int of filters in the Conv2d layer or a str indication
            of MaxPool2d
        n       :   number for channels for the Conv2d layer
        'M'     :   MaxPool2d
        
        Returns
        -------------
        num_channels    :   int
            number of channels outputted and inputted to the next layer
        layers          :   nn.Sequential
            Pytorch NN sequence for one block
        """
        layers_list = []
        num_channels = num_next_input_channels
        for layer in block:
                if isinstance(layer, int):
                    layers_list += [
                        nn.Conv2d(in_channels=num_channels, out_channels=layer, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                        nn.ReLU()
                    ]
                    num_channels = layer
                else:
                    layers_list += [
                        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
                    ]
        layers = nn.Sequential(*layers_list)
        return num_channels, layers


if __name__ == "__main__":
    testing = VGGModel("VGG19", 224, 1000)
    # t1 = torch.randn(1,3,224,224)
    # print(testing(t1).size())
    summary(testing, input_size=(1,3,224,224), device="cpu")

# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# VGGModel                                 [1, 1000]                 --
# ├─Sequential: 1-1                        [1, 64, 112, 112]         --
# │    └─Conv2d: 2-1                       [1, 64, 224, 224]         1,792
# │    └─ReLU: 2-2                         [1, 64, 224, 224]         --
# │    └─Conv2d: 2-3                       [1, 64, 224, 224]         36,928
# │    └─ReLU: 2-4                         [1, 64, 224, 224]         --
# │    └─MaxPool2d: 2-5                    [1, 64, 112, 112]         --
# ├─Sequential: 1-2                        [1, 128, 56, 56]          --
# │    └─Conv2d: 2-6                       [1, 128, 112, 112]        73,856
# │    └─ReLU: 2-7                         [1, 128, 112, 112]        --
# │    └─Conv2d: 2-8                       [1, 128, 112, 112]        147,584
# │    └─ReLU: 2-9                         [1, 128, 112, 112]        --
# │    └─MaxPool2d: 2-10                   [1, 128, 56, 56]          --
# ├─Sequential: 1-3                        [1, 256, 28, 28]          --
# │    └─Conv2d: 2-11                      [1, 256, 56, 56]          295,168
# │    └─ReLU: 2-12                        [1, 256, 56, 56]          --
# │    └─Conv2d: 2-13                      [1, 256, 56, 56]          590,080
# │    └─ReLU: 2-14                        [1, 256, 56, 56]          --
# │    └─Conv2d: 2-15                      [1, 256, 56, 56]          590,080
# │    └─ReLU: 2-16                        [1, 256, 56, 56]          --
# │    └─Conv2d: 2-17                      [1, 256, 56, 56]          590,080
# │    └─ReLU: 2-18                        [1, 256, 56, 56]          --
# │    └─MaxPool2d: 2-19                   [1, 256, 28, 28]          --
# ├─Sequential: 1-4                        [1, 512, 14, 14]          --
# │    └─Conv2d: 2-20                      [1, 512, 28, 28]          1,180,160
# │    └─ReLU: 2-21                        [1, 512, 28, 28]          --
# │    └─Conv2d: 2-22                      [1, 512, 28, 28]          2,359,808
# │    └─ReLU: 2-23                        [1, 512, 28, 28]          --
# │    └─Conv2d: 2-24                      [1, 512, 28, 28]          2,359,808
# │    └─ReLU: 2-25                        [1, 512, 28, 28]          --
# │    └─Conv2d: 2-26                      [1, 512, 28, 28]          2,359,808
# │    └─ReLU: 2-27                        [1, 512, 28, 28]          --
# │    └─MaxPool2d: 2-28                   [1, 512, 14, 14]          --
# ├─Sequential: 1-5                        [1, 512, 7, 7]            --
# │    └─Conv2d: 2-29                      [1, 512, 14, 14]          2,359,808
# │    └─ReLU: 2-30                        [1, 512, 14, 14]          --
# │    └─Conv2d: 2-31                      [1, 512, 14, 14]          2,359,808
# │    └─ReLU: 2-32                        [1, 512, 14, 14]          --
# │    └─Conv2d: 2-33                      [1, 512, 14, 14]          2,359,808
# │    └─ReLU: 2-34                        [1, 512, 14, 14]          --
# │    └─Conv2d: 2-35                      [1, 512, 14, 14]          2,359,808
# │    └─ReLU: 2-36                        [1, 512, 14, 14]          --
# │    └─MaxPool2d: 2-37                   [1, 512, 7, 7]            --
# ├─Flatten: 1-6                           [1, 25088]                --
# ├─Sequential: 1-7                        [1, 1000]                 --
# │    └─Linear: 2-38                      [1, 4096]                 102,764,544
# │    └─ReLU: 2-39                        [1, 4096]                 --
# │    └─Linear: 2-40                      [1, 4096]                 16,781,312
# │    └─ReLU: 2-41                        [1, 4096]                 --
# │    └─Linear: 2-42                      [1, 1000]                 4,097,000
# │    └─Dropout: 2-43                     [1, 1000]                 --
# │    └─Softmax: 2-44                     [1, 1000]                 --
# ==========================================================================================
# Total params: 143,667,240
# Trainable params: 143,667,240
# Non-trainable params: 0
# Total mult-adds (Units.GIGABYTES): 19.65
# ==========================================================================================
# Input size (MB): 0.60
# Forward/backward pass size (MB): 118.89
# Params size (MB): 574.67
# Estimated Total Size (MB): 694.16
# ==========================================================================================