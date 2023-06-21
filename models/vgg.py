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
        z = self.block2(z)
        z = self.block3(z)
        z = self.block4(z)
        z = self.block5(z)
        z = self.flat(z)
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
                        # Changed output channel number from 4096 to 1000.
                        nn.Linear(512*int(input_width / 32)**2, 1000), # 32 = 2**5
                        nn.ReLU(),
                        # REMOVED BELOW TO WORK CIFAR10
                        # nn.Dropout(p=0.5),
                        # nn.Linear(4096,4096),
                        # nn.ReLU(),
                        # nn.Linear(4096, num_outputs),
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
                        nn.BatchNorm2d(layer), # Added to work on CIFAR10
                        nn.ReLU()
                    ]
                    num_channels = layer
                else:
                    layers_list += [
                        nn.MaxPool2d(kernel_size=2, stride=2)
                    ]
        layers = nn.Sequential(*layers_list)
        return num_channels, layers


if __name__ == "__main__":
    testing = VGGModel("VGG19", 224, 1000)
    t1 = torch.randn(1,3,224,224)
    print(testing(t1).size())
    summary(testing, input_size=(1,3,224,224), device="cpu")
