import torch
import torch.nn as nn
from torchsummary import summary

CONFIGURATION = {
    "VGG11": [64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
    "VGG13": [64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
    "VGG16": [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    "VGG19": [64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']
}

class VGGModel(nn.Module):
    # 
    # Pytorch implementation of the various VGG models from Very Deep Convulutional Networks For Large-Scale Image Recognition
    # 
    def __init__(self, architecture_name, num_output):
        super(VGGModel, self).__init__()
        self.architecture = self.create_architecture(CONFIGURATION.get(architecture_name), num_output)


    def forward(self,x):
        z = self.architecture(x)
        return z

    
    def create_architecture(self, architecture, num_outputs):
        """
        Create the CNN architecture with num_outputs outputs in the end.
        
        Parameters
        -------------
        architecture  :   list of int and string
              Each entry is either the number of filters in the Conv2d layer or an indication
              of MaxPool2d
        num_outputs   :   int
              Number of output classes in the end of the network
        
        Returns
        -------------
        nn.Sequential
              Pytorch NN sequence
        """
        sequence_list = []
        num_next_input_channels = 3

        for layer in architecture:
            if isinstance(layer, int):
                sequence_list += [
                    nn.Conv2d(in_channels=num_next_input_channels, out_channels=layer, kernel_size=(3,3), stride=1, padding=1),
                    nn.ReLU()
                ]
                num_next_input_channels = layer
            else:
                sequence_list += [
                    nn.MaxPool2d(kernel_size=(2,2), stride=2)
                ]
        
        sequence_list += [
            nn.Flatten(),
            nn.Linear(7*7*512, 4096),
            nn.ReLU(),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Linear(4096, num_outputs),
            nn.Dropout(p=0.5),
            nn.Softmax(dim=1),
        ]
        print(sequence_list)
        sequence = nn.Sequential(*sequence_list)
        return sequence


    

if __name__ == "__main__":
    testing = VGGModel("VGG19", 1000)
    summary(testing, input_size=(3,224,224), device="cpu")