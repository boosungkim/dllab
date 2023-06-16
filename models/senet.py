import torch
import torch.nn as nn
from torchinfo import summary

class se_block(nn.Module):
    def __init__(self, input_channels, reduction_ratio):
        super(se_block,self).__init__()
        self.sequence = nn.Sequential(
            # 1. Squeeze
            nn.AdaptiveAvgPool2d((1,1)), # output: bxCx1x1
            nn.Flatten(),  # output: bxC
            # 2. Excitation
            nn.Linear(input_channels, input_channels // reduction_ratio, bias=False), # output: bxC/r
            nn.ReLU(), # output: bxC/r
            nn.Linear(input_channels // reduction_ratio, input_channels), # output: bxC
            nn.Sigmoid(), # output: bxC
            nn.Unflatten(1, (input_channels,1,1)) # output: bxCx1x1
        )

    def forward(self,x):
        z = self.sequence(x)
        # 3. Rescale
        z = x*z # output: bxCxHxW
        return z

    

if __name__ == "__main__":
    model = se_block(64, 16)
    # test = torch.rand(1,256,8,8)
    # print(model(test).size())
    summary(model, input_size=(1,64,32,32), col_names=["input_size","output_size","num_params"])