import torch
import torch.nn as nn

from torchinfo import summary

EfficientNet_B0 = [
    # expansion, channels, layer_num, kernel_size, stride 
    [1, 16, 1, 3, 1],
    [6, 24, 2, 3, 2],
    [6, 40, 2, 5, 2],
    [6, 80, 3, 3, 2],
    [6, 112, 3, 5, 1],
    [6, 192, 4, 5, 2],
    [6, 320, 1, 3, 1]
]

# phi_value, resolution, drop_rate
phi_values = (0, 224, 0.2)

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, groups=1, bias=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, groups=groups)
        self.batch = nn.BatchNorm2d(output_channels)
        self.silu = nn.SiLU()

    def forward(self,x):
        z = self.conv(x)
        z = self.batch(z)
        return self.silu(z)
        

class SE_Net(nn.Module):
    def __init__(self, input_channels, reduction_ratio):
        super(SE_Net,self).__init__()
        self.sequence = nn.Sequential(
            # 1. Squeeze
            nn.AdaptiveAvgPool2d((1,1)), # output: bxCx1x1
            nn.Flatten(),  # output: bxC
            # 2. Excitation
            nn.Linear(input_channels, input_channels // reduction_ratio, bias=False), # output: bxC/r
            nn.SiLU(), # output: bxC/r          USE SILU INSTEAD OF RELU
            nn.Linear(input_channels // reduction_ratio, input_channels), # output: bxC
            nn.Sigmoid(), # output: bxC
            nn.Unflatten(1, (input_channels,1,1)) # output: bxCx1x1
        )

    def forward(self,x):
        z = self.sequence(x)
        # 3. Rescale
        z = x*z # output: bxCxHxW
        return z

class InvertedResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, expansion, reduction_ratio=4, survival_prob=0.5):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.residual = input_channels == output_channels and stride == 1
        hidden_dim = input_channels*expansion
        self.expand = input_channels != hidden_dim

        if self.expand:
            self.expand_conv = ConvBlock(input_channels, hidden_dim, kernel_size=3, stride=1, padding=1)

        self.conv = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim),
            SE_Net(hidden_dim, input_channels // reduction_ratio),
            nn.Conv2d(hidden_dim, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self,x):
        z = self.expand_conv(x) if self.expand else x

        if self.residual:
            return self.stochastic_depth(self.conv(z))
        else:
            return self.conv(z)
    
    def stochastic_depth(self, x):
        if not self.training:
            return x
        else:
            binary_tensor = torch.rand(x.shape[0],1,1,1,device=x.device) < self.survival_prob
            return torch.div(x, self.survival_prob) * binary_tensor

class EfficientNet(nn.Module):
    def __init__(self, version, num_output):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = int(1280*width_factor)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_output)
        )
    def forward(self, x):
        z = self.pool(self.features(x))
        z = self.classifier(z.view(z.shape[0],-1))
        return z
    
    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, resolution, drop_rate = phi_values
        depth_factor = alpha**phi
        width_factor = beta**phi
        return width_factor, depth_factor, drop_rate
    
    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32*width_factor)
        features = [ConvBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, num_layers, kernel_size, stride in EfficientNet_B0:
            output_channels = 4*int(channels*width_factor/4)
            layer_num = int(num_layers*depth_factor)

            for layer in range(layer_num):
                features.append(InvertedResidualBlock(in_channels,
                                                      output_channels,
                                                      kernel_size=kernel_size,
                                                      stride=stride if layer == 0 else 1,
                                                      padding=kernel_size//2,
                                                      expansion=expand_ratio))
                in_channels = output_channels
        features.append(ConvBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0))
        return nn.Sequential(*features)

if __name__ == "__main__":
    model = EfficientNet(EfficientNet_B0, 10)
    # test = torch.rand(1,3,224,224)
    # print(model(test).size())
    summary(model, input_size=(1,3,224,224), col_names=["input_size","output_size","num_params"])

