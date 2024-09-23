import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchsummary import summary


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)

class NestedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(NestedUNet, self).__init__()
        
        # Initialize lists to hold layers
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Downsampling path
        self.down_convs = nn.ModuleList()
        for feature in features:
            self.down_convs.append(DoubleConv(in_channels=in_channels, out_channels=feature))
            in_channels = feature
        
        # Bottleneck layer
        self.bottleneck = DoubleConv(in_channels=features[-1], out_channels=features[-1]*2)
        
        # Upsampling path
        self.up_convs = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(in_channels=feature*2, out_channels=feature, kernel_size=2, stride=2))
            self.up_convs.append(DoubleConv(in_channels=feature*2, out_channels=feature))
            
        # Final output layer
        self.final_conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Downsampling
        for down in self.down_convs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Upsampling with nested skip connections
        skip_connections = skip_connections[::-1]  # Reverse the list
        for idx in range(0, len(self.ups)):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx]
            
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            # Concatenate the skip connection and the upsampled output
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.up_convs[idx](concat_skip)
        
        return self.final_conv(x)


if __name__ == "__main__":
    model = NestedUNet(in_channels=1, out_channels=1)
    summary(model.cuda(), (1, 572, 572))
