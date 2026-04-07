import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class FractalBlock(nn.Module):
    def __init__(self, depth, channels):
        super().__init__()
        self.depth = depth
        if depth == 1:
            self.base_conv = ConvBlock(channels, channels)
        else:
            self.left = FractalBlock(depth - 1, channels)
            self.right1 = FractalBlock(depth - 1, channels)
            self.right2 = FractalBlock(depth - 1, channels)

    def forward(self, x):
        if self.depth == 1:
            return self.base_conv(x)
        else:
            left_out = self.left(x)
            right_out = self.right2(self.right1(x))
            return (left_out + right_out) / 2.0

class FullFractalNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Folosim o convoluție inițială pentru a aduce imaginea la 64 canale
        self.init_conv = ConvBlock(3, 64) 
        # Adâncime 3 este un echilibru perfect între complexitate și viteză de antrenare
        self.fractal = FractalBlock(depth=3, channels=64) 
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.fractal(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)