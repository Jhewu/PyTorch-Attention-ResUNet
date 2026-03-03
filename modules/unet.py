import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        Standard UNet DoubleConvBlocks with ReLU activations
        """
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_op(x)

class DownSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        DoubleConv followed by MaxPool2d for Downsampling
        """
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        down = self.conv(x)
        p = self.pool(down)
        return down, p

class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        UpSample with ConvTranspose2d followed by Concatenation and DoubleConv
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
       x1 = self.up(x1)
       x = torch.cat([x1, x2], 1)
       return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels: int, widths: list[int], num_classes: int):
        super().__init__()

        self.input = DownSample(in_channels, widths[0])

        self.downblocks = nn.ModuleList()
        for i in range(len(widths) - 2):
            self.downblocks.append(DownSample(widths[i], widths[i+1]))

        self.bottle_neck = DoubleConv(widths[-2], widths[-1])

        self.upblocks = nn.ModuleList()
        for i in range(len(widths) - 2, -1, -1):
            self.upblocks.append(UpSample(widths[i+1], widths[i]))

        self.output = nn.Conv2d(in_channels=widths[0], out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        x_down, p_down = self.input(x)
        skips.append(x_down)

        current_input = p_down
        for down_block in self.downblocks:
            x_down, p_down = down_block(current_input)
            skips.append(x_down)
            current_input = p_down

        b = self.bottle_neck(current_input)

        up_output = b
        for i, up_block in enumerate(self.upblocks):
            skip_connection = skips[-(i + 1)]
            up_output = up_block(up_output, skip_connection)

        out = self.output(up_output)
        return out
    
    def predict(self, x: torch.Tensor, binarize: bool = True) -> torch.Tensor: 
        out = self.forward(x)
        if binarize:
            out = (torch.sigmoid(out) > 0.5).float()  
        return out
