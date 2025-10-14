import torch.nn as nn
from modules.unet_parts import DoubleConv, DownSample, UpSample

class UNet(nn.Module):
    def __init__(self, in_channels, widths, num_classes):
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
