from typing import Any
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple, Union, Optional
from tensordict.nn import TensorDictModule


class ConvBn(nn.Module):
    def __init__(
        self,
        dim: str,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] | Tuple[int, int],
        stride: int,
        padding: Union[str, int],
        bias: bool,
    ) -> None:
        """
        Args:
        """
        super().__init__()
        conv = getattr(nn, f'Conv{dim}')
        bn = getattr(nn, f'BatchNorm{dim}')

        self.conv = conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = bn(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
        Returns:
        """
        return self.bn(self.conv(x))


def ConvBn2d(**kwargs) -> nn.Module:
    return ConvBn(dim='2d', **kwargs)


def ConvBn3d(**kwargs) -> nn.Module:
    return ConvBn(dim='3d', **kwargs)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        dim: str,
        activation: str,
        **kwargs: Any,
    ) -> None:
        """
        Args:
        """
        super().__init__()
        kwargs |= {'dim': dim}
        in_channels = kwargs['in_channels']
        out_channels = kwargs['out_channels']
        bias = kwargs['bias']
        self.convbn1 = ConvBn(**kwargs)
        self.activation = getattr(nn, activation)() 
        kwargs['in_channels'] = out_channels
        self.convbn2 = ConvBn(**kwargs)
        self.res = getattr(nn, f'Conv{dim}')(in_channels, 
                                             out_channels, 
                                             kernel_size=1, 
                                             bias=bias,
                                             )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
        Returns:
        """
        residual = self.res(x)
        x = self.activation(self.convbn1(x))
        x = self.convbn2(x) + residual
        return x


def ResidualBlock2d(**kwargs) -> nn.Module:
    return ResidualBlock(dim='2d', **kwargs)

def ResidualBlock3d(**kwargs) -> nn.Module:
    return ResidualBlock(dim='3d', **kwargs)


class ME0SegCNN3d(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels_list: list[int] = [2, 4],
        kernel_size: list[int] | tuple[int, int, int] = (3, 3, 3),
        stride: int = 1,
        padding: Union[str, int] = "same",
        activation: str = "ReLU",
        bias: bool = False,
    ) -> None:
        """
        Args:
        """
        super().__init__()
        if isinstance(kernel_size, list):
            kernel_size = tuple(kernel_size)
        self.activation = getattr(nn, activation)() 
        self.residual_block_list = nn.ModuleList()

        Block = ResidualBlock3d
        for ch in hidden_channels_list:
            block = Block(
                in_channels=in_channels,
                out_channels=ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                activation=activation,
                bias=bias,
            )
            self.residual_block_list.append(block)
            in_channels = ch

        self.head = Block(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation=activation,
            bias=bias,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
        Returns:
        """
        dim = x.dim()
        if dim == 4: # (N, 8, 6 192)
            x = x.unsqueeze(dim=1) # add channel dim

        mask = x[:,0].eq(0)
        for block in self.residual_block_list:
            x = self.activation(block(x))
        x = self.head(x)
        x = x.squeeze()

        preds = F.sigmoid(x).masked_fill(mask, 0.)

        return x, preds

    def to_tensor_dict_module(self):
        return TensorDictModule(
            module=self,
            in_keys=['input'],
            out_keys=['logits', 'preds']
        )
