import torch.nn as nn
from ikan.kat_1dgroup_torch import KAT_Group_Torch


class GroupKANLinearTorch(nn.Module):
    """
    Torch fallback of GroupKANLinear that avoids Triton kernels.
    """
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_mode="gelu",
        drop=0.0,
        num_groups=8,
        device=None,
    ):
        super().__init__()
        self.act = KAT_Group_Torch(num_groups=num_groups, mode=act_mode)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        x_origin_dim = x.ndim
        if x_origin_dim == 2:
            x = x.unsqueeze(1)
        x = self.act(x)
        x = self.drop(x)
        if x_origin_dim == 2:
            x = x.squeeze(1)
        return self.linear(x)


class GroupKANTorch(nn.Module):
    """
    Torch fallback of GroupKAN using GroupKANLinearTorch.
    """
    def __init__(
        self,
        layers_hidden,
        act_mode="gelu",
        drop=0.0,
        bias=True,
        num_groups=8,
        device=None,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.layers.append(
                GroupKANLinearTorch(
                    in_features=layers_hidden[i],
                    out_features=layers_hidden[i + 1],
                    bias=bias,
                    act_mode=act_mode,
                    drop=drop,
                    num_groups=num_groups,
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def get_groupkan_backend(backend):
    if backend == "torch":
        return GroupKANLinearTorch, GroupKANTorch
    if backend == "triton":
        from ikan.GroupKAN import GroupKANLinear, GroupKAN
        return GroupKANLinear, GroupKAN
    raise ValueError(f"Unsupported groupkan_backend: {backend}")
