import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from .groupkan_backend import get_groupkan_backend
from .kaa_grit_layer import KaaGritLayer


class KaaGritEncoder(nn.Module):
    """
    Lightweight KAA-GRIT encoder.

    Returns:
        z: node embeddings [N, out_dim]
        node_alpha: node-level importance [N]
        edge_alpha: edge-level attention [E, H]
    """
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_heads=4,
        num_layers=2,
        k_steps=5,
        dropout=0.1,
        num_groups=8,
        edge_num_groups=1,
        alpha_temperature=1.0,
        act_mode="swish",
        attn_act="relu",
        attn_clamp=5.0,
        signed_sqrt=True,
        edge_enhance=True,
        ffn_mult=4,
        groupkan_backend="triton",
        device=None,
    ):
        super().__init__()
        if in_dim % num_groups != 0 or out_dim % num_groups != 0:
            raise ValueError("in_dim and out_dim must be divisible by num_groups")
        if hidden_dim % num_groups != 0:
            raise ValueError("hidden_dim must be divisible by num_groups")
        if alpha_temperature <= 0:
            raise ValueError("alpha_temperature must be > 0")

        self.alpha_temperature = float(alpha_temperature)

        GroupKANLinear, _ = get_groupkan_backend(groupkan_backend)

        self.input_proj = GroupKANLinear(
            in_dim,
            hidden_dim,
            act_mode=act_mode,
            drop=dropout,
            num_groups=num_groups,
            device=device,
        )

        self.layers = nn.ModuleList([
            KaaGritLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                k_steps=k_steps,
                num_groups=num_groups,
                edge_num_groups=edge_num_groups,
                dropout=dropout,
                act_mode=act_mode,
                attn_act=attn_act,
                attn_clamp=attn_clamp,
                signed_sqrt=signed_sqrt,
                edge_enhance=edge_enhance,
                ffn_mult=ffn_mult,
                groupkan_backend=groupkan_backend,
                device=device,
            )
            for _ in range(num_layers)
        ])

        self.output_proj = GroupKANLinear(
            hidden_dim,
            out_dim,
            act_mode=act_mode,
            drop=dropout,
            num_groups=num_groups,
            device=device,
        )

    def forward(self, x, edge_index, rrwp):
        h = self.input_proj(x)
        edge_alpha = None

        for layer in self.layers:
            h, edge_alpha = layer(h, edge_index, rrwp)

        z = self.output_proj(h)
        node_alpha = self._edge_to_node_alpha(edge_alpha, edge_index, z.size(0))

        return z, node_alpha, edge_alpha

    def _edge_to_node_alpha(self, edge_alpha, edge_index, num_nodes):
        if edge_alpha is None or edge_alpha.numel() == 0:
            return torch.full((num_nodes,), 1.0 / num_nodes, device=edge_index.device)

        dst = edge_index[1]
        edge_score = edge_alpha.sum(dim=-1)
        node_score = scatter(edge_score, dst, dim=0, dim_size=num_nodes, reduce="sum")
        # 对入度做均值与温度调节，避免 softmax 饱和
        counts = scatter(
            torch.ones_like(edge_score),
            dst,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        ).clamp(min=1)
        node_score = node_score / counts / self.alpha_temperature
        if torch.all(node_score == 0):
            return torch.full((num_nodes,), 1.0 / num_nodes, device=edge_index.device)

        return F.softmax(node_score, dim=0)
