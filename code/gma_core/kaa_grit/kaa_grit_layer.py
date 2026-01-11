import math
import torch
import torch.nn as nn
from torch_geometric.utils import softmax
from torch_scatter import scatter
from .groupkan_backend import get_groupkan_backend


class KaaGritLayer(nn.Module):
    """
    Single KAA-GRIT layer with RRWP-modulated attention and GroupKAN projections.
    """
    def __init__(
        self,
        hidden_dim,
        num_heads,
        k_steps,
        num_groups=8,
        edge_num_groups=1,
        dropout=0.1,
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
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if hidden_dim % num_groups != 0 or (hidden_dim * 2) % num_groups != 0:
            raise ValueError("hidden_dim and 2*hidden_dim must be divisible by num_groups")
        if k_steps % edge_num_groups != 0:
            raise ValueError("k_steps must be divisible by edge_num_groups")

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.signed_sqrt = signed_sqrt
        self.edge_enhance = edge_enhance
        self.attn_clamp = abs(attn_clamp) if attn_clamp is not None else None

        # 注意力前归一化，避免 GroupKAN 输出量级过大导致 softmax 饱和
        self.attn_norm = nn.LayerNorm(hidden_dim)
        # 可学习的缩放因子，提升注意力分数数值稳定性
        self.logit_scale = nn.Parameter(torch.zeros(1))

        GroupKANLinear, GroupKAN = get_groupkan_backend(groupkan_backend)

        self.q_proj = GroupKANLinear(
            hidden_dim,
            hidden_dim,
            act_mode=act_mode,
            drop=dropout,
            num_groups=num_groups,
            device=device,
        )
        self.k_proj = GroupKANLinear(
            hidden_dim,
            hidden_dim,
            act_mode=act_mode,
            drop=dropout,
            num_groups=num_groups,
            device=device,
        )
        self.v_proj = GroupKANLinear(
            hidden_dim,
            hidden_dim,
            act_mode=act_mode,
            drop=dropout,
            num_groups=num_groups,
            device=device,
        )

        # RRWP projection to attention heads (多头拓扑感知)
        self.edge_proj = GroupKANLinear(
            k_steps,
            hidden_dim * 2,
            act_mode=act_mode,
            drop=dropout,
            num_groups=edge_num_groups,
            device=device,
        )

        self.out_proj = GroupKANLinear(
            hidden_dim,
            hidden_dim,
            act_mode=act_mode,
            drop=dropout,
            num_groups=num_groups,
            device=device,
        )

        self.ffn = GroupKAN(
            layers_hidden=[hidden_dim, hidden_dim * ffn_mult, hidden_dim],
            act_mode=act_mode,
            drop=dropout,
            num_groups=num_groups,
            device=device,
        )

        self.attn_act = nn.ReLU() if attn_act == "relu" else nn.Identity()
        self.Aw = nn.Parameter(torch.zeros(self.head_dim, self.num_heads))
        nn.init.xavier_normal_(self.Aw)

        if self.edge_enhance:
            self.VeRow = nn.Parameter(torch.zeros(self.head_dim, self.num_heads, self.head_dim))
            nn.init.xavier_normal_(self.VeRow)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, rrwp):
        h_in = x
        src, dst = edge_index
        rrwp = rrwp.to(x.device)

        x_norm = self.attn_norm(x)
        q = self.q_proj(x_norm).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x_norm).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x_norm).view(-1, self.num_heads, self.head_dim)

        q_i = q[dst]
        k_j = k[src]

        score = k_j + q_i
        if rrwp is not None and rrwp.numel() > 0:
            e = self.edge_proj(rrwp).view(-1, self.num_heads, self.head_dim * 2)
            e_w = e[:, :, :self.head_dim]
            e_b = e[:, :, self.head_dim:]
            score = score * e_w
            if self.signed_sqrt:
                score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
            score = score + e_b

        score = self.attn_act(score)

        e_t = score
        score = torch.einsum("ehd,dh->eh", score, self.Aw)
        score = score * self.logit_scale.exp()
        if self.attn_clamp is not None:
            score = torch.clamp(score, min=-self.attn_clamp, max=self.attn_clamp)

        alpha = softmax(score, dst)

        msg = v[src] * alpha.unsqueeze(-1)
        out = scatter(msg, dst, dim=0, dim_size=x.size(0), reduce="sum")

        if self.edge_enhance and rrwp is not None and rrwp.numel() > 0:
            rowV = scatter(e_t * alpha.unsqueeze(-1), dst, dim=0, dim_size=x.size(0), reduce="sum")
            rowV = torch.einsum("nhd,dhc->nhc", rowV, self.VeRow)
            out = out + rowV

        out = out.view(-1, self.hidden_dim)
        out = self.out_proj(out)

        h = self.norm1(h_in + self.dropout(out))
        h = self.norm2(h + self.ffn(h))

        return h, alpha
