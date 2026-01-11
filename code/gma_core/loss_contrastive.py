import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        if in_dim <= 0 or hidden_dim <= 0 or out_dim <= 0:
            raise ValueError("Projection dimensions must be positive.")
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class CrossLayerInfoNCELoss(nn.Module):
    """
    Cross-layer InfoNCE with identity positives (node-aligned).
    """

    def __init__(
        self,
        temperature=0.2,
        max_samples=0,
        use_projection=False,
        in_dim=0,
        proj_hidden_dim=0,
        proj_out_dim=0,
        proj_dropout=0.0,
        pos_mode="id",
        topk=1,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.max_samples = int(max_samples)
        self.use_projection = bool(use_projection)
        self.pos_mode = str(pos_mode)
        self.topk = max(1, int(topk))
        self.projector = None
        if self.use_projection:
            if in_dim <= 0:
                raise ValueError("in_dim must be positive when using projection head.")
            if proj_hidden_dim <= 0:
                proj_hidden_dim = in_dim
            if proj_out_dim <= 0:
                proj_out_dim = in_dim
            self.projector = ProjectionHead(
                in_dim=in_dim,
                hidden_dim=proj_hidden_dim,
                out_dim=proj_out_dim,
                dropout=proj_dropout,
            )

    def forward(self, z_a, z_b, batch_a=None, batch_b=None):
        if z_a.numel() == 0 or z_b.numel() == 0:
            return z_a.new_tensor(0.0)

        if self.projector is not None:
            z_a = self.projector(z_a)
            z_b = self.projector(z_b)

        z_a = F.normalize(z_a, p=2, dim=-1)
        z_b = F.normalize(z_b, p=2, dim=-1)

        if batch_a is None or batch_b is None:
            return self._loss_from_z(z_a, z_b)

        num_graphs = int(batch_a.max().item()) + 1 if batch_a.numel() > 0 else 0
        if num_graphs <= 0:
            return z_a.new_tensor(0.0)

        total = z_a.new_tensor(0.0)
        count = 0
        for gid in range(num_graphs):
            mask_a = batch_a == gid
            mask_b = batch_b == gid
            if mask_a.sum().item() == 0 or mask_b.sum().item() == 0:
                continue
            za_g = z_a[mask_a]
            zb_g = z_b[mask_b]
            n = min(za_g.size(0), zb_g.size(0))
            if n < 2:
                continue
            total = total + self._loss_from_z(za_g[:n], zb_g[:n])
            count += 1
        if count == 0:
            return z_a.new_tensor(0.0)
        return total / count

    def _loss_from_z(self, z_a, z_b):
        n = min(z_a.size(0), z_b.size(0))
        if n < 2:
            return z_a.new_tensor(0.0)

        if self.max_samples > 0 and n > self.max_samples:
            idx = torch.randperm(n, device=z_a.device)[: self.max_samples]
            z_a = z_a[idx]
            z_b = z_b[idx]
            n = z_a.size(0)

        logits = torch.matmul(z_a, z_b.t()) / max(1e-6, self.temperature)
        if self.pos_mode == "id":
            labels = torch.arange(n, device=z_a.device)
        elif self.pos_mode == "knn":
            with torch.no_grad():
                sim = logits.detach()
                k = min(self.topk, sim.size(1))
                topk_idx = torch.topk(sim, k=k, dim=1).indices
                if k == 1:
                    labels = topk_idx.squeeze(1)
                else:
                    choice = torch.randint(0, k, (n,), device=z_a.device)
                    labels = topk_idx[torch.arange(n, device=z_a.device), choice]
        else:
            raise ValueError(f"Unsupported pos_mode: {self.pos_mode}")

        loss_a2b = F.cross_entropy(logits, labels)
        logits_t = logits.t()
        if self.pos_mode == "id":
            labels_t = torch.arange(n, device=z_a.device)
        elif self.pos_mode == "knn":
            with torch.no_grad():
                sim_t = logits_t.detach()
                k = min(self.topk, sim_t.size(1))
                topk_idx_t = torch.topk(sim_t, k=k, dim=1).indices
                if k == 1:
                    labels_t = topk_idx_t.squeeze(1)
                else:
                    choice_t = torch.randint(0, k, (n,), device=z_a.device)
                    labels_t = topk_idx_t[torch.arange(n, device=z_a.device), choice_t]
        else:
            raise ValueError(f"Unsupported pos_mode: {self.pos_mode}")

        loss_b2a = F.cross_entropy(logits_t, labels_t)
        return 0.5 * (loss_a2b + loss_b2a)
