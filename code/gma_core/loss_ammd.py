import torch
import torch.nn as nn


class AttentionAwareMMDLoss(nn.Module):
    """
    Attention-aware Maximum Mean Discrepancy (A-MMD) Loss.

    L = sum_{i,j} a_i a_j k(x_i, x_j) + sum_{i,j} b_i b_j k(y_i, y_j)
        - 2 sum_{i,j} a_i b_j k(x_i, y_j)
    """
    def __init__(
        self,
        sigmas=(1.0, 2.0, 4.0, 8.0, 16.0),
        max_samples=0,
        sample_mode="topk",
    ):
        super().__init__()
        self.sigmas = sigmas
        self.max_samples = int(max_samples)
        self.sample_mode = sample_mode

    def forward(self, z1, z2, alpha1, alpha2, batch1=None, batch2=None):
        if z1.numel() == 0 or z2.numel() == 0:
            return z1.new_tensor(0.0)

        if batch1 is not None and batch2 is not None:
            return self._batched_forward(z1, z2, alpha1, alpha2, batch1, batch2)

        alpha1 = self._normalize_alpha(alpha1, z1)
        alpha2 = self._normalize_alpha(alpha2, z2)
        z1, alpha1 = self._maybe_sample(z1, alpha1)
        z2, alpha2 = self._maybe_sample(z2, alpha2)

        k_xx = self._multi_rbf(z1, z1)
        k_yy = self._multi_rbf(z2, z2)
        k_xy = self._multi_rbf(z1, z2)

        term_xx = (alpha1[:, None] * alpha1[None, :] * k_xx).sum()
        term_yy = (alpha2[:, None] * alpha2[None, :] * k_yy).sum()
        term_xy = (alpha1[:, None] * alpha2[None, :] * k_xy).sum()

        return term_xx + term_yy - 2.0 * term_xy

    def _batched_forward(self, z1, z2, alpha1, alpha2, batch1, batch2):
        num_graphs = int(min(batch1.max().item(), batch2.max().item())) + 1
        total = z1.new_tensor(0.0)
        used = 0
        for gid in range(num_graphs):
            idx1 = (batch1 == gid).nonzero(as_tuple=False).view(-1)
            idx2 = (batch2 == gid).nonzero(as_tuple=False).view(-1)
            if idx1.numel() == 0 or idx2.numel() == 0:
                continue
            z1_g = z1[idx1]
            z2_g = z2[idx2]
            a1_g = self._normalize_alpha(alpha1[idx1] if alpha1 is not None else None, z1_g)
            a2_g = self._normalize_alpha(alpha2[idx2] if alpha2 is not None else None, z2_g)
            z1_g, a1_g = self._maybe_sample(z1_g, a1_g)
            z2_g, a2_g = self._maybe_sample(z2_g, a2_g)
            k_xx = self._multi_rbf(z1_g, z1_g)
            k_yy = self._multi_rbf(z2_g, z2_g)
            k_xy = self._multi_rbf(z1_g, z2_g)
            term_xx = (a1_g[:, None] * a1_g[None, :] * k_xx).sum()
            term_yy = (a2_g[:, None] * a2_g[None, :] * k_yy).sum()
            term_xy = (a1_g[:, None] * a2_g[None, :] * k_xy).sum()
            total = total + (term_xx + term_yy - 2.0 * term_xy)
            used += 1
        if used == 0:
            return z1.new_tensor(0.0)
        return total / used

    def _normalize_alpha(self, alpha, z):
        if alpha is None:
            return torch.full((z.size(0),), 1.0 / z.size(0), device=z.device)
        alpha = alpha.view(-1).to(z.device)
        alpha = torch.clamp(alpha, min=0.0)
        s = alpha.sum()
        if s <= 0:
            return torch.full((z.size(0),), 1.0 / z.size(0), device=z.device)
        return alpha / s

    def _maybe_sample(self, z, alpha):
        if self.max_samples <= 0 or z.size(0) <= self.max_samples:
            return z, alpha
        k = self.max_samples
        if self.sample_mode == "topk":
            idx = torch.topk(alpha, k=k, dim=0).indices
        else:
            probs = alpha / alpha.sum()
            idx = torch.multinomial(probs, num_samples=k, replacement=False)
        return z[idx], alpha[idx]

    def _multi_rbf(self, x, y):
        dist_sq = torch.cdist(x, y, p=2).pow(2)
        kernels = []
        for sigma in self.sigmas:
            gamma = 1.0 / (2.0 * sigma * sigma)
            kernels.append(torch.exp(-gamma * dist_sq))
        return torch.stack(kernels, dim=0).mean(dim=0)
