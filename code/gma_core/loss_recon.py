import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling


class GraphReconstructionLoss(nn.Module):
    """
    Edge reconstruction loss with negative sampling.

    This approximates BCE(sigmoid(ZZ^T), A) without forming the full NxN matrix.
    """
    def __init__(
        self,
        neg_ratio=1.0,
        logit_scale=1.0,
        use_l2_norm=True,
        two_hop_ratio=0.0,
        deg_norm=True,
        hard_neg_ratio=0.0,
        hard_pool_ratio=1.0,
    ):
        super().__init__()
        if neg_ratio < 0:
            raise ValueError("neg_ratio must be >= 0")
        self.neg_ratio = float(neg_ratio)
        self.logit_scale = float(logit_scale)
        self.use_l2_norm = bool(use_l2_norm)
        self.two_hop_ratio = float(two_hop_ratio)
        self.deg_norm = bool(deg_norm)
        self.hard_neg_ratio = max(0.0, min(1.0, float(hard_neg_ratio)))
        self.hard_pool_ratio = max(1.0, float(hard_pool_ratio))

    def forward(self, z, edge_index, num_nodes=None, batch=None):
        if z.numel() == 0 or edge_index.numel() == 0:
            return z.new_tensor(0.0)

        if self.use_l2_norm:
            z = F.normalize(z, p=2, dim=-1)

        if batch is None:
            if num_nodes is None:
                num_nodes = z.size(0)
            return self._loss_from_z(z, edge_index, num_nodes)

        return self._batched_loss(z, edge_index, batch)

    def _loss_from_z(self, z, edge_index, num_nodes):
        num_pos = edge_index.size(1)
        if num_pos == 0:
            return z.new_tensor(0.0)

        num_neg_total = max(1, int(num_pos * self.neg_ratio))
        num_two_hop = int(num_neg_total * self.two_hop_ratio)
        num_rand_neg = max(0, num_neg_total - num_two_hop)

        neg_edges = []
        if num_rand_neg > 0:
            pool_size = max(1, int(num_rand_neg * self.hard_pool_ratio))
            neg_edge_pool = negative_sampling(
                edge_index=edge_index,
                num_nodes=num_nodes,
                num_neg_samples=pool_size,
            )
            if neg_edge_pool.numel() > 0:
                if self.hard_neg_ratio > 0.0:
                    num_hard = int(num_rand_neg * self.hard_neg_ratio)
                    if num_hard <= 0:
                        neg_edges.append(neg_edge_pool)
                    else:
                        scores = (z[neg_edge_pool[0]] * z[neg_edge_pool[1]]).sum(dim=-1)
                        if scores.numel() > 0:
                            k = min(num_hard, scores.numel())
                            top_idx = torch.topk(scores, k=k, dim=0).indices
                            if num_hard >= num_rand_neg or scores.numel() == k:
                                neg_edge = neg_edge_pool[:, top_idx]
                            else:
                                mask = torch.ones(scores.numel(), dtype=torch.bool, device=scores.device)
                                mask[top_idx] = False
                                remain_idx = mask.nonzero(as_tuple=False).view(-1)
                                if remain_idx.numel() > 0:
                                    perm = torch.randperm(remain_idx.numel(), device=scores.device)
                                    extra = remain_idx[perm[: max(0, num_rand_neg - k)]]
                                    neg_edge = torch.cat(
                                        [neg_edge_pool[:, top_idx], neg_edge_pool[:, extra]],
                                        dim=1,
                                    )
                                else:
                                    neg_edge = neg_edge_pool[:, top_idx]
                            neg_edges.append(neg_edge)
                        else:
                            neg_edges.append(neg_edge_pool)
                else:
                    neg_edges.append(neg_edge_pool)

        if num_two_hop > 0:
            two_hop_edge = self._sample_two_hop_negatives(edge_index, num_nodes, num_two_hop)
            if two_hop_edge.numel() > 0:
                neg_edges.append(two_hop_edge)

        pos_logit = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1) * self.logit_scale
        pos_target = torch.ones_like(pos_logit)
        pos_weight = self._edge_weights(edge_index, num_nodes, pos_logit) if self.deg_norm else None

        if neg_edges:
            neg_edge = torch.cat(neg_edges, dim=1)
            neg_logit = (z[neg_edge[0]] * z[neg_edge[1]]).sum(dim=-1) * self.logit_scale
            neg_target = torch.zeros_like(neg_logit)
            neg_weight = self._edge_weights(neg_edge, num_nodes, neg_logit) if self.deg_norm else None
            logits = torch.cat([pos_logit, neg_logit], dim=0)
            targets = torch.cat([pos_target, neg_target], dim=0)
            if pos_weight is not None and neg_weight is not None:
                weights = torch.cat([pos_weight, neg_weight], dim=0)
            else:
                weights = None
        else:
            logits = pos_logit
            targets = pos_target
            weights = pos_weight

        return F.binary_cross_entropy_with_logits(logits, targets, weight=weights)

    def _edge_weights(self, edge_index, num_nodes, ref_tensor):
        deg = torch.bincount(edge_index[0], minlength=num_nodes).to(ref_tensor.device)
        deg = deg.float() + 1.0
        w = 1.0 / torch.sqrt(deg[edge_index[0]] * deg[edge_index[1]])
        return w

    def _sample_two_hop_negatives(self, edge_index, num_nodes, num_samples):
        src = edge_index[0]
        dst = edge_index[1]
        if src.numel() == 0:
            return edge_index.new_empty((2, 0))

        perm = torch.argsort(src)
        src_sorted = src[perm]
        dst_sorted = dst[perm]
        deg = torch.bincount(src_sorted, minlength=num_nodes)
        rowptr = torch.cat([deg.new_zeros(1), deg.cumsum(0)])

        edge_ids = torch.randint(0, src.size(0), (num_samples,), device=edge_index.device)
        u = src[edge_ids]
        v = dst[edge_ids]
        start = rowptr[v]
        end = rowptr[v + 1]
        span = (end - start).clamp(min=1)
        rand = (torch.rand_like(span.float()) * span.float()).long()
        w = dst_sorted[start + rand]

        mask = u != w
        if mask.sum().item() == 0:
            return edge_index.new_empty((2, 0))
        return torch.stack([u[mask], w[mask]], dim=0)

    def _batched_loss(self, z, edge_index, batch):
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        if num_graphs <= 0:
            return z.new_tensor(0.0)

        total = z.new_tensor(0.0)
        src = edge_index[0]
        dst = edge_index[1]

        for gid in range(num_graphs):
            node_mask = batch == gid
            node_idx = node_mask.nonzero(as_tuple=False).view(-1)
            if node_idx.numel() < 2:
                continue

            edge_mask = (batch[src] == gid) & (batch[dst] == gid)
            if edge_mask.sum().item() == 0:
                continue

            local_map = torch.full(
                (z.size(0),),
                -1,
                device=z.device,
                dtype=torch.long,
            )
            local_map[node_idx] = torch.arange(node_idx.size(0), device=z.device)
            local_edge = local_map[edge_index[:, edge_mask]]
            local_z = z[node_idx]
            total = total + self._loss_from_z(local_z, local_edge, node_idx.size(0))

        return total / max(1, num_graphs)
