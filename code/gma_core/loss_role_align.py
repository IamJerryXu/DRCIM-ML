import torch
import torch.nn as nn
import torch.nn.functional as F


class RoleBasedAlignmentLoss(nn.Module):
    """
    基于功能角色的跨层对齐损失
    
    核心思想：
    - 不是同ID对齐，而是"功能对等"对齐
    - 高影响力节点(高α)应该与高影响力节点对齐
    - α值相近的节点应该有相似的嵌入
    
    包含三个子损失：
    1. Hub Contrastive: Top-K高α节点互为正样本的对比学习
    2. Alpha Distribution Matching: α分布引导的软匹配
    3. Role Prototype Alignment: 角色原型（hub/bridge/peripheral）对齐
    """
    
    def __init__(
        self,
        temperature=0.1,
        alpha_temperature=0.5,
        topk_ratio=0.1,
        min_topk=5,
        max_topk=50,
        use_hub_contrastive=True,
        use_alpha_matching=True,
        use_prototype=True,
        num_prototypes=3,
        hub_weight=1.0,
        alpha_weight=1.0,
        prototype_weight=0.5,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.alpha_temperature = float(alpha_temperature)
        self.topk_ratio = float(topk_ratio)
        self.min_topk = int(min_topk)
        self.max_topk = int(max_topk)
        self.use_hub_contrastive = bool(use_hub_contrastive)
        self.use_alpha_matching = bool(use_alpha_matching)
        self.use_prototype = bool(use_prototype)
        self.num_prototypes = int(num_prototypes)
        self.hub_weight = float(hub_weight)
        self.alpha_weight = float(alpha_weight)
        self.prototype_weight = float(prototype_weight)
    
    def forward(self, z_a, z_b, alpha_a, alpha_b, batch_a=None, batch_b=None):
        """
        Args:
            z_a: Layer A embeddings [N, D]
            z_b: Layer B embeddings [M, D]
            alpha_a: Layer A node importance [N]
            alpha_b: Layer B node importance [M]
        """
        if z_a.numel() == 0 or z_b.numel() == 0:
            return z_a.new_tensor(0.0), {}
        
        # 处理batched情况
        if batch_a is not None and batch_b is not None:
            return self._batched_forward(z_a, z_b, alpha_a, alpha_b, batch_a, batch_b)
        
        # 单图情况
        losses = {}
        total = z_a.new_tensor(0.0)
        
        # 归一化嵌入
        z_a_norm = F.normalize(z_a, p=2, dim=-1)
        z_b_norm = F.normalize(z_b, p=2, dim=-1)
        
        # 归一化alpha
        alpha_a = self._normalize_alpha(alpha_a, z_a.size(0))
        alpha_b = self._normalize_alpha(alpha_b, z_b.size(0))
        
        # 1. Hub Contrastive Loss
        if self.use_hub_contrastive:
            loss_hub = self._hub_contrastive_loss(z_a_norm, z_b_norm, alpha_a, alpha_b)
            losses["loss_hub_contrastive"] = loss_hub
            total = total + self.hub_weight * loss_hub
        
        # 2. Alpha Distribution Matching Loss
        if self.use_alpha_matching:
            loss_alpha = self._alpha_matching_loss(z_a_norm, z_b_norm, alpha_a, alpha_b)
            losses["loss_alpha_matching"] = loss_alpha
            total = total + self.alpha_weight * loss_alpha
        
        # 3. Role Prototype Alignment Loss
        if self.use_prototype:
            loss_proto = self._prototype_alignment_loss(z_a_norm, z_b_norm, alpha_a, alpha_b)
            losses["loss_prototype"] = loss_proto
            total = total + self.prototype_weight * loss_proto
        
        losses["loss_role_total"] = total
        return total, losses
    
    def _batched_forward(self, z_a, z_b, alpha_a, alpha_b, batch_a, batch_b):
        """处理batched数据"""
        num_graphs = int(max(batch_a.max().item(), batch_b.max().item())) + 1
        total = z_a.new_tensor(0.0)
        all_losses = {
            "loss_hub_contrastive": z_a.new_tensor(0.0),
            "loss_alpha_matching": z_a.new_tensor(0.0),
            "loss_prototype": z_a.new_tensor(0.0),
        }
        count = 0
        
        for gid in range(num_graphs):
            mask_a = batch_a == gid
            mask_b = batch_b == gid
            if mask_a.sum() < 2 or mask_b.sum() < 2:
                continue
            
            loss, losses = self.forward(
                z_a[mask_a], z_b[mask_b],
                alpha_a[mask_a], alpha_b[mask_b],
                batch_a=None, batch_b=None,
            )
            total = total + loss
            for k, v in losses.items():
                if k in all_losses:
                    all_losses[k] = all_losses[k] + v
            count += 1
        
        if count == 0:
            all_losses["loss_role_total"] = z_a.new_tensor(0.0)
            return z_a.new_tensor(0.0), all_losses
        
        for k in all_losses:
            all_losses[k] = all_losses[k] / count
        all_losses["loss_role_total"] = total / count
        return total / count, all_losses
    
    def _normalize_alpha(self, alpha, num_nodes):
        """归一化alpha到[0,1]并确保sum=1"""
        if alpha is None:
            return torch.full((num_nodes,), 1.0 / num_nodes, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        alpha = alpha.view(-1)
        if alpha.size(0) < num_nodes:
            alpha = F.pad(alpha, (0, num_nodes - alpha.size(0)), value=0.0)
        alpha = alpha[:num_nodes]
        alpha = torch.clamp(alpha, min=0.0)
        s = alpha.sum()
        if s <= 0:
            return torch.full((num_nodes,), 1.0 / num_nodes, device=alpha.device)
        return alpha / s
    
    def _get_topk(self, num_nodes):
        """计算Top-K数量"""
        k = int(num_nodes * self.topk_ratio)
        return max(self.min_topk, min(self.max_topk, k))
    
    def _hub_contrastive_loss(self, z_a, z_b, alpha_a, alpha_b):
        """
        Hub Contrastive Loss:
        - 两层的Top-K高α节点互为正样本
        - 使用InfoNCE损失
        """
        n_a, n_b = z_a.size(0), z_b.size(0)
        k_a = min(self._get_topk(n_a), n_a)
        k_b = min(self._get_topk(n_b), n_b)
        k = min(k_a, k_b)
        
        if k < 2:
            return z_a.new_tensor(0.0)
        
        # 获取两层的Top-K高影响力节点
        topk_idx_a = torch.topk(alpha_a, k=k).indices
        topk_idx_b = torch.topk(alpha_b, k=k).indices
        
        z_hub_a = z_a[topk_idx_a]  # [K, D]
        z_hub_b = z_b[topk_idx_b]  # [K, D]
        
        # 方法：每个hub节点与对方层的所有hub节点计算相似度
        # 正样本：所有跨层hub节点对
        # 负样本：与非hub节点的相似度
        
        # 简化版：让两组hub节点的质心对齐 + 内部散布一致
        center_a = z_hub_a.mean(dim=0)
        center_b = z_hub_b.mean(dim=0)
        
        # 质心对齐
        loss_center = 1.0 - F.cosine_similarity(center_a.unsqueeze(0), center_b.unsqueeze(0)).squeeze()
        
        # InfoNCE: hub_a中的每个节点，与hub_b中所有节点计算相似度
        # 所有hub_b节点都是"弱正样本"
        sim_matrix = torch.mm(z_hub_a, z_hub_b.t()) / self.temperature  # [K, K]
        
        # 每个hub_a节点，其正样本是所有hub_b节点（均匀权重）
        # 负样本是非hub节点
        # 这里简化：使用soft label，让相似度矩阵趋向均匀分布
        target = torch.ones(k, k, device=z_a.device) / k
        log_softmax_sim = F.log_softmax(sim_matrix, dim=-1)
        loss_infonce = -1.0 * (target * log_softmax_sim).sum() / k
        
        return loss_center + 0.5 * loss_infonce
    
    def _alpha_matching_loss(self, z_a, z_b, alpha_a, alpha_b):
        """
        Alpha Distribution Matching Loss (修复版):
        - 使用分位数匹配：按α排名配对节点
        - 直接约束：rank相近的节点嵌入相似
        """
        n_a, n_b = z_a.size(0), z_b.size(0)
        n_min = min(n_a, n_b)
        
        if n_min < 5:
            return z_a.new_tensor(0.0)
        
        # === 核心修复：按排名配对，而非按α值 ===
        # 将两层节点按α排序
        rank_a = torch.argsort(torch.argsort(alpha_a, descending=True))  # rank: 0=最高α
        rank_b = torch.argsort(torch.argsort(alpha_b, descending=True))
        
        # 归一化排名到[0, 1]
        norm_rank_a = rank_a.float() / (n_a - 1) if n_a > 1 else rank_a.float()
        norm_rank_b = rank_b.float() / (n_b - 1) if n_b > 1 else rank_b.float()
        
        # 采样节点（均匀采样，覆盖各个排名）
        max_nodes = 150
        if n_a > max_nodes:
            # 分层采样：保证高中低排名都有
            idx_a = torch.cat([
                torch.topk(alpha_a, k=max_nodes//3).indices,           # 高α
                torch.randperm(n_a, device=z_a.device)[:max_nodes//3], # 随机
                torch.topk(-alpha_a, k=max_nodes//3).indices           # 低α
            ])
            idx_a = torch.unique(idx_a)[:max_nodes]
        else:
            idx_a = torch.arange(n_a, device=z_a.device)
            
        if n_b > max_nodes:
            idx_b = torch.cat([
                torch.topk(alpha_b, k=max_nodes//3).indices,
                torch.randperm(n_b, device=z_b.device)[:max_nodes//3],
                torch.topk(-alpha_b, k=max_nodes//3).indices
            ])
            idx_b = torch.unique(idx_b)[:max_nodes]
        else:
            idx_b = torch.arange(n_b, device=z_b.device)
        
        z_a_s, z_b_s = z_a[idx_a], z_b[idx_b]
        rank_a_s, rank_b_s = norm_rank_a[idx_a], norm_rank_b[idx_b]
        
        # 计算排名差异矩阵（排名越接近，目标相似度越高）
        rank_diff = torch.abs(rank_a_s.unsqueeze(1) - rank_b_s.unsqueeze(0))  # [N, M]
        
        # 转换为目标相似度：排名差<0.1视为"同类"
        # 使用更尖锐的分布
        target_sim = torch.exp(-rank_diff / 0.15)  # [N, M]
        target_sim = target_sim / target_sim.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        # 计算嵌入相似度
        embed_sim = torch.mm(z_a_s, z_b_s.t()) / self.temperature  # [N, M]
        pred_sim = F.log_softmax(embed_sim, dim=-1)
        
        # KL散度
        loss = F.kl_div(pred_sim, target_sim, reduction='batchmean')
        
        # === 额外约束：直接的排名对齐损失 ===
        # 对每个a节点，找到rank最接近的b节点，拉近它们
        best_match_idx = torch.argmin(rank_diff, dim=1)  # [N]
        z_b_matched = z_b_s[best_match_idx]  # [N, D]
        direct_loss = 1.0 - F.cosine_similarity(z_a_s, z_b_matched, dim=-1).mean()
        
        return loss + 0.5 * direct_loss
    
    def _prototype_alignment_loss(self, z_a, z_b, alpha_a, alpha_b):
        """
        Role Prototype Alignment Loss:
        - 将节点分为K个角色（如hub/bridge/peripheral）
        - 对齐两层对应角色的原型向量
        """
        n_a, n_b = z_a.size(0), z_b.size(0)
        k = self.num_prototypes
        
        if n_a < k or n_b < k:
            return z_a.new_tensor(0.0)
        
        # 基于α将节点分组（等频分箱）
        def get_prototypes(z, alpha, k):
            """计算k个角色原型"""
            n = z.size(0)
            # 按α排序
            sorted_idx = torch.argsort(alpha, descending=True)
            
            # 等分为k组
            prototypes = []
            group_size = n // k
            for i in range(k):
                start = i * group_size
                end = (i + 1) * group_size if i < k - 1 else n
                group_idx = sorted_idx[start:end]
                if len(group_idx) > 0:
                    # 加权平均（用α作为权重）
                    group_z = z[group_idx]
                    group_alpha = alpha[group_idx]
                    weights = group_alpha / group_alpha.sum()
                    prototype = (weights.unsqueeze(1) * group_z).sum(dim=0)
                    prototypes.append(prototype)
            
            return torch.stack(prototypes, dim=0) if prototypes else None  # [K, D]
        
        proto_a = get_prototypes(z_a, alpha_a, k)
        proto_b = get_prototypes(z_b, alpha_b, k)
        
        if proto_a is None or proto_b is None:
            return z_a.new_tensor(0.0)
        
        # 归一化
        proto_a = F.normalize(proto_a, p=2, dim=-1)
        proto_b = F.normalize(proto_b, p=2, dim=-1)
        
        # 对应角色的原型应该对齐（按顺序：最高α组对齐最高α组）
        loss = 1.0 - F.cosine_similarity(proto_a, proto_b, dim=-1).mean()
        
        return loss


class HubPreservingContrastiveLoss(nn.Module):
    """
    Hub-Preserving Contrastive Loss
    
    核心思想：
    - 正样本：α_i^A 和 α_j^B 都高的节点对 (i, j)
    - 负样本：一方高α另一方低α的节点对
    
    让高影响力节点在跨层空间中形成一个"紧凑簇"
    """
    
    def __init__(
        self,
        temperature=0.1,
        hub_threshold=0.7,  # Top 30% 视为hub
        margin=0.5,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.hub_threshold = float(hub_threshold)
        self.margin = float(margin)
    
    def forward(self, z_a, z_b, alpha_a, alpha_b):
        if z_a.numel() == 0 or z_b.numel() == 0:
            return z_a.new_tensor(0.0)
        
        n_a, n_b = z_a.size(0), z_b.size(0)
        
        # 归一化
        z_a = F.normalize(z_a, p=2, dim=-1)
        z_b = F.normalize(z_b, p=2, dim=-1)
        
        # 确定hub阈值
        thresh_a = torch.quantile(alpha_a, self.hub_threshold)
        thresh_b = torch.quantile(alpha_b, self.hub_threshold)
        
        hub_mask_a = alpha_a >= thresh_a  # [N]
        hub_mask_b = alpha_b >= thresh_b  # [M]
        
        # 跨层相似度矩阵
        sim = torch.mm(z_a, z_b.t()) / self.temperature  # [N, M]
        
        # 正样本mask：两边都是hub
        pos_mask = hub_mask_a.unsqueeze(1) & hub_mask_b.unsqueeze(0)  # [N, M]
        
        # 负样本mask：一边是hub，另一边不是
        neg_mask = (hub_mask_a.unsqueeze(1) & ~hub_mask_b.unsqueeze(0)) | \
                   (~hub_mask_a.unsqueeze(1) & hub_mask_b.unsqueeze(0))  # [N, M]
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return z_a.new_tensor(0.0)
        
        # Triplet-style loss: pos_sim > neg_sim + margin
        pos_sim = sim[pos_mask].mean() if pos_mask.any() else z_a.new_tensor(0.0)
        neg_sim = sim[neg_mask].mean() if neg_mask.any() else z_a.new_tensor(0.0)
        
        loss = F.relu(neg_sim - pos_sim + self.margin)
        
        return loss
