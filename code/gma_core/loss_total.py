import torch.nn as nn
from .loss_ammd import AttentionAwareMMDLoss
from .loss_recon import GraphReconstructionLoss
from .loss_contrastive import CrossLayerInfoNCELoss
from .loss_role_align import RoleBasedAlignmentLoss, HubPreservingContrastiveLoss


class AlignmentReconstructionLoss(nn.Module):
    """
    Total loss = lambda_align * A-MMD 
               + lambda_recon * (recon_A + recon_B)
               + lambda_contrastive * InfoNCE
               + lambda_role * Role-based Alignment (NEW!)
    
    Role-based Alignment 是核心创新：
    - 跨层高影响力节点对齐（而非同ID对齐）
    - 让hub与hub对齐，peripheral与peripheral对齐
    """
    def __init__(
        self,
        lambda_align=1.0,
        lambda_recon=1.0,
        lambda_contrastive=0.0,
        lambda_role=0.0,
        ammd_loss=None,
        recon_loss=None,
        contrastive_loss=None,
        role_loss=None,
        # Role alignment specific params
        role_temperature=0.1,
        role_alpha_temperature=0.5,
        role_topk_ratio=0.1,
        role_use_hub_contrastive=True,
        role_use_alpha_matching=True,
        role_use_prototype=True,
        role_num_prototypes=3,
    ):
        super().__init__()
        self.lambda_align = float(lambda_align)
        self.lambda_recon = float(lambda_recon)
        self.lambda_contrastive = float(lambda_contrastive)
        self.lambda_role = float(lambda_role)
        self.ammd_loss = ammd_loss if ammd_loss is not None else AttentionAwareMMDLoss()
        self.recon_loss = recon_loss if recon_loss is not None else GraphReconstructionLoss()
        self.contrastive_loss = (
            contrastive_loss if contrastive_loss is not None else CrossLayerInfoNCELoss()
        )
        self.role_loss = role_loss if role_loss is not None else RoleBasedAlignmentLoss(
            temperature=role_temperature,
            alpha_temperature=role_alpha_temperature,
            topk_ratio=role_topk_ratio,
            use_hub_contrastive=role_use_hub_contrastive,
            use_alpha_matching=role_use_alpha_matching,
            use_prototype=role_use_prototype,
            num_prototypes=role_num_prototypes,
        )

    def set_lambdas(self, lambda_align=None, lambda_recon=None, lambda_contrastive=None, lambda_role=None):
        if lambda_align is not None:
            self.lambda_align = float(lambda_align)
        if lambda_recon is not None:
            self.lambda_recon = float(lambda_recon)
        if lambda_contrastive is not None:
            self.lambda_contrastive = float(lambda_contrastive)
        if lambda_role is not None:
            self.lambda_role = float(lambda_role)

    def forward(
        self,
        z_a,
        z_b,
        alpha_a,
        alpha_b,
        edge_index_a,
        edge_index_b,
        batch_a=None,
        batch_b=None,
    ):
        loss_align = self.ammd_loss(
            z_a,
            z_b,
            alpha_a,
            alpha_b,
            batch1=batch_a,
            batch2=batch_b,
        )
        loss_recon_a = self.recon_loss(z_a, edge_index_a, batch=batch_a)
        loss_recon_b = self.recon_loss(z_b, edge_index_b, batch=batch_b)
        loss_recon = loss_recon_a + loss_recon_b

        if self.lambda_contrastive > 0.0:
            loss_contrastive = self.contrastive_loss(
                z_a,
                z_b,
                batch_a=batch_a,
                batch_b=batch_b,
            )
        else:
            loss_contrastive = z_a.new_tensor(0.0)

        # NEW: Role-based alignment loss
        if self.lambda_role > 0.0:
            loss_role, role_metrics = self.role_loss(
                z_a,
                z_b,
                alpha_a,
                alpha_b,
                batch_a=batch_a,
                batch_b=batch_b,
            )
        else:
            loss_role = z_a.new_tensor(0.0)
            role_metrics = {}

        total = (
            self.lambda_align * loss_align
            + self.lambda_recon * loss_recon
            + self.lambda_contrastive * loss_contrastive
            + self.lambda_role * loss_role
        )
        metrics = {
            "loss_align": loss_align,
            "loss_recon": loss_recon,
            "loss_recon_a": loss_recon_a,
            "loss_recon_b": loss_recon_b,
            "loss_contrastive": loss_contrastive,
            "loss_role": loss_role,
            "loss_total": total,
        }
        # 合并role的子损失
        metrics.update(role_metrics)
        return total, metrics
