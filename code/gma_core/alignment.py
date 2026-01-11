import torch
import torch.nn.functional as F
from .loss_ammd import AttentionAwareMMDLoss


class ManifoldAlignment:
    """
    Trainable alignment wrapper for KAA-GRIT encoders.
    """
    def __init__(self, encoder_a, encoder_b, loss_fn=None):
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b
        self.loss_fn = loss_fn or AttentionAwareMMDLoss()

    def forward(self, data_a, data_b):
        z_a, alpha_a, _ = self.encoder_a(data_a.x, data_a.edge_index, data_a.rrwp)
        z_b, alpha_b, _ = self.encoder_b(data_b.x, data_b.edge_index, data_b.rrwp)
        loss = self.loss_fn(z_a, z_b, alpha_a, alpha_b)
        return z_a, z_b, loss

    @staticmethod
    def compute_similarity_matrix(z_a, z_b):
        z_a_norm = F.normalize(z_a, p=2, dim=1)
        z_b_norm = F.normalize(z_b, p=2, dim=1)
        return torch.mm(z_a_norm, z_b_norm.t()).detach().cpu().numpy()
