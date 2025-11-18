import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.eps = 1e-6

    def forward(self, features, labels=None):
        """
        features: [B, n_views, dim]
        labels: [B]
        """
        device = features.device
        B, n_views, dim = features.shape

        # normalize along feature dim
        features = F.normalize(features, dim=2)
        features = features.view(B*n_views, dim)  # [B*n_views, dim]

        labels = labels.view(-1,1)  # [B,1]
        mask = torch.eq(labels, labels.T).float().to(device)  # [B,B]
        mask = mask.repeat(n_views, n_views)  # [B*n_views, B*n_views]


        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        # subtract max for numerical stability
        logits_max,_ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask out self-contrast
        logits_mask = torch.ones_like(mask) - torch.eye(B*n_views, device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)+self.eps)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+self.eps)
        loss = -mean_log_prob_pos.mean()
        return loss

    

class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        loss = self.criterion(logits, labels)
        return loss