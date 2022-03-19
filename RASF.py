import torch
import torch.nn as nn
import torch.nn.functional as F


class RASF(nn.Module):
    def __init__(self, resolution=(32,32,32), channel=16, num_local_points=32):
        
        super().__init__()
        # self.field shape: (C, H, W, (D,))
        self.field = nn.Parameter(torch.rand(1, channel, *resolution), requires_grad=True)
        self.k = len(resolution)
        self.num_local_points = num_local_points
        self.register_buffer('zoom_factor', torch.zeros(1))
        self.momentum = 0.1
        
    def batch_samples(self, batch_points):
        """
        Sample RASF feature for a batch of points
        Args:
            batch_points: Tensor of shape (B, num_p, 3), the batch of points to sample features,
        Returns:
            RASF_feature: Tensor of shape (B, C, num_p), the RASF feautre of the input batch points, C is the feature
                          dim of the RASF_featre
        """
        B, num_p, _ = batch_points.shape

        inner = torch.matmul(batch_points, batch_points.transpose(1, 2))
        xx = torch.sum(batch_points**2, dim=2, keepdim=True)
        indices = (xx - 2*inner + xx.transpose(2, 1)).topk(32, dim=-1, largest=False, sorted=False)[1]

        local_points = batch_points.unsqueeze(1).expand(-1,num_p,-1,-1).gather(2, indices.unsqueeze(-1).expand(-1,-1,-1,3)) # B, num_p, num_local_points, 3
        relative_local_points = local_points - batch_points.unsqueeze(2)
        
        if self.training:
            # learn zoom_factor from training samples, place the KNN points into the range of RASF grid
            zoom_factor = torch.max(relative_local_points.min().abs(), relative_local_points.max().abs())
            with torch.no_grad():
                self.zoom_factor = self.momentum * zoom_factor + (1 - self.momentum) * self.zoom_factor
        else:
            # fix zoom_factor in inference to speed up
            zoom_factor = self.zoom_factor

        relative_local_points = relative_local_points / zoom_factor
        out = F.grid_sample(self.field.expand(B,-1,-1,-1,-1), relative_local_points.unsqueeze(1), padding_mode="border").squeeze(2) # B, C, N, L

        return out.max(-1)[0] # B, C, num_p
