"""Modules to compute the IoU matching cost and solve the corresponding LSAP."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network."""

    @torch.no_grad()
    def forward(self, mask_preds, mask_targets):
        """Performs the matching.

        Params:
            mask_preds: Tensor of dim [batch_size, n_objects, N, N] with the predicted masks
            mask_targets: Tensor of dim [batch_size, n_objects, N, N]
            with the target masks from another augmentation

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions
                - index_j is the indices of the corresponding selected targets
        """
        bs, n_objects, _, _ = mask_preds.shape
        # Compute the iou cost betwen masks
        cost_iou = -get_iou_matrix(mask_preds, mask_targets)
        cost_iou = cost_iou.reshape(bs, n_objects, bs, n_objects).cpu()
        self.costs = torch.stack([cost_iou[i, :, i, :][None] for i in range(bs)])
        indices = [linear_sum_assignment(c[0]) for c in self.costs]
        return torch.as_tensor(np.array(indices))


def get_iou_matrix(preds, targets):

    bs, n_objects, H, W = targets.shape
    targets = targets.reshape(bs * n_objects, H * W).float()
    preds = preds.reshape(bs * n_objects, H * W).float()

    intersection = torch.matmul(targets, preds.t())
    targets_area = targets.sum(dim=1).view(1, -1)
    preds_area = preds.sum(dim=1).view(1, -1)
    union = (targets_area.t() + preds_area) - intersection

    return torch.where(
        union == 0,
        torch.tensor(0.0, device=targets.device),
        intersection / union,
    )
