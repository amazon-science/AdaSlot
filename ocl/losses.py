from functools import partial
from math import log
from typing import Callable, List, Optional, Union

import pytorch_lightning as pl
import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.ops import generalized_box_iou

from ocl import base, consistency, path_defaults, scheduling
from ocl.base import Instances
from ocl.matching import CPUHungarianMatcher
from ocl.utils.bboxes import box_cxcywh_to_xyxy
from ocl.utils.routing import RoutableMixin


def _constant_weight(weight: float, global_step: int):
    return weight


class ReconstructionLoss(nn.Module, RoutableMixin):
    def __init__(
        self,
        loss_type: str,
        weight: Union[Callable, float] = 1.0,
        normalize_target: bool = False,
        input_path: Optional[str] = None,
        target_path: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self,
            {"input": input_path, "target": target_path, "global_step": path_defaults.GLOBAL_STEP},
        )
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "mse_sum":
            # Used for slot_attention and video slot attention.
            self.loss_fn = (
                lambda x1, x2: nn.functional.mse_loss(x1, x2, reduction="sum") / x1.shape[0]
            )
        elif loss_type == "l1":
            self.loss_name = "l1_loss"
            self.loss_fn = nn.functional.l1_loss
        elif loss_type == "cosine":
            self.loss_name = "cosine_loss"
            self.loss_fn = lambda x1, x2: -nn.functional.cosine_similarity(x1, x2, dim=-1).mean()
        elif loss_type == "cross_entropy_sum":
            # Used for SLATE, average is over the first (batch) dim only.
            self.loss_name = "cross_entropy_sum_loss"
            self.loss_fn = (
                lambda x1, x2: nn.functional.cross_entropy(
                    x1.reshape(-1, x1.shape[-1]), x2.reshape(-1, x2.shape[-1]), reduction="sum"
                )
                / x1.shape[0]
            )
        else:
            raise ValueError(
                f"Unknown loss {loss_type}. Valid choices are (mse, l1, cosine, cross_entropy)."
            )
        # If weight is callable use it to determine scheduling otherwise use constant value.
        self.weight = weight if callable(weight) else partial(_constant_weight, weight)
        self.normalize_target = normalize_target

    @RoutableMixin.route
    def forward(self, input: torch.Tensor, target: torch.Tensor, global_step: int):
        target = target.detach()
        if self.normalize_target:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = self.loss_fn(input, target)
        weight = self.weight(global_step)
        return weight * loss

class SparsePenalty(nn.Module, RoutableMixin):
    def __init__(
        self,
        linear_weight: Union[Callable, float] = 1.0,
        quadratic_weight: Union[Callable, float] = 0.0,
        quadratic_bias: Union[Callable, float] = 0.5,
        input_path: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self,
            {"input": input_path, "global_step": path_defaults.GLOBAL_STEP},
        )

        self.linear_weight = linear_weight if callable(linear_weight) else partial(_constant_weight, linear_weight)
        self.quadratic_weight = quadratic_weight if callable(quadratic_weight) else partial(_constant_weight, quadratic_weight)
        self.quadratic_bias = quadratic_bias if callable(quadratic_bias) else partial(_constant_weight, quadratic_bias)

    @RoutableMixin.route
    def forward(self, input: torch.Tensor, global_step: int):
        # print("spaese_input.shape")
        # print(input.shape)
        sparse_degree = torch.mean(input)

        linear_weight = self.linear_weight(global_step)
        quadratic_weight = self.quadratic_weight(global_step)
        quadratic_bias = self.quadratic_bias(global_step)

        return linear_weight * sparse_degree + quadratic_weight * (sparse_degree - quadratic_bias) ** 2
        #return linear_weight * sparse_degree + quadratic_weight * torch.mean((input - quadratic_bias) ** 2)



class LatentDupplicateSuppressionLoss(nn.Module, RoutableMixin):
    def __init__(
        self,
        weight: Union[float, scheduling.HPSchedulerT],
        eps: float = 1e-08,
        grouping_path: Optional[str] = "perceptual_grouping",
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self, {"grouping": grouping_path, "global_step": path_defaults.GLOBAL_STEP}
        )
        self.weight = weight
        self.similarity = nn.CosineSimilarity(dim=-1, eps=eps)

    @RoutableMixin.route
    def forward(self, grouping: base.PerceptualGroupingOutput, global_step: int):
        if grouping.objects.dim() == 4:
            # Build large tensor of reconstructed video.
            objects = grouping.objects
            bs, n_frames, n_objects, n_features = objects.shape

            off_diag_indices = torch.triu_indices(
                n_objects, n_objects, offset=1, device=objects.device
            )

            sq_similarities = (
                self.similarity(
                    objects[:, :, off_diag_indices[0], :], objects[:, :, off_diag_indices[1], :]
                )
                ** 2
            )

            if grouping.is_empty is not None:
                p_not_empty = 1.0 - grouping.is_empty
                # Assume that the probability of of individual objects being present is independent,
                # thus the probability of both being present is the product of the individual
                # probabilities.
                p_pair_present = (
                    p_not_empty[..., off_diag_indices[0]] * p_not_empty[..., off_diag_indices[1]]
                )
                # Use average expected penalty as loss for each frame.
                losses = (sq_similarities * p_pair_present) / torch.sum(
                    p_pair_present, dim=-1, keepdim=True
                )
            else:
                losses = sq_similarities.mean(dim=-1)

            weight = self.weight(global_step) if callable(self.weight) else self.weight
            return weight * losses.sum() / (bs * n_frames)
        elif grouping.objects.dim() == 3:
            # Build large tensor of reconstructed image.
            objects = grouping.objects
            bs, n_objects, n_features = objects.shape

            off_diag_indices = torch.triu_indices(
                n_objects, n_objects, offset=1, device=objects.device
            )

            sq_similarities = (
                self.similarity(
                    objects[:, off_diag_indices[0], :], objects[:, off_diag_indices[1], :]
                )
                ** 2
            )

            if grouping.is_empty is not None:
                p_not_empty = 1.0 - grouping.is_empty
                # Assume that the probability of of individual objects being present is independent,
                # thus the probability of both being present is the product of the individual
                # probabilities.
                p_pair_present = (
                    p_not_empty[..., off_diag_indices[0]] * p_not_empty[..., off_diag_indices[1]]
                )
                # Use average expected penalty as loss for each frame.
                losses = (sq_similarities * p_pair_present) / torch.sum(
                    p_pair_present, dim=-1, keepdim=True
                )
            else:
                losses = sq_similarities.mean(dim=-1)

            weight = self.weight(global_step) if callable(self.weight) else self.weight
            return weight * losses.sum() / bs
        else:
            raise ValueError("Incompatible input format.")


class ConsistencyLoss(nn.Module, RoutableMixin):
    """Task that returns the previously extracted objects.

    Intended to make the object representations accessible to downstream functions, e.g. metrics.
    """

    def __init__(
        self,
        matcher: consistency.HungarianMatcher,
        loss_type: str = "CE",
        loss_weight: float = 0.25,
        mask_path: Optional[str] = None,
        mask_target_path: Optional[str] = None,
        params_path: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self,
            {
                "mask": mask_path,
                "mask_target": mask_target_path,
                "cropping_params": params_path,
                "global_step": path_defaults.GLOBAL_STEP,
            },
        )
        self.matcher = matcher
        if loss_type == "CE":
            self.loss_name = "masks_consistency_CE"
            self.weight = (
                loss_weight if callable(loss_weight) else partial(_constant_weight, loss_weight)
            )
            self.loss_fn = nn.CrossEntropyLoss()

    @RoutableMixin.route
    def forward(
        self,
        mask: torch.Tensor,
        mask_target: torch.Tensor,
        cropping_params: torch.Tensor,
        global_step: int,
    ):
        _, n_objects, size, _ = mask.shape
        mask_one_hot = self._to_binary_mask(mask)
        mask_target = self.crop_views(mask_target, cropping_params, size)
        mask_target_one_hot = self._to_binary_mask(mask_target)
        match = self.matcher(mask_one_hot, mask_target_one_hot)
        matched_mask = torch.stack([mask[match[i, 1]] for i, mask in enumerate(mask)])
        assert matched_mask.shape == mask.shape
        assert mask_target.shape == mask.shape
        flattened_matched_mask = matched_mask.permute(0, 2, 3, 1).reshape(-1, n_objects)
        flattened_mask_target = mask_target.permute(0, 2, 3, 1).reshape(-1, n_objects)
        weight = self.weight(global_step) if callable(self.weight) else self.weight
        return weight * self.loss_fn(flattened_matched_mask, flattened_mask_target)

    @staticmethod
    def _to_binary_mask(masks: torch.Tensor):
        _, n_objects, _, _ = masks.shape
        m_lables = masks.argmax(dim=1)
        mask_one_hot = torch.nn.functional.one_hot(m_lables, n_objects)
        return mask_one_hot.permute(0, 3, 1, 2)

    def crop_views(self, view: torch.Tensor, param: torch.Tensor, size: int):
        return torch.cat([self.crop_maping(v, p, size) for v, p in zip(view, param)])

    @staticmethod
    def crop_maping(view: torch.Tensor, p: torch.Tensor, size: int):
        p = tuple(p.cpu().numpy().astype(int))
        return transforms.functional.resized_crop(view, *p, size=(size, size))[None]


def focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, mean_in_dim1=True):
    """Loss used in RetinaNet for dense detection. # noqa: D411.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if mean_in_dim1:
        return loss.mean(1).sum() / num_boxes
    else:
        return loss.sum() / num_boxes


def CompDETRCostMatrix(
    outputs,
    targets,
    use_focal=True,
    class_weight: float = 1,
    bbox_weight: float = 1,
    giou_weight: float = 1,
):
    """Compute cost matrix between outputs instances and target instances.

    Params:
        outputs: This is a dict that contains at least these entries:
                "pred_logits": Tensor of dim [batch_size, num_queries, num_classes]
                            with the classification logits
                "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the
                            predicted box coordinates

        targets: a list of targets (len(targets) = batch_size), where each target is a instance:
                "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                        ground-truth objects in the target) containing the class labels
                "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

    Returns:
        costMatrix: A iter of tensors of size [num_outputs, num_targets].
    """
    with torch.no_grad():
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if use_focal:
            out_prob = outputs["pred_logits"].flatten(0, 1)
        else:
            AssertionError("only support focal for now.")
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        if use_focal:
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        else:
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(  # noqa: F821
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)  # noqa: F821
        )

        # Final cost matrix
        C = bbox_weight * cost_bbox + class_weight * cost_class + giou_weight * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]

        return C.split(sizes, -1)


class MOTRLoss(nn.Module, RoutableMixin):
    def __init__(
        self,
        num_classes: int = 1,
        loss_weight: float = 1.0,
        input_bbox_path: Optional[str] = None,
        target_bbox_path: Optional[str] = None,
        input_cls_path: Optional[str] = None,
        target_cls_path: Optional[str] = None,
        target_id_path: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        RoutableMixin.__init__(
            self,
            {
                "input_bbox": input_bbox_path,
                "target_bbox": target_bbox_path,
                "input_cls": input_cls_path,
                "target_cls": target_cls_path,
                "target_id": target_id_path,
                "global_step": path_defaults.GLOBAL_STEP,
            },
        )
        self.matcher = CPUHungarianMatcher()

    def bbox_loss(self, input_bbox, target_bbox):
        return F.l1_loss(input_bbox, target_bbox, reduction="mean")

    def objectness_loss(self, input_objectness, target_objectness):
        batch_size, num_objects = target_objectness.shape
        return F.cross_entropy(
            input_objectness.view(batch_size * num_objects, -1),
            target_objectness.view(batch_size * num_objects),
        )

    def clip_target_to_instances(self, clip_target_cls, clip_target_bbox, clip_target_id):
        """Converting the target in one clip into instances.

        Args:
            clip_target_cls (_type_): object class, -1 is background.
            clip_target_bbox (_type_): object bounding box.
            clip_target_id (_type_): object id.
        """
        num_frames = clip_target_bbox.shape[0]
        clip_gt_instances = []
        for fidx in range(num_frames):
            frame_gt_instances = base.Instances((1, 1))
            frame_gt_cls = clip_target_cls[fidx]
            non_empty_mask_idx = frame_gt_cls > -1
            if non_empty_mask_idx.sum() > 0:
                frame_gt_instances.boxes = clip_target_bbox[fidx][non_empty_mask_idx]
                frame_gt_instances.labels = frame_gt_cls[non_empty_mask_idx]
                frame_gt_instances.obj_ids = clip_target_id[fidx][non_empty_mask_idx]
                clip_gt_instances.append(frame_gt_instances)
        return clip_gt_instances

    def _generate_empty_tracks(self, num_queries, device):
        track_instances = Instances((1, 1))

        # At init, the number of track_instances is the same as slot number
        track_instances.obj_idxes = torch.full((num_queries,), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full(
            (num_queries,), -1, dtype=torch.long, device=device
        )
        track_instances.disappear_time = torch.zeros((num_queries,), dtype=torch.long, device=device)
        track_instances.iou = torch.zeros((num_queries,), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((num_queries,), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros((num_queries,), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((num_queries, 4), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros(
            (num_queries, self.num_classes), dtype=torch.float, device=device
        )

        return track_instances.to(device)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_boxes(self, outputs, gt_instances: List[Instances], indices: List[tuple], num_boxes):
        """Compute the losses related to the bounding boxes.

        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w),
        normalized by the image size.
        """
        # We ignore the regression loss of the track-disappear slots.
        # TODO: Make this filter process more elegant.
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0
        )

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat(
            [gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0
        )  # size(16)
        mask = target_obj_ids != -1

        # only use l1 loss for now, will consider giou loss later
        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction="none")
        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes[mask]), box_cxcywh_to_xyxy(target_boxes[mask])
            )
        )

        # Note, we will not normalize by the num_boxes here. Will handle later.
        return loss_bbox.sum() + loss_giou.sum()

    def loss_labels(self, outputs, gt_instances: List[Instances], indices, num_boxes, log=False):
        """Classification loss (NLL).

        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        # The matched gt for disappear track query is set -1.
        labels = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            labels_per_img = torch.ones_like(J)
            # set labels of track-appear slots to 0.
            if len(gt_per_img) > 0:
                labels_per_img[J != -1] = gt_per_img.labels[J[J != -1]]
            labels.append(labels_per_img)
        target_classes_o = torch.cat(labels)
        target_classes[idx] = target_classes_o
        # we use focal loss for each class
        gt_labels_target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[
            :, :, :-1
        ]  # no loss for the last (background) class
        gt_labels_target = gt_labels_target.to(src_logits)
        loss_ce = focal_loss(
            src_logits.flatten(1),
            gt_labels_target.flatten(1),
            alpha=0.25,
            gamma=2,
            num_boxes=num_boxes,
            mean_in_dim1=False,
        )
        loss_ce = loss_ce.sum()
        return loss_ce

    @RoutableMixin.route
    def forward(
        self,
        input_bbox: torch.Tensor,
        target_bbox: torch.Tensor,
        input_cls: torch.Tensor,
        target_cls: torch.Tensor,
        target_id: torch.Tensor,
        global_step: int,
    ):
        target_bbox = target_bbox.detach()
        target_cls = target_cls.detach()
        target_id = target_id.detach()

        batch_size, num_frames, num_queries, _ = input_bbox.shape
        device = input_bbox.device

        total_loss = 0
        num_samples = 0

        # Iterate through each clip. Might think about if parallelable
        for cidx in range(batch_size):
            clip_target_instances = self.clip_target_to_instances(
                target_cls[cidx], target_bbox[cidx], target_id[cidx]
            )
            # Init empty prediction tracks
            track_instances = self._generate_empty_tracks(num_queries, device)
            for fidx in range(num_frames):
                gt_instances_i = clip_target_instances[fidx]
                # put the prediction at the current frame into track instances.
                track_scores = input_cls[cidx, fidx].max(dim=-1).values
                track_instances.scores = track_scores
                pred_logits_i = input_cls[cidx, fidx]
                pred_boxes_i = input_bbox[cidx, fidx]
                outputs_i = {
                    "pred_logits": pred_logits_i.unsqueeze(0),
                    "pred_boxes": pred_boxes_i.unsqueeze(0),
                }
                track_instances.pred_logits = pred_logits_i
                track_instances.pred_boxes = pred_boxes_i

                # step 0: collect existing matched pairs
                obj_idxes = gt_instances_i.obj_ids
                obj_idxes_list = obj_idxes.detach().cpu().numpy().tolist()
                obj_idx_to_gt_idx = {
                    obj_idx: gt_idx for gt_idx, obj_idx in enumerate(obj_idxes_list)
                }

                # step1. inherit and update the previous tracks.
                num_disappear_track = 0
                for j in range(len(track_instances)):
                    obj_id = track_instances.obj_idxes[j].item()
                    # set new target idx.
                    if obj_id >= 0:
                        if obj_id in obj_idx_to_gt_idx:
                            track_instances.matched_gt_idxes[j] = obj_idx_to_gt_idx[obj_id]
                        else:
                            num_disappear_track += 1
                            track_instances.matched_gt_idxes[j] = -1  # track-disappear case.
                    else:
                        track_instances.matched_gt_idxes[j] = -1

                full_track_idxes = torch.arange(len(track_instances), dtype=torch.long).to(
                    input_bbox.device
                )
                matched_track_idxes = track_instances.obj_idxes >= 0  # occu
                prev_matched_indices = torch.stack(
                    [
                        full_track_idxes[matched_track_idxes],
                        track_instances.matched_gt_idxes[matched_track_idxes],
                    ],
                    dim=1,
                ).to(input_bbox.device)

                # step2. select the unmatched slots.
                # note that the FP tracks whose obj_idxes are -2 will not be selected here.
                unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]

                # step3. select the untracked gt instances (new tracks).
                tgt_indexes = track_instances.matched_gt_idxes
                tgt_indexes = tgt_indexes[tgt_indexes != -1]
                tgt_state = torch.zeros(len(gt_instances_i)).to(pred_logits_i.device)
                tgt_state[tgt_indexes] = 1
                untracked_tgt_indexes = torch.arange(len(gt_instances_i)).to(pred_logits_i.device)[
                    tgt_state == 0
                ]
                untracked_gt_instances = {
                    "labels": gt_instances_i[untracked_tgt_indexes].labels,
                    "boxes": gt_instances_i[untracked_tgt_indexes].boxes,
                }

                def match_for_single_decoder_layer(
                    unmatched_outputs,
                    matcher,
                    untracked_gt_instances,
                    unmatched_track_idxes,
                    untracked_tgt_indexes,
                    device,
                ):
                    costMatrix = CompDETRCostMatrix(
                        unmatched_outputs, [untracked_gt_instances]
                    )  # list[tuple(src_idx, tgt_idx)]

                    new_track_indices = []
                    for c in costMatrix:
                        AssignmentMatrix, _ = self.matcher(c)
                        assert AssignmentMatrix.shape[0] == 1, "Only match for one frame."
                        new_track_indices.append(torch.where(AssignmentMatrix[0] > 0))

                    assert len(new_track_indices) == 1, "Only match for one frame."
                    src_idx = new_track_indices[0][0]
                    tgt_idx = new_track_indices[0][1]
                    # concat src and tgt.
                    res_new_matched_indices = torch.stack(
                        [unmatched_track_idxes[src_idx], untracked_tgt_indexes[tgt_idx]], dim=1
                    ).to(device)
                    return res_new_matched_indices

                # step4. do matching between the unmatched slots and GTs.
                unmatched_outputs = {
                    "pred_logits": track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
                    "pred_boxes": track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(0),
                }
                if unmatched_outputs["pred_logits"].shape[1] == 0:
                    # NOTE, this is a hack when try to use random_strided_window
                    # Figure out how it really works.
                    new_matched_indices = (
                        torch.zeros([0, 2]).long().to(track_instances.pred_logits.device)
                    )
                else:
                    new_matched_indices = match_for_single_decoder_layer(
                        unmatched_outputs,
                        self.matcher,
                        untracked_gt_instances,
                        unmatched_track_idxes,
                        untracked_tgt_indexes,
                        pred_logits_i.device,
                    )

                # step5. update obj_idxes according to the new matching result.
                track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances_i.obj_ids[
                    new_matched_indices[:, 1]
                ].long()
                track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[
                    :, 1
                ]

                # step6. merge the new pairs and the matched pairs.
                matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)

                # step7. calculate losses.
                num_samples += len(gt_instances_i) + num_disappear_track
                cls_loss = self.loss_labels(
                    outputs=outputs_i,
                    gt_instances=[gt_instances_i],
                    indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                    num_boxes=1,
                )
                bbox_loss = self.loss_boxes(
                    outputs=outputs_i,
                    gt_instances=[gt_instances_i],
                    indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                    num_boxes=1,
                )
                total_loss += cls_loss + bbox_loss
        # A naive normalization, might not be the best
        total_loss /= num_samples
        total_loss *= self.loss_weight
        return total_loss


class CLIPLoss(nn.Module, RoutableMixin):
    def __init__(
        self,
        normalize_inputs: bool = True,
        learn_scale: bool = True,
        max_temperature: Optional[float] = None,
        first_path: Optional[str] = None,
        second_path: Optional[str] = None,
        model_path: Optional[str] = path_defaults.MODEL,
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self, {"first_rep": first_path, "second_rep": second_path, "model": model_path}
        )
        self.normalize_inputs = normalize_inputs
        if learn_scale:
            self.logit_scale = nn.Parameter(torch.zeros([]) * log(1 / 0.07))  # Same init as CLIP.
        else:
            self.register_buffer("logit_scale", torch.zeros([]))  # exp(0) = 1, i.e. no scaling.
        self.max_temperature = max_temperature

    @RoutableMixin.route
    def forward(
        self,
        first_rep: torch.Tensor,
        second_rep: torch.Tensor,
        model: Optional[pl.LightningModule] = None,
    ):
        # Collect all representations.
        if self.normalize_inputs:
            first_rep = first_rep / first_rep.norm(dim=-1, keepdim=True)
            second_rep = second_rep / second_rep.norm(dim=-1, keepdim=True)

        temperature = self.logit_scale.exp()
        if self.max_temperature:
            temperature = torch.clamp_max(temperature, self.max_temperature)

        if model is not None and hasattr(model, "trainer") and model.trainer.world_size > 1:
            # Running on multiple GPUs.
            global_rank = model.global_rank
            all_first_rep, all_second_rep = model.all_gather(
                [first_rep, second_rep], sync_grads=True
            )
            world_size, batch_size = all_first_rep.shape[:2]
            labels = (
                torch.arange(batch_size, dtype=torch.long, device=first_rep.device)
                + batch_size * global_rank
            )
            # Flatten the GPU dim into batch.
            all_first_rep = all_first_rep.flatten(0, 1)
            all_second_rep = all_second_rep.flatten(0, 1)

            # Compute inner product for instances on the current GPU.
            logits_per_first = temperature * first_rep @ all_second_rep.t()
            logits_per_second = temperature * second_rep @ all_first_rep.t()

            # For visualization purposes, return the cosine similarities on the local batch.
            similarities = (
                1
                / temperature
                * logits_per_first[:, batch_size * global_rank : batch_size * (global_rank + 1)]
            )
            # shape = [local_batch_size, global_batch_size]
        else:
            batch_size = first_rep.shape[0]
            labels = torch.arange(batch_size, dtype=torch.long, device=first_rep.device)
            # When running with only a single GPU we can save some compute time by reusing
            # computations.
            logits_per_first = temperature * first_rep @ second_rep.t()
            logits_per_second = logits_per_first.t()
            similarities = 1 / temperature * logits_per_first

        return (
            (F.cross_entropy(logits_per_first, labels) + F.cross_entropy(logits_per_second, labels))
            / 2,
            {"similarities": similarities, "temperature": temperature},
        )


def CompDETRSegCostMatrix(
    predicts,
    targets,
):
    """Compute cost matrix between outputs instances and target instances.

    Returns:
        costMatrix: A iter of tensors of size [num_outputs, num_targets].
    """
    # filter out valid targets
    npr, h, w = predicts.shape
    nt = targets.shape[0]

    predicts = repeat(predicts, "npr h w -> (npr repeat) h w", repeat=nt)
    targets = repeat(targets, "nt h w -> (repeat nt) h w", repeat=npr)

    cost = F.binary_cross_entropy(predicts, targets.float(), reduction="none").mean(-1).mean(-1)
    cost = rearrange(cost, "(npr nt) -> npr nt", npr=npr, nt=nt)
    return cost


class DETRSegLoss(nn.Module, RoutableMixin):
    def __init__(
        self,
        loss_weight: float = 1.0,
        ignore_background: bool = True,
        foreground_weight: float = 1.0,
        foreground_matching_weight: float = 1.0,
        global_loss: bool = True,
        input_mask_path: Optional[str] = None,
        target_mask_path: Optional[str] = None,
        foreground_logits_path: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self,
            {
                "input_mask": input_mask_path,
                "target_mask": target_mask_path,
                "foreground_logits": foreground_logits_path,
                "model": path_defaults.MODEL,
            },
        )
        self.loss_weight = loss_weight
        self.ignore_background = ignore_background
        self.foreground_weight = foreground_weight
        self.foreground_matching_weight = foreground_matching_weight
        self.global_loss = global_loss
        self.matcher = CPUHungarianMatcher()

    @RoutableMixin.route
    def forward(
        self,
        input_mask: torch.Tensor,
        target_mask: torch.Tensor,
        foreground_logits: Optional[torch.Tensor] = None,
        model: Optional[pl.LightningModule] = None,
    ):
        target_mask = target_mask.detach() > 0
        device = target_mask.device

        # A nan mask is not considered.
        valid_targets = ~(target_mask.isnan().all(-1).all(-1)).any(-1)
        # Discard first dimension mask as it is background.
        if self.ignore_background:
            # Assume first class in masks is background.
            if len(target_mask.shape) > 4:  # Video data (bs, frame, classes, w, h).
                target_mask = target_mask[:, :, 1:]
            else:  # Image data (bs, classes, w, h).
                target_mask = target_mask[:, 1:]

        targets = target_mask[valid_targets]
        predictions = input_mask[valid_targets]
        if foreground_logits is not None:
            foreground_logits = foreground_logits[valid_targets]

        total_loss = torch.tensor(0.0, device=device)
        num_samples = 0

        # Iterate through each clip. Might think about if parallelable
        for i, (prediction, target) in enumerate(zip(predictions, targets)):
            # Filter empty masks.
            target = target[target.sum(-1).sum(-1) > 0]

            # Compute matching.
            costMatrixSeg = CompDETRSegCostMatrix(
                prediction,
                target,
            )
            # We cannot rely on the matched cost for computing the loss due to
            # normalization issues between segmentation component (normalized by
            # number of matches) and classification component (normalized by
            # number of predictions). Thus compute both components separately
            # after deriving the matching matrix.
            if foreground_logits is not None and self.foreground_matching_weight != 0.0:
                # Positive classification component.
                logits = foreground_logits[i]
                costMatrixTotal = (
                    costMatrixSeg
                    + self.foreground_weight
                    * F.binary_cross_entropy_with_logits(
                        logits, torch.ones_like(logits), reduction="none"
                    ).detach()
                )
            else:
                costMatrixTotal = costMatrixSeg

            # Matcher takes a batch but we are doing this one by one.
            matching_matrix = self.matcher(costMatrixTotal.unsqueeze(0))[0].squeeze(0)
            n_matches = min(predictions.shape[0], target.shape[0])
            if n_matches > 0:
                instance_cost = (costMatrixSeg * matching_matrix).sum(-1).sum(-1) / n_matches
            else:
                instance_cost = torch.tensor(0.0, device=device)

            if foreground_logits is not None:
                ismatched = (matching_matrix > 0).any(-1)
                logits = foreground_logits[i].squeeze(-1)
                instance_cost += self.foreground_weight * F.binary_cross_entropy_with_logits(
                    logits, ismatched.float(), reduction="mean"
                )

            total_loss += instance_cost
            # Normalize by number of matches.
            num_samples += 1

        if (
            model is not None
            and hasattr(model, "trainer")
            and model.trainer.world_size > 1
            and self.global_loss
        ):
            # As data is sparsely labeled return the average loss over all GPUs.
            # This should make the loss a mit more smooth.
            all_losses, sample_counts = model.all_gather([total_loss, num_samples], sync_grads=True)
            total_count = sample_counts.sum()
            if total_count > 0:
                total_loss = all_losses.sum() / total_count
            else:
                total_loss = torch.tensor(0.0, device=device)

            return total_loss * self.loss_weight
        else:
            if num_samples == 0:
                # Avoid division by zero if a batch does not contain any labels.
                return torch.tensor(0.0, device=targets.device)

            total_loss /= num_samples
            total_loss *= self.loss_weight
            return total_loss


class EM_rec_loss(nn.Module, RoutableMixin):
    def __init__(
        self,
        loss_weight: float = 20,
        attn_path: Optional[str] = None,
        rec_path: Optional[str] = None,
        tgt_path: Optional[str] = None,
        img_path: Optional[str] = None,
        tgt_vis_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        attn_index_path: Optional[str] = None,
        slot_path: Optional[str] = None,
        pred_feat_path: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self,
            {
                "segmentations": attn_path,
                "reconstructions": rec_path,
                "masks": tgt_path,
                "masks_vis": tgt_vis_path,
                "rec_tgt": img_path,
                "weights": weights_path,
                "attn_index": attn_index_path,
                "slots": slot_path,
                "pred_slots": pred_feat_path,
            },
        )
        self.loss_weight = loss_weight
        self.loss_fn = lambda x1, x2: nn.functional.mse_loss(x1, x2, reduction="none")

    @RoutableMixin.route
    def forward(
        self,
        segmentations: torch.Tensor,  # rollout_decode.masks
        masks: torch.Tensor,  # decoder.masks
        reconstructions: torch.Tensor,
        rec_tgt: torch.Tensor,
        masks_vis: torch.Tensor,
        attn_index: torch.Tensor,
        slots: torch.Tensor,
        pred_slots: torch.Tensor,
        smooth=1,
    ):
        b, f, c, h, w = segmentations.shape
        _, _, n_slots, n_buffer = attn_index.shape

        segmentations = (
            segmentations.reshape(-1, n_buffer, h, w).unsqueeze(1).repeat(1, n_slots, 1, 1, 1)
        )
        masks = masks.reshape(-1, n_slots, h, w).unsqueeze(2).repeat(1, 1, n_buffer, 1, 1)
        masks = masks > 0.5
        masks_vis = (
            masks_vis.reshape(-1, n_slots, h, w)
            .unsqueeze(2)
            .unsqueeze(3)
            .repeat(1, 1, n_buffer, 3, 1, 1)
        )
        masks_vis = masks_vis > 0.5
        attn_index = attn_index.reshape(-1, n_slots, n_buffer)
        rec_tgt = (
            rec_tgt.reshape(-1, 3, h, w)
            .unsqueeze(1)
            .unsqueeze(2)
            .repeat(1, n_slots, n_buffer, 1, 1, 1)
        )
        reconstructions = (
            reconstructions.reshape(-1, n_buffer, 3, h, w)
            .unsqueeze(1)
            .repeat(1, n_slots, 1, 1, 1, 1)
        )
        rec_pred = reconstructions * masks_vis
        rec_tgt_ = rec_tgt * masks_vis
        loss = torch.sum(
            F.binary_cross_entropy(segmentations, masks.float(), reduction="none"), (-1, -2)
        ) / (h * w) + 0.1 * torch.sum(self.loss_fn(rec_pred, rec_tgt_), (-3, -2, -1))
        total_loss = torch.sum(attn_index * loss, (0, 1, 2)) / (b * f * n_slots * n_buffer)
        return (total_loss) * self.loss_weight
