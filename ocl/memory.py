import dataclasses
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ocl import path_defaults
from ocl.memory_rollout import GPT
from ocl.mha import MultiHeadAttention, MultiHeadAttention_for_index
from ocl.utils.routing import RoutableMixin


@dataclasses.dataclass
class MemoryOutput:
    # rollout: TensorType["batch_size", "n_objects", "object_dim"]  # noqa: F821
    # idx_mask: TensorType["batch_size", "n_objects"]  # noqa: F821
    # matched_idx: dict  # noqa: F821
    # object_features: TensorType["batch_size", "n_objects", "n_spatial_features"]  # noqa: F821
    rollout: torch.Tensor  # noqa: F821
    object_features: torch.Tensor  # noqa: F821
    mem: torch.Tensor
    eval_mem_features: torch.Tensor
    table: torch.Tensor
    attn_index: torch.Tensor


class SelfSupervisedMemory(nn.Module, RoutableMixin):
    def __init__(
        self,
        embed_dim: int = 128,
        num_objects: int = 20,
        memory_len: int = 30,
        mlp_size: int = 512,
        mlp_layer: int = 3,
        stale_number: int = 5,
        appearance_threshold: float = 0.2,
        dropout_rate: float = 0.1,
        object_features_path: Optional[str] = path_defaults.OBJECTS,
        conditioning_path: Optional[str] = path_defaults.CONDITIONING,
        attention_maps_path: Optional[str] = None,
        frame_features_path: Optional[str] = path_defaults.FEATURES,
        first_box_path: Optional[str] = None,
        init_flag_path: Optional[str] = None,
        matched_idx_path: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self,
            {
                "observation": object_features_path,
                "conditions": conditioning_path,
                "attention_maps": attention_maps_path,
                "frame_features": frame_features_path,
                "first_frame_boxes": first_box_path,
                "init_flag": init_flag_path,
                "matched_index": matched_idx_path,
            },
        )
        self.embed_dim = embed_dim
        self.memory_len = memory_len
        self.object_num = num_objects
        self.stale_number = stale_number
        self.threshold = appearance_threshold
        self.num_heads = 4
        self.roll_out_module = GPT(buffer_len=memory_len, n_layer=8, n_head=8, n_embd=embed_dim)
        self.register_buffer("memory", torch.zeros(8, memory_len, num_objects, embed_dim))
        self.register_buffer("memory_table", torch.zeros(8, num_objects))
        self.register_buffer("stale_counter", torch.zeros(8, num_objects))
        self.MultiHead_1 = MultiHeadAttention_for_index(
            n_head=4, d_model=embed_dim, d_k=embed_dim, d_v=embed_dim
        )  # n_head=1
        self.MultiHead_2 = MultiHeadAttention(
            n_head=4, d_model=embed_dim, d_k=embed_dim, d_v=embed_dim
        )

    def remove_duplicated_slot_id(self, slot_masks):
        slot_masks = slot_masks > 0.5
        n, h, w = slot_masks.shape
        # remove empty slots
        mask_sum = torch.sum(slot_masks.reshape(-1, h * w), dim=-1)
        # In SAVi, we give the first slot background init.
        # BUG: this is not always the case.
        bg_value = mask_sum[0]
        bg_idx = (mask_sum == bg_value).nonzero(as_tuple=True)[0]
        empty_idx = (mask_sum <= 10).nonzero(as_tuple=True)[0]

        # remove duplicated masks
        mask = slot_masks.unsqueeze(1).to(torch.bool).reshape(n, 1, -1)
        mask_ = slot_masks.unsqueeze(0).to(torch.bool).reshape(1, n, -1)
        intersection = torch.sum(mask & mask_, dim=-1).to(torch.float64)
        union = torch.sum(mask | mask_, dim=-1).to(torch.float64)
        pairwise_iou = intersection / union
        pairwise_iou[union == 0] = 1.0
        dup_idx = []
        for i in range(n):
            for j in range(i + 1, n):
                if pairwise_iou[i, j] > 0.5:
                    dup_idx.append(i)
        invalid_idx = [*set(list(bg_idx) + list(empty_idx) + list(dup_idx))]
        valid_idx = []
        for i in range(n):
            if i not in invalid_idx:
                valid_idx.append(i)
        if len(empty_idx) == n:
            valid_idx.append(0)

        return valid_idx

    def search_bg_id(self, slot_masks):
        slot_masks = slot_masks > 0.5
        n, h, w = slot_masks.shape
        mask_sum = torch.sum(slot_masks.reshape(-1, h * w), dim=-1)
        idx = torch.argmin(mask_sum)
        # print(idx, mask_sum)
        return idx

    def initialization(self, box, conditions, cur_slots, cur_slot_masks, prev_slot_masks, frame_id):
        if frame_id == 0:
            # For each video, we should initialize the register buffers as zero
            bs = conditions.shape[0]
            memory_shape = (bs, self.memory_len, self.object_num, self.embed_dim)
            memory_table_shape = (bs, self.object_num)
            stale_counter_shape = (bs, self.object_num)
            self.memory = torch.zeros(memory_shape).to(conditions.device)
            self.memory_table = torch.zeros(memory_table_shape).to(conditions.device)
            self.stale_counter = torch.zeros(stale_counter_shape).to(conditions.device)
            for b in range(bs):
                valid_idx = self.remove_duplicated_slot_id(cur_slot_masks[b])
                num_obj = len(valid_idx)
                # bg
                self.memory[b, 0, 0, :] = conditions[b, 0, :]
                # non duplicated objects
                self.memory[b, 0, 1 : num_obj + 1, :] = conditions[b, valid_idx, :]
                self.memory_table[b, : num_obj + 1] += 1
        else:
            """IoU score to find new objects"""
            bs, n, h, w = prev_slot_masks.shape
            for b in range(bs):
                # self.memory_eval[b, frame_id, -1, :] = ori_slots[b, 0, :]
                cur_valid_idx = self.remove_duplicated_slot_id(cur_slot_masks[b])
                pre_valid_idx = self.remove_duplicated_slot_id(prev_slot_masks[b])

                cur_slot_mask = cur_slot_masks[b][cur_valid_idx] > 0.5
                prev_slot_mask = prev_slot_masks[b][pre_valid_idx] > 0.5

                # calculate pairwise iou
                cur_mask = (
                    cur_slot_mask.unsqueeze(1).to(torch.bool).reshape(len(cur_valid_idx), 1, -1)
                )
                prev_mask = (
                    prev_slot_mask.unsqueeze(0).to(torch.bool).reshape(1, len(pre_valid_idx), -1)
                )
                intersection = torch.sum(cur_mask & prev_mask, dim=-1).to(torch.float64)
                # union = torch.sum(cur_mask | prev_mask, dim=-1).to(torch.float64)
                # pairwise_iou = intersection / union
                # Remove NaN from divide-by-zero: class does not occur, and class was not predicted.
                # pairwise_iou[union == 0] = 1.0
                sim, _ = torch.max(intersection, dim=-1)
                # NOTE: now using absolute value to determine new object. This might not be optimal.
                # Can have a check with IOU tracker to see their In-logic.
                new_obj_idx = list((sim < 10).nonzero(as_tuple=True)[0])

                new_obj_idx_ori = [cur_valid_idx[id] for id in new_obj_idx]
                num_new_obj = len(new_obj_idx_ori)

                new_mem_idx = list((self.memory_table[b] == 0).nonzero(as_tuple=True)[0])
                old_mem_idx = list((self.memory_table[b] != 0).nonzero(as_tuple=True)[0])
                if num_new_obj > 0 and len(new_mem_idx) > 0:
                    last_pos = old_mem_idx[-1] + 1
                    if last_pos + num_new_obj - 1 in new_mem_idx:
                        self.memory[b, 0, last_pos : last_pos + num_new_obj] = cur_slots[
                            b, new_obj_idx_ori
                        ]
                        self.memory_table[b, last_pos : last_pos + num_new_obj] += 1

    def soft_update(self, observations, predictions):
        inputs = torch.cat((observations, predictions), -1)
        alpha = self.amodal_prediction(inputs)
        outputs = alpha * observations + (1 - alpha) * predictions
        return F.normalize(outputs, dim=-1)

    def buffer_terminate(self):
        bs = self.stale_counter.shape[0]
        for b in range(bs):
            terminate_idx = (self.stale_counter >= self.stale_number).nonzero(as_tuple=True)[0]
            num_dead_buffer = len(list(terminate_idx))
            tmp = torch.zeros((self.memory_len, num_dead_buffer, self.embed_dim)).to(
                self.memory.device
            )
            self.memory[b, :, terminate_idx] = tmp
            self.stale_counter[b, terminate_idx] = 0

    def sms_attn_index_only(self, observations, predictions):
        # implement for multi-head-attention
        b, h, w = observations.shape

        attn_o_to_p, attn_o_to_p_weights = self.MultiHead_1(
            F.normalize(observations, dim=-1),
            F.normalize(predictions, dim=-1),
            F.normalize(predictions, dim=-1),
        )
        sim = attn_o_to_p_weights
        mask = torch.zeros(sim.shape).to(sim.device)
        b, h, w = mask.shape
        for i in range(b):
            for j in range(w):
                index = torch.argmax(sim[i, :, j])
                mask[i, index, j] = 1
        b, h, w = mask.shape
        mask = sim + (mask - sim).detach()
        mask = mask.transpose(1, 2)
        object_feature = torch.einsum("bcn,bnk->bck", [mask, observations])
        # momentum update memory
        alpha = 1
        # print(alpha)
        mem_features = predictions * alpha + object_feature * (1 - alpha)
        mem_features = F.normalize(mem_features, dim=-1)

        return object_feature, mem_features, attn_o_to_p_weights

    def sms_attn(self, observations, predictions, eval_flag):
        attn_o_to_p, attn_o_to_p_weights = self.MultiHead_1(observations, predictions, predictions)

        mask = torch.zeros(attn_o_to_p_weights.shape).to(attn_o_to_p_weights.device)
        b, w, h = mask.shape
        for i in range(b):
            for j in range(w):
                index = torch.argmax(attn_o_to_p_weights[i, j, :])
                mask[i, j, index] = 1
        # mask = attn_o_to_p_weights + (mask - attn_o_to_p_weights).detach()

        # attn_o_to_p_weights_gumbel = F.gumbel_softmax(attn_o_to_p_weights, tau=1, hard=True)
        weights = mask.clone()

        # MultiHead_2 layer
        if not eval_flag:
            attn_o_to_p_weights_trans = torch.transpose(attn_o_to_p_weights, 1, 2)
        else:
            attn_o_to_p_weights_trans = torch.transpose(weights, 1, 2)

        attn_p_to_o, attn_p_to_o_weights = self.MultiHead_2(
            predictions, observations, observations, mask=attn_o_to_p_weights_trans
        )

        # replace the attn_p_to_o with predictions if the buffer is not assigned
        if eval_flag:
            b, h, w = weights.shape
            weights_new = torch.zeros((b, h + 1, w)).to(attn_o_to_p.device)  # [b, n+1, m]
            weights_new[:, 0:h, :] = weights
            weights_new[:, h, :] = torch.sum(weights, dim=1)
            weights_new_convert_zero = weights_new[:, h, :].clone()
            weights_new_convert_zero[weights_new[:, h, :] == 0] = 1
            weights_new_convert_zero[weights_new[:, h, :] > 0] = 0
            weights_new[:, h, :] = weights_new_convert_zero
            b_p, h_p, w_p = attn_p_to_o.shape  # merged features
            for j in range(b_p):
                index = weights_new[j, h, :].nonzero(as_tuple=True)[0]  # hard index
                if len(index) > 0:
                    # update the buffer that no slots matched with zero embeddings
                    attn_p_to_o[j][index] = torch.zeros((len(index), self.embed_dim)).to(
                        observations.device
                    )
                    # attn_p_to_o[j][index] = predictions[j][index].clone()
        else:
            b_p, h_p, w_p = attn_p_to_o.shape  # merged features
            for j in range(b_p):
                # index = weights_new[j, h, :].nonzero(as_tuple=True)[0]  # hard index
                index = (self.memory_table[j] == 0).nonzero(as_tuple=True)[0]
                if len(index) > 0:
                    # update the buffer that no slots matched with zero embeddings
                    attn_p_to_o[j][index] = torch.zeros((len(index), self.embed_dim)).to(
                        observations.device
                    )
        return attn_p_to_o, weights, attn_o_to_p_weights

    def sms_attn_merge(self, observations, predictions):
        # MultiHead_2 layer
        attn_p_to_o, attn_p_to_o_weights = self.MultiHead_2(predictions, observations, observations)
        attn_p_to_o = self.merge_mlp(attn_p_to_o)
        # replace the attn_p_to_o with predictions if the buffer is not assigned
        b_p, h_p, w_p = attn_p_to_o.shape
        return attn_p_to_o, 0

    def update_sms(self, object_features):
        # implement for c2 sms memory update

        object_features_ = object_features.clone().detach()

        # update memory
        for b in range(object_features_.shape[0]):
            for i in range(object_features_.shape[1]):
                tmp = torch.sum(object_features_[b, i, :], dim=-1)
                if tmp != 0 and self.memory_table[b, i] != 0:
                    # if self.memory_table[b, i] != 0:
                    pos = self.memory_table[b, i].cpu().numpy().astype(int)
                    self.memory[b, pos, i] = object_features_[b, i]
                    # self.memory_eval[b, pos, i] = object_features_[b, i]
                    self.memory_table[b, i] += 1
                else:
                    self.stale_counter[b, i] += 1

        return object_features

    @RoutableMixin.route
    def forward(
        self,
        box: torch.Tensor,
        observations: torch.Tensor,
        prev_slot_masks: torch.Tensor,
        cur_slot_masks: torch.Tensor,
        conditions: torch.Tensor,
        frame_id: int,
    ):
        eval = not self.training
        self.initialization(box, conditions, observations, cur_slot_masks, prev_slot_masks, frame_id)
        if frame_id == 0:
            predictions = self.memory[:, 0].clone()
            object_features = predictions.clone()
            b, n_slots = observations.shape[:2]
            n_buffer = predictions.shape[1]
            attn_index = torch.zeros((b, n_slots, n_buffer)).to(observations.device)
        else:
            # memory roll out
            predictions = self.roll_out_module(self.memory, self.memory_table)
            object_features, weights, attn_index = self.sms_attn(
                observations, predictions, eval_flag=eval
            )
            # memory update
            _ = self.update_sms(object_features)

        # memory terminate
        # NOTE No termination?
        # self.buffer_terminate()
        return MemoryOutput(
            rollout=predictions,
            object_features=object_features,
            mem=self.memory,
            eval_mem_features=object_features,
            table=self.memory_table,
            attn_index=attn_index,
        )
