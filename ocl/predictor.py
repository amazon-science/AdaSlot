import torch
import torchvision
from torch import nn


class Predictor(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        qkv_size: int = 128,
        mlp_size: int = 256,
        pre_norm: bool = False,
    ):
        nn.Module.__init__(self)

        self.embed_dim = embed_dim
        self.qkv_size = qkv_size
        self.mlp_size = mlp_size
        self.num_heads = num_heads
        self.pre_norm = pre_norm
        self.MHA = nn.MultiheadAttention(embed_dim, num_heads)

        self.head_dim = qkv_size // num_heads
        self.mlp = torchvision.ops.MLP(embed_dim, [mlp_size, embed_dim])
        # layernorms
        self.layernorm_query = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm_mlp = nn.LayerNorm(embed_dim, eps=1e-6)
        # weights
        self.dense_q = nn.Linear(embed_dim, qkv_size)
        self.dense_k = nn.Linear(embed_dim, qkv_size)
        self.dense_v = nn.Linear(embed_dim, qkv_size)
        if self.num_heads > 1:
            self.dense_o = nn.Linear(qkv_size, embed_dim)
            self.multi_head = True
        else:
            self.multi_head = False

    def forward(
        self, object_features: torch.Tensor
    ):  # TODO: add general attention for q, k, v, not just for x = qkv
        assert object_features.ndim == 3
        B, L, _ = object_features.shape
        head_dim = self.embed_dim // self.num_heads

        if self.pre_norm:
            # Self-attention.
            x = self.layernorm_query(object_features)
            q = self.dense_q(x).view(B, L, self.num_heads, head_dim)
            k = self.dense_k(x).view(B, L, self.num_heads, head_dim)
            v = self.dense_v(x).view(B, L, self.num_heads, head_dim)
            x, _ = self.MHA(q, k, v)
            if self.multi_head:
                x = self.dense_o(x.reshape(B, L, self.qkv_size)).view(B, L, self.embed_dim)
            else:
                x = x.squeeze(-2)
            x = x + object_features

            y = x

            # MLP
            z = self.layernorm_mlp(y)
            z = self.mlp(z)
            z = z + y
        else:
            # Self-attention on queries.
            x = object_features
            q = self.dense_q(x).view(B, L, self.num_heads, head_dim)
            k = self.dense_k(x).view(B, L, self.num_heads, head_dim)
            v = self.dense_v(x).view(B, L, self.num_heads, head_dim)
            x, _ = self.MHA(q, k, v)
            if self.multi_head:
                x = self.dense_o(x.reshape(B, L, self.qkv_size)).view(B, L, self.embed_dim)
            else:
                x = x.squeeze(-2)
            x = x + object_features
            x = self.layernorm_query(x)

            y = x

            # MLP
            z = self.mlp(y)
            z = z + y
            z = self.layernorm_mlp(z)
        return z
