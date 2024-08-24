"""Memory roll-out module, following GPT-2 architecture.

References:
1) minGPT by Andrej Karpathy:
https://github.com/karpathy/minGPT/tree/master/mingpt
2) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
"""

import math

import torch
from torch import nn

# -----------------------------------------------------------------------------


class GELU(nn.Module):
    def forward(self, x):
        return (
            0.5
            * x
            * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        )


class Block(nn.Module):
    """One GPT-2 decoder block, consists of a Masked Self-Attn and a FFN."""

    def __init__(self, n_embd, n_heads, dropout_rate):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(n_embd, 4 * n_embd),
                c_proj=nn.Linear(4 * n_embd, n_embd),
                act=GELU(),
                dropout=nn.Dropout(dropout_rate),
            )
        )
        m = self.mlp
        self.ffn = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x, causal_mask):
        att, att_weights = self.attn(
            query=self.ln_1(x), key=self.ln_1(x), value=self.ln_1(x), attn_mask=causal_mask
        )

        x = x + att
        x = x + self.ffn(self.ln_2(x))
        return x, att_weights


class GPT(nn.Module):
    """Memory roll-out GPT."""

    def __init__(
        self, buffer_len, n_layer, n_head, n_embd, embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0
    ):
        super().__init__()
        self.buffer_len = buffer_len
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(self.n_embd, self.n_embd, bias=False),
                wpe=nn.Embedding(self.buffer_len, self.n_embd),
                drop=nn.Dropout(self.embd_pdrop),
                h=nn.ModuleList(
                    [Block(self.n_embd, self.n_head, self.resid_pdrop) for _ in range(self.n_layer)]
                ),
                ln_f=nn.LayerNorm(self.n_embd),
            )
        )
        # roll out to the same dimension
        self.roll_out_head = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, mem, mem_table, targets=None):
        device = mem.device
        b, t, n, d = mem.shape

        # reshape to merge the batch and num_buffer dimensionsni
        mem = mem.permute(0, 2, 1, 3).reshape(b * n, t, d)
        mem_table = mem_table.view(b * n, -1)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(mem)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # create causal attention masks
        # need to check correctness
        causal_masks = []
        for idx in range(b * n):
            occupied_len = mem_table[idx].cpu().numpy().astype(int)[0]
            if occupied_len == 0:
                occupied_len = 1
            # causal_mask = torch.tril(torch.ones(self.buffer_len, self.buffer_len).to(device)).view(
            #     1, self.buffer_len, self.buffer_len
            # )
            causal_mask = (
                torch.zeros(self.buffer_len, self.buffer_len)
                .to(device)
                .view(1, self.buffer_len, self.buffer_len)
            )
            causal_mask[:, occupied_len:, occupied_len:] = 1
            causal_mask = causal_mask > 0
            causal_masks.append(causal_mask)
        causal_masks = torch.stack(causal_masks)
        causal_masks = causal_masks.repeat(1, self.n_head, 1, 1).view(-1, t, t)

        for block in self.transformer.h:
            x, attn_weights = block(x, causal_masks)
        x = self.transformer.ln_f(x)
        x = self.roll_out_head(x)  # [b*n, t, d]

        out = torch.zeros((b * n, d)).to(device)

        for idx in range(b * n):
            t_pos = mem_table[idx].cpu().numpy().astype(int)[0]
            if t_pos > 0 and t_pos < t:
                out[idx] = x[idx, t_pos - 1]
        return out.view(b, n, d)
