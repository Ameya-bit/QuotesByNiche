"""Niche transformer model, restructured for importability and interpretability.

Every module takes a :class:`Config` (constructor injection) instead of reading
module-level globals, so the same definitions can be imported into a notebook,
a training script, or an interp script without hidden coupling.

For attention interpretability, pass ``store_attn=True`` through ``forward``.
The fast path uses ``F.scaled_dot_product_attention`` (which discards the
attention pattern); the ``store_attn`` path runs the equivalent attention
manually and stashes the pre-dropout softmax on each attention module as
``.attn_weights`` with shape ``(B, n_head, T, T)``.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    """Model hyperparameters. Built directly from a checkpoint's ``config`` dict."""

    vocab_size: int = 177
    n_embd: int = 256
    n_head: int = 4
    n_layers: int = 6
    block_size: int = 256
    dropout: float = 0.2


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CompleteSelfAttention(nn.Module):
    """Batched multi-head causal self-attention (the path the trained model uses)."""

    def __init__(self, config: Config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.dropout_p = config.dropout

        self.qkv = nn.Linear(config.n_embd, config.n_embd * 3)
        self.lin_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "mask", torch.tril(torch.ones(config.block_size, config.block_size))
        )

        # Populated when forward is called with store_attn=True: (B, n_head, T, T).
        self.attn_weights: torch.Tensor | None = None

    def forward(self, xb: torch.Tensor, store_attn: bool = False) -> torch.Tensor:
        B, T, C = xb.shape

        qkv = self.qkv(xb)
        q, k, v = qkv.split(self.n_embd, dim=-1)

        k = k.reshape(B, T, self.n_head, self.head_size).transpose(-2, -3)  # (B, n_head, T, head_size)
        q = q.reshape(B, T, self.n_head, self.head_size).transpose(-2, -3)
        v = v.reshape(B, T, self.n_head, self.head_size).transpose(-2, -3)

        if store_attn:
            # Manual path: identical math to SDPA, but exposes the softmax pattern.
            weights = q @ k.transpose(-2, -1) / self.head_size**0.5  # (B, n_head, T, T)
            weights = weights.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
            weights = F.softmax(weights, dim=-1)
            self.attn_weights = weights.detach()  # capture before dropout
            weights = self.attn_dropout(weights)
            out = weights @ v
        else:
            self.attn_weights = None
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout_p if self.training else 0.0,
            )

        out = out.transpose(-2, -3).reshape(B, T, C)
        out = self.lin_proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        return self.ff(xb)


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.comp_full_attend = CompleteSelfAttention(config)
        self.ff = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, xb: torch.Tensor, store_attn: bool = False) -> torch.Tensor:
        attention = xb + self.comp_full_attend(self.ln1(xb), store_attn=store_attn)
        feed_forward = attention + self.ff(self.ln2(attention))
        return feed_forward


class LTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embedding = nn.Embedding(config.block_size, config.n_embd)
        # ModuleList (not Sequential) so store_attn can be threaded per block;
        # the parameter key prefix ("blocks.0.") is identical, so checkpoints load.
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.last_layer = nn.Linear(config.n_embd, config.vocab_size)

    def forward(
        self,
        xb: torch.Tensor,
        targets: torch.Tensor | None = None,
        store_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = xb.shape

        tok_embd = self.token_embedding(xb)  # (B, T, C)
        pos_embd = self.pos_embedding(torch.arange(T, device=xb.device))  # (T, C)
        x = tok_embd + pos_embd

        for block in self.blocks:
            x = block(x, store_attn=store_attn)

        logits = self.last_layer(self.ln_f(x))

        if targets is None:
            loss = None
        else:
            flat_logits = logits.view(-1, self.config.vocab_size)
            flat_targets = targets.view(-1)
            loss = F.cross_entropy(flat_logits, flat_targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens: int, idx: torch.Tensor) -> torch.Tensor:
        block_size = self.config.block_size
        for _ in range(tokens):
            logits, _ = self(idx[:, -block_size:])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_tok), dim=1)
        return idx


def build_model(config: Config) -> LTransformer:
    return LTransformer(config)


@dataclass(frozen=True)
class LoadedModel:
    """A loaded model bundled with its tokenizer and metadata.

    ``encode`` maps a string to a ``(1, T)`` token tensor on the model's device;
    ``decode`` maps a token tensor (any shape) back to a string.
    """

    model: LTransformer
    config: Config
    stoi: dict[str, int]
    itos: dict[int, str]
    val_loss: float
    encode: Callable[[str], torch.Tensor]
    decode: Callable[[torch.Tensor], str]


def load_model(path: str, map_location: str = "cpu") -> LoadedModel:
    """Load a checkpoint saved as ``{'model_state', 'config', 'stoi', 'itos', ...}``.

    Returns a :class:`LoadedModel` holding the model (eval mode, on
    ``map_location``), its config, the tokenizer maps, and ``encode`` / ``decode``
    helpers bound to that tokenizer.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    config = Config(**ckpt["config"])
    model = build_model(config).to(map_location)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    stoi: dict[str, int] = ckpt["stoi"]
    itos: dict[int, str] = ckpt["itos"]
    device = next(model.parameters()).device

    def encode(text: str) -> torch.Tensor:
        return torch.tensor([[stoi[c] for c in text]], dtype=torch.long, device=device)

    def decode(tokens: torch.Tensor) -> str:
        return "".join(itos[int(i)] for i in tokens.flatten().tolist())

    return LoadedModel(
        model=model,
        config=config,
        stoi=stoi,
        itos=itos,
        val_loss=ckpt["val_loss"],
        encode=encode,
        decode=decode,
    )
