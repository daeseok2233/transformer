# src/models/tf.py
"""
Minimal-yet-solid Seq2Seq Transformer (PyTorch)
- Proper padding/causal masks
- Label smoothing (optional)
- AMP (torch.amp.*) + gradient clipping
- Greedy & (optional) beam search decoding

Expected batch format from your loaders:
  (src, tgt) where both are LongTensor [B, T]
  tgt contains [SOS ... EOS PAD ...]
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


# ----------------------------
# Config
# ----------------------------
@dataclass
class TFConfig:
    device: torch.device
    src_pad_id: int
    tgt_pad_id: int
    sos_id: int
    eos_id: int
    label_smoothing: float = 0.0
    max_norm: float = 1.0
    use_amp: bool = True


# ----------------------------
# Positional Encoding & Embeddings
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x) * self.scale


# ----------------------------
# Seq2Seq Transformer wrapper
# ----------------------------
class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.src_tok = TokenEmbedding(src_vocab, d_model)
        self.tgt_tok = TokenEmbedding(tgt_vocab, d_model)
        self.pos = PositionalEncoding(d_model, dropout)

        self.trf = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # [B, T, C]
            norm_first=True,
        )
        self.generator = nn.Linear(d_model, tgt_vocab)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ---- encoder/decoder ----
    def encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor):
        # src: [B, S], src_key_padding_mask: [B, S] (True=PAD)
        src_emb = self.pos(self.src_tok(src))
        memory = self.trf.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ):
        # tgt: [B, T]
        tgt_emb = self.pos(self.tgt_tok(tgt))
        out = self.trf.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.generator(out)  # [B, T, V]

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        mem = self.encode(src, src_key_padding_mask)
        logits = self.decode(
            tgt=tgt_in,
            memory=mem,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return logits


# ----------------------------
# Masks
# ----------------------------
def create_padding_mask(pad_id: int, seq: torch.Tensor) -> torch.Tensor:
    # seq: [B, T] -> True where PAD
    return seq.eq(pad_id)


def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    # causal mask [T, T] with True above diagonal (masked positions)
    return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)


# ----------------------------
# Label Smoothing Loss
# ----------------------------
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes: int, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: [N, C], target: [N]
        with torch.no_grad():
            true_dist = pred.new_full((pred.size(0), self.cls), self.smoothing / (self.cls - 1))
            mask = target.ne(self.ignore_index)
            true_dist[mask, target[mask]] = self.confidence
            true_dist[~mask] = 0
        return torch.mean(torch.sum(-true_dist * F.log_softmax(pred, dim=-1), dim=-1))


def _criterion(vocab_size: int, cfg: TFConfig):
    if cfg.label_smoothing > 0:
        return LabelSmoothingLoss(vocab_size, cfg.label_smoothing, ignore_index=cfg.tgt_pad_id)
    else:
        return nn.CrossEntropyLoss(ignore_index=cfg.tgt_pad_id)


# ----------------------------
# Builder
# ----------------------------
def build_model(
    src_vocab: int,
    tgt_vocab: int,
    src_pad_id: int,
    tgt_pad_id: int,
    sos_id: int = 1,
    eos_id: int = 2,
    d_model: int = 512,
    nhead: int = 8,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    label_smoothing: float = 0.1,
    use_amp: bool = True,
    max_norm: float = 1.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerSeq2Seq(
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)

    cfg = TFConfig(
        device=device,
        src_pad_id=src_pad_id,
        tgt_pad_id=tgt_pad_id,
        sos_id=sos_id,
        eos_id=eos_id,
        label_smoothing=label_smoothing,
        max_norm=max_norm,
        use_amp=use_amp,
    )
    return model, cfg


# ----------------------------
# Training / Eval
# ----------------------------
def _prep_batch(batch, cfg: TFConfig):
    # Accept (src, tgt) tuple or dict with keys 'src', 'tgt'
    if isinstance(batch, dict):
        src = batch["src"]
        tgt = batch["tgt"]
    else:
        src, tgt = batch  # [B, S], [B, T]

    src = src.to(cfg.device)
    tgt = tgt.to(cfg.device)

    # Decoder inputs/targets
    tgt_in = tgt[:, :-1]   # [SOS ...]
    tgt_out = tgt[:, 1:]   # [... EOS]

    src_pad = create_padding_mask(cfg.src_pad_id, src)     # [B, S]
    tgt_pad = create_padding_mask(cfg.tgt_pad_id, tgt_in)  # [B, T-1]

    T = tgt_in.size(1)
    causal = generate_square_subsequent_mask(T, cfg.device)  # [T, T] (bool)

    return src, tgt_in, tgt_out, src_pad, tgt_pad, causal


def train_one_epoch_transformer(model: TransformerSeq2Seq, loader, optimizer, cfg: TFConfig, vocab_size: Optional[int] = None) -> float:
    model.train()
    device_type = "cuda" if cfg.device.type == "cuda" else "cpu"
    use_amp_now = cfg.use_amp and device_type == "cuda"

    scaler = torch.amp.GradScaler(device_type) if use_amp_now else None
    if vocab_size is None:
        vocab_size = model.generator.out_features
    criterion = _criterion(vocab_size, cfg)

    total_loss, total_tokens = 0.0, 0

    for batch in loader:
        src, tgt_in, tgt_out, src_pad, tgt_pad, causal = _prep_batch(batch, cfg)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device_type, enabled=use_amp_now):
            logits = model(
                src=src,
                tgt_in=tgt_in,
                src_key_padding_mask=src_pad,
                tgt_key_padding_mask=tgt_pad,
                tgt_mask=causal,
            )  # [B, T, V]
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        if scaler is not None:
            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), cfg.max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.max_norm)
            optimizer.step()

        ntoks = tgt_out.numel() - tgt_out.eq(cfg.tgt_pad_id).sum().item()
        total_loss += loss.item() * max(1, ntoks)
        total_tokens += max(1, ntoks)

    return total_loss / max(1, total_tokens)


def evaluate_transformer(model: TransformerSeq2Seq, loader, cfg: TFConfig, vocab_size: Optional[int] = None) -> float:
    model.eval()
    if vocab_size is None:
        vocab_size = model.generator.out_features
    criterion = _criterion(vocab_size, cfg)

    total_loss, total_tokens = 0.0, 0
    device_type = "cuda" if cfg.device.type == "cuda" else "cpu"
    use_amp_now = cfg.use_amp and device_type == "cuda"

    with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=use_amp_now):
        for batch in loader:
            src, tgt_in, tgt_out, src_pad, tgt_pad, causal = _prep_batch(batch, cfg)
            logits = model(
                src=src,
                tgt_in=tgt_in,
                src_key_padding_mask=src_pad,
                tgt_key_padding_mask=tgt_pad,
                tgt_mask=causal,
            )
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            ntoks = tgt_out.numel() - tgt_out.eq(cfg.tgt_pad_id).sum().item()
            total_loss += loss.item() * max(1, ntoks)
            total_tokens += max(1, ntoks)

    return total_loss / max(1, total_tokens)


# ----------------------------
# Inference
# ----------------------------
@torch.no_grad()
def greedy_decode_transformer(
    model: TransformerSeq2Seq,
    src: torch.Tensor,              # [B, S]
    cfg: TFConfig,
    max_len: int = 128,
) -> torch.Tensor:
    model.eval()
    B = src.size(0)
    src = src.to(cfg.device)
    src_pad = create_padding_mask(cfg.src_pad_id, src)

    memory = model.encode(src, src_pad)  # [B, S, C]
    ys = torch.full((B, 1), cfg.sos_id, dtype=torch.long, device=cfg.device)
    finished = torch.zeros(B, dtype=torch.bool, device=cfg.device)

    for _ in range(max_len - 1):
        tgt_pad = create_padding_mask(cfg.tgt_pad_id, ys)
        causal = generate_square_subsequent_mask(ys.size(1), cfg.device)
        out = model.decode(ys, memory, causal, tgt_pad, src_pad)  # [B, T, V]
        next_tok = out[:, -1, :].argmax(dim=-1)
        ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
        finished |= next_tok.eq(cfg.eos_id)
        if finished.all():
            break
    return ys


@torch.no_grad()
def beam_search_decode_transformer(
    model: TransformerSeq2Seq,
    src: torch.Tensor,              # [B, S]
    cfg: TFConfig,
    beam_size: int = 4,
    max_len: int = 128,
    length_penalty: float = 0.7,
) -> torch.Tensor:
    """Simple batched beam search (returns best 1 beam per batch)."""
    model.eval()
    B = src.size(0)
    device = cfg.device
    src = src.to(device)
    src_pad = create_padding_mask(cfg.src_pad_id, src)
    memory = model.encode(src, src_pad)  # [B, S, C]

    beams = torch.full((B, beam_size, 1), cfg.sos_id, dtype=torch.long, device=device)
    scores = torch.zeros(B, beam_size, device=device)
    finished = torch.zeros(B, beam_size, dtype=torch.bool, device=device)

    for step in range(1, max_len):
        cur = beams.reshape(B * beam_size, -1)  # [B*K, T]
        mem = memory.unsqueeze(1).expand(B, beam_size, *memory.shape[1:]).reshape(B * beam_size, *memory.shape[1:])
        spad = src_pad.unsqueeze(1).expand(B, beam_size, *src_pad.shape[1:]).reshape(B * beam_size, *src_pad.shape[1:])

        tgt_pad = create_padding_mask(cfg.tgt_pad_id, cur)
        causal = generate_square_subsequent_mask(cur.size(1), device)
        out = model.decode(cur, mem, causal, tgt_pad, spad)  # [B*K, T, V]
        logp = F.log_softmax(out[:, -1, :], dim=-1)  # [B*K, V]

        logp = logp.view(B, beam_size, -1)
        # Keep EOS if already finished
        logp[finished] = -1e9
        logp[finished, cfg.eos_id] = 0.0

        cand_scores = scores.unsqueeze(-1) + logp  # [B, K, V]
        cand_scores = cand_scores.view(B, -1)      # [B, K*V]
        topk_scores, topk_idx = torch.topk(cand_scores, k=beam_size, dim=-1)

        vocab = logp.size(-1)
        beam_idx = topk_idx // vocab
        token_idx = topk_idx % vocab

        next_beams = torch.empty((B, beam_size, step + 1), dtype=torch.long, device=device)
        next_finished = torch.empty((B, beam_size), dtype=torch.bool, device=device)
        for b in range(B):
            next_beams[b] = torch.cat([beams[b, beam_idx[b]], token_idx[b].unsqueeze(-1)], dim=-1)
            next_finished[b] = finished[b, beam_idx[b]] | token_idx[b].eq(cfg.eos_id)

        beams, scores, finished = next_beams, topk_scores, next_finished
        if finished.all():
            break

    lengths = beams.ne(cfg.tgt_pad_id).sum(dim=-1).clamp(min=1).float()
    lp = ((5.0 + lengths) / 6.0) ** length_penalty
    norm_scores = scores / lp
    best = norm_scores.argmax(dim=-1)
    out = beams[torch.arange(B, device=device), best]
    return out


# ----------------------------
# Quick self-test (optional)
# ----------------------------
if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[tf.py] device:", dev.type)
    src_vocab, tgt_vocab = 100, 120
    model, cfg = build_model(src_vocab, tgt_vocab, src_pad_id=0, tgt_pad_id=0, sos_id=1, eos_id=2)
    B, S, T = 4, 10, 12
    src = torch.randint(3, src_vocab, (B, S))
    tgt = torch.randint(3, tgt_vocab, (B, T))
    tgt[:, 0] = cfg.sos_id
    tgt[:, -1] = cfg.eos_id
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = [(src, tgt) for _ in range(3)]
    loss = train_one_epoch_transformer(model, loader, opt, cfg)
    print("dummy train loss:", round(loss, 4))
    out = greedy_decode_transformer(model, src, cfg, max_len=16)
    print("greedy shape:", tuple(out.shape))