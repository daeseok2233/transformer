from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== 기본 토큰 인덱스 (필요시 모델/함수 인자로 덮어쓰기 가능) =====
DEFAULT_SOS = 0
DEFAULT_EOS = 1
DEFAULT_PAD = 2
DEFAULT_UNK = 3


# ===== 유틸 =====
def shift_for_teacher_forcing(y: torch.Tensor, pad_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    y: [B, L]  (SOS ... EOS PAD PAD ...)
    return: (dec_input, target) 둘 다 [B, L]
    """
    dec_inp = y[:, :-1]
    target = y[:, 1:]
    # 마지막 칸 맞추기
    dec_inp = F.pad(dec_inp, (0, 1), value=pad_idx)
    target = F.pad(target, (0, 1), value=pad_idx)
    return dec_inp, target


def ids_to_text(ids: List[int], idx2word: dict, pad_idx=DEFAULT_PAD, sos_idx=DEFAULT_SOS, eos_idx=DEFAULT_EOS) -> str:
    out = []
    for i in ids:
        if i in (pad_idx, sos_idx):
            continue
        if i == eos_idx:
            break
        out.append(idx2word.get(int(i), "<unk>"))
    return " ".join(out)


# ===== Encoder =====
class EncoderRNN(nn.Module):
    def __init__(self, input_size: int, emb_dim: int, hidden_size: int, pad_idx: int, dropout_p: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T]
        return:
          enc_out: [B, T, H]
          h: [1, B, H]
        """
        emb = self.dropout(self.embedding(x))
        enc_out, h = self.gru(emb)
        return enc_out, h


# ===== Bahdanau Attention =====
class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden: int, dec_hidden: int, attn_dim: Optional[int] = None):
        super().__init__()
        if attn_dim is None:
            attn_dim = dec_hidden
        self.W_h = nn.Linear(enc_hidden, attn_dim, bias=False)
        self.W_s = nn.Linear(dec_hidden, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, enc_out: torch.Tensor, enc_mask: torch.Tensor, s_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        enc_out: [B, T, H]
        enc_mask: [B, T]  (True=PAD)
        s_prev: [B, H]
        return: (context [B,H], attn [B,T])
        """
        score = self.v(torch.tanh(self.W_h(enc_out) + self.W_s(s_prev)[:, None, :])).squeeze(-1)  # [B,T]
        score.masked_fill_(enc_mask, float("-inf"))
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)  # [B,H]
        return context, attn


# ===== Decoder with Attention =====
class AttnDecoderRNN(nn.Module):
    def __init__(self, output_size: int, emb_dim: int, enc_hidden: int, dec_hidden: int,
                 pad_idx: int, sos_idx: int, dropout_p: float = 0.1):
        super().__init__()
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx

        self.embedding = nn.Embedding(output_size, emb_dim, padding_idx=pad_idx)
        self.attn = BahdanauAttention(enc_hidden=enc_hidden, dec_hidden=dec_hidden)
        self.gru = nn.GRU(emb_dim + enc_hidden, dec_hidden, batch_first=True)
        self.fc = nn.Linear(dec_hidden + enc_hidden, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward_step(self, y_prev: torch.Tensor, s_prev: torch.Tensor,
                     enc_out: torch.Tensor, enc_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        y_prev: [B] (토큰)
        s_prev: [1,B,H] or [B,H] → 내부에서 [1,B,H]로 사용
        enc_out: [B,T,H]
        enc_mask: [B,T]
        return: (logits [B,V], s [B,H], attn [B,T])
        """
        if s_prev.dim() == 3:  # [1,B,H] -> [B,H]
            s_t = s_prev.squeeze(0)
        else:
            s_t = s_prev

        emb = self.dropout(self.embedding(y_prev)).unsqueeze(1)  # [B,1,E]
        context, attn = self.attn(enc_out, enc_mask, s_t)        # [B,H], [B,T]
        inp = torch.cat([emb, context.unsqueeze(1)], dim=-1)     # [B,1,E+H]
        out, s = self.gru(inp, s_t.unsqueeze(0))                  # out:[B,1,H], s:[1,B,H]
        out = out.squeeze(1); s = s.squeeze(0)                   # [B,H], [B,H]
        logits = self.fc(torch.cat([out, context], dim=-1))      # [B,V]
        return logits, s, attn

    def forward(self, enc_out: torch.Tensor, enc_mask: torch.Tensor, enc_hidden: torch.Tensor,
                tgt: Optional[torch.Tensor], max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher Forcing 전체 unfold.
        enc_out: [B,T,H], enc_mask: [B,T], enc_hidden: [1,B,H]
        tgt: [B,L] or None (추론시)
        return: (logits [B,L,V], attn_all [B,L,T])
        """
        B = enc_out.size(0)
        L = max_length if tgt is None else tgt.size(1)

        y_prev = torch.full((B,), self.sos_idx, dtype=torch.long, device=enc_out.device)
        s = enc_hidden.squeeze(0)  # [B,H]

        logits_list = []
        attn_list = []
        for t in range(L):
            logit_t, s, attn_t = self.forward_step(y_prev, s, enc_out, enc_mask)  # [B,V], [B,H], [B,T]
            logits_list.append(logit_t.unsqueeze(1))
            attn_list.append(attn_t.unsqueeze(1))
            if tgt is not None:
                y_prev = tgt[:, t]
            else:
                y_prev = logit_t.argmax(dim=-1)

        logits = torch.cat(logits_list, dim=1)   # [B,L,V]
        attn_all = torch.cat(attn_list, dim=1)   # [B,L,T]
        return logits, attn_all


# ===== High-level Seq2Seq with Attention =====
class Seq2SeqAttn(nn.Module):
    """
    사용 예:
      model = Seq2SeqAttn(src_vocab=in_lang.n_words, tgt_vocab=out_lang.n_words,
                          emb_dim=256, hidden=256, pad_idx=2, sos_idx=0, eos_idx=1)
      logits, target = model(src_ids, tgt_ids)  # 학습
      pred_ids = model.greedy_decode(src_ids, max_len=64)  # 추론
    """
    def __init__(self,
                 src_vocab: int,
                 tgt_vocab: int,
                 emb_dim: int = 256,
                 hidden: int = 256,
                 pad_idx: int = DEFAULT_PAD,
                 sos_idx: int = DEFAULT_SOS,
                 eos_idx: int = DEFAULT_EOS,
                 dropout: float = 0.1):
        super().__init__()
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

        self.encoder = EncoderRNN(src_vocab, emb_dim, hidden, pad_idx, dropout_p=dropout)
        self.decoder = AttnDecoderRNN(tgt_vocab, emb_dim, enc_hidden=hidden, dec_hidden=hidden,
                                      pad_idx=pad_idx, sos_idx=sos_idx, dropout_p=dropout)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        src, tgt: [B,L]
        return: (logits [B,L,V], target [B,L])
        """
        enc_out, h = self.encoder(src)                 # [B,T,H], [1,B,H]
        enc_mask = (src == self.pad_idx)              # [B,T]  True=PAD
        dec_inp, target = shift_for_teacher_forcing(tgt, self.pad_idx)  # [B,L],[B,L]
        logits, _ = self.decoder(enc_out, enc_mask, h, dec_inp, max_length=dec_inp.size(1))
        return logits, target

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, max_len: int = 64) -> torch.Tensor:
        """
        src: [B,L]
        return: ids [B, max_len]
        """
        self.eval()
        enc_out, h = self.encoder(src)
        enc_mask = (src == self.pad_idx)
        B = src.size(0)
        y = torch.full((B,), self.sos_idx, dtype=torch.long, device=src.device)
        outs = []
        for _ in range(max_len):
            logit_t, h_next, _ = self.decoder.forward_step(y, h, enc_out, enc_mask)
            y = logit_t.argmax(dim=-1)
            outs.append(y.unsqueeze(1))
            h = h_next
        return torch.cat(outs, dim=1)  # [B, max_len]


# ===== Training helpers =====
def train_one_epoch(model: Seq2SeqAttn, loader, device: torch.device, pad_idx: int,
                    lr: float = 3e-4, clip: float = 1.0) -> float:
    model.train()
    crit = nn.CrossEntropyLoss(ignore_index=pad_idx)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)  # [B,L]
        logits, target = model(xb, yb)         # [B,L,V], [B,L]
        loss = crit(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        total += loss.item()
    return total / max(1, len(loader))


# ===== Inference & (optional) Attention viz =====
@torch.no_grad()
def translate_one_batch_with_attention(model: Seq2SeqAttn, dataloader, input_lang, output_lang,
                                       max_length: int, device: torch.device, num_samples: int = 1,
                                       print_attention: bool = False):
    """
    dataloader에서 한 배치만 꺼내어 번역/정답/어텐션(옵션)을 출력.
    """
    model.eval()
    xb, yb = next(iter(dataloader))
    xb, yb = xb.to(device), yb.to(device)

    pred_ids = model.greedy_decode(xb, max_len=max_length)  # [B,L]
    B = xb.size(0)

    shown = 0
    for i in range(B):
        if shown >= num_samples:
            break
        src_text = ids_to_text(xb[i].tolist(), input_lang.index2word)
        gt_text = ids_to_text(yb[i].tolist(), output_lang.index2word)
        pred_text = ids_to_text(pred_ids[i].tolist(), output_lang.index2word)
        print("=" * 80)
        print(f"[SRC] {src_text}")
        print(f"[GT ] {gt_text}")
        print(f"[Bah] {pred_text}")
        shown += 1

    if print_attention:
        print("\n(attention 시각화는 학습 루프 중 step attention을 저장하도록 추가 구현 필요)")