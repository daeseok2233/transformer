from pathlib import Path
import sys, re, unicodedata
import torch

from src.models.bahdanau import (
    Seq2SeqAttn, ids_to_text,
    DEFAULT_PAD, DEFAULT_SOS, DEFAULT_EOS, DEFAULT_UNK
)
from src.models.tf import (
    build_model,
    greedy_decode_transformer,
)

import config as C


# ---------------- Utils ----------------
def pick_device():
    pol = getattr(C, "DEVICE_POLICY", None)
    if pol == "cpu":  return torch.device("cpu")
    if pol == "cuda": return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_load(path, map_location):
    # 모델 체크포인트용 안전 로더 (PyTorch 2.4+만 weights_only 지원)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def clean_ko_text(s: str) -> str:
    # surrogate 제거 + 널 제거 + NFC 정규화
    s = ''.join(ch for ch in s if not (0xD800 <= ord(ch) <= 0xDFFF))
    s = s.replace('\x00', ' ')
    return unicodedata.normalize('NFC', s).strip()

def simple_ko_tokenize(s: str):
    # 아주 단순한 토크나이저 (빠름)
    return re.findall(r'[가-힣]+|[A-Za-z]+|\d+|[^\s]', s)

def as_idx2word_map(i2w):
    # ids_to_text가 dict의 .get을 쓰므로, list면 dict로 변환
    return i2w if isinstance(i2w, dict) else {i: w for i, w in enumerate(i2w)}

class Lang:
    def __init__(self, word2index, index2word):
        self.word2index = dict(word2index)
        if isinstance(index2word, dict):
            # dict → 리스트로 복원
            m = max(int(k) for k in index2word.keys()) if index2word else -1
            lst = [None]*(m+1)
            for k, v in index2word.items():
                lst[int(k)] = v
            self.index2word = lst
        else:
            self.index2word = list(index2word)
        self.n_words = len(self.word2index)

def load_vocab_from_artifacts() -> tuple[Lang, Lang]:
    vocab_path = Path("artifacts/vocab.pt")
    if not vocab_path.exists():
        raise FileNotFoundError("artifacts/vocab.pt 가 없습니다. 학습 시 vocab 스냅샷을 저장해 주세요.")
    # vocab.pt는 일반 파이썬 객체라 weights_only 사용하지 않음
    v = torch.load(vocab_path, map_location="cpu")
    in_lang  = Lang(v["src_word2index"], v["src_index2word"])
    out_lang = Lang(v["tgt_word2index"], v["tgt_index2word"])
    return in_lang, out_lang

@torch.no_grad()
def encode_sentence_ko(sentence: str, input_lang: Lang, max_length: int, sos_idx: int, eos_idx: int, pad_idx: int):
    sentence = clean_ko_text(sentence)
    tokens = simple_ko_tokenize(sentence)
    ids = [sos_idx] + [input_lang.word2index.get(tok, DEFAULT_UNK) for tok in tokens] + [eos_idx]
    if len(ids) < max_length:
        ids = ids + [pad_idx] * (max_length - len(ids))
    else:
        ids = ids[:max_length]; ids[-1] = eos_idx
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # [1, L]


# ---------------- Load models ----------------
def load_bah_model(input_lang: Lang, output_lang: Lang, pad_idx, sos_idx, eos_idx, device):
    ckpt = Path("checkpoints/bahdanau_best.pt")
    if not ckpt.exists():
        print("[Bahdanau] checkpoint not found:", ckpt)
        return None
    m = Seq2SeqAttn(
        src_vocab=input_lang.n_words, tgt_vocab=output_lang.n_words,
        emb_dim=256, hidden=256, pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx
    ).to(device)
    state = safe_load(ckpt, map_location=device)
    m.load_state_dict(state["model"])
    m.eval()
    print("[Bahdanau] loaded:", ckpt)
    return m

def load_trf_model(input_lang: Lang, output_lang: Lang, pad_idx, sos_idx, eos_idx):
    m, cfg = build_model(
        src_vocab=input_lang.n_words, tgt_vocab=output_lang.n_words,
        src_pad_id=pad_idx, tgt_pad_id=pad_idx,
        sos_id=sos_idx, eos_id=eos_idx,
        d_model=512, nhead=8,
        num_encoder_layers=6, num_decoder_layers=6,
        dim_feedforward=2048, dropout=0.1,
        label_smoothing=0.1, use_amp=True, max_norm=1.0,
    )
    ckpt = Path("checkpoints/transformer_best.pt")
    if not ckpt.exists():
        print("[Transformer] checkpoint not found:", ckpt)
        return None, cfg
    state = safe_load(ckpt, map_location=cfg.device)
    m.load_state_dict(state["model"])
    m.eval()
    print("[Transformer] loaded:", ckpt)
    return m, cfg


# ---------------- Main ----------------
if __name__ == "__main__":
    device = pick_device()
    print("Device:", device.type)

    # 1) vocab 스냅샷 바로 로드 (훈련 데이터/OKT 로딩 없음)
    try:
        input_lang, output_lang = load_vocab_from_artifacts()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # 2) 토큰 인덱스
    pad_idx = getattr(C, "PAD_TOKEN", DEFAULT_PAD)
    sos_idx = getattr(C, "SOS_TOKEN", DEFAULT_SOS)
    eos_idx = getattr(C, "EOS_TOKEN", DEFAULT_EOS)

    # 3) 모델 로드
    bah = load_bah_model(input_lang, output_lang, pad_idx, sos_idx, eos_idx, device)
    trf, cfg = load_trf_model(input_lang, output_lang, pad_idx, sos_idx, eos_idx)

    if bah is None and trf is None:
        print("No checkpoints found. Train first (run main.py).")
        sys.exit(1)

    def infer_line(s: str):
        xb = encode_sentence_ko(s, input_lang, C.MAX_LENGTH, sos_idx, eos_idx, pad_idx)
        print("=" * 80)
        print("[SRC]", s)

        i2w = as_idx2word_map(output_lang.index2word)  # ← list/dict 안전 변환

        if bah is not None:
            pred_ids = bah.greedy_decode(xb.to(device), max_len=C.MAX_LENGTH)
            out = ids_to_text(pred_ids[0].tolist(), i2w, pad_idx, sos_idx, eos_idx)
            print("[Bah]", out)

        if trf is not None:
            pred_ids = greedy_decode_transformer(trf, xb.to(cfg.device), cfg, max_len=C.MAX_LENGTH)
            out = ids_to_text(pred_ids[0].tolist(), i2w, pad_idx, sos_idx, eos_idx)
            print("[Trf]", out)

    # CLI 인자 있으면 한 번, 없으면 REPL
    if len(sys.argv) > 1:
        infer_line(" ".join(sys.argv[1:]))
    else:
        print("Type Korean sentences (empty = quit)")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            infer_line(line)
