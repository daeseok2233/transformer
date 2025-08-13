import json
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import random
from typing import List, Tuple
import numpy as np

# --- Tokenizers ---
from konlpy.tag import Okt
import nltk
from nltk.tokenize import word_tokenize

# ===== Constants =====
SOS_token = 0
EOS_token = 1
PAD_token = 2
UNK_token = 3

DEFAULT_MAX_LENGTH = 64  # 필요시 main.py에서 인자로 바꿔 넘겨주세요.

# ===== Utils =====
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_nltk():
    """NLTK 토크나이저 리소스 확보 (이미 있으면 스킵)."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")

# ===== IO =====
def load_json(file_path: str, max_samples: int = None):
    """JSON 파일을 읽고 data 키의 리스트를 반환. max_samples로 앞부분 제한 가능."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw["data"]
    if max_samples is not None:
        items = items[:max_samples]
    return items

def split_fields(items, src_key="ko", tgt_key="mt"):
    """JSON 항목 리스트에서 한국어/영문 문장 리스트 분리."""
    ko = [it[src_key] for it in items]
    en = [it[tgt_key] for it in items]
    return ko, en

# ===== Vocabulary =====
SPECIAL_TOKENS = {"PAD": PAD_token, "SOS": SOS_token, "EOS": EOS_token, "<unk>": UNK_token}

class Lang:
    """단어<->인덱스 사전."""
    def __init__(self, name: str):
        self.name = name
        # 올바른 방향의 매핑
        self.word2index = dict(SPECIAL_TOKENS)  # {"PAD":2, "SOS":0, "EOS":1, "<unk>":3}
        self.index2word = {v: k for k, v in self.word2index.items()}
        # 특수 토큰 카운트도 미리 0으로 초기화
        self.word2count = {tok: 0 for tok in self.word2index.keys()}
        self.n_words = len(self.word2index)  # 4

    def add_sentence(self, sentence: str, tokenizer):
        for w in tokenizer(sentence):
            # 문장 안에 특수 토큰 문자열이 들어있어도 건너뛰기
            if w in self.word2index:  # "SOS", "EOS", "PAD", "<unk>" 등
                # 굳이 skip하지 않고 카운트만 올리고 싶다면 다음 한 줄만 두세요
                self.word2count[w] = self.word2count.get(w, 0) + 1
                continue
            self.add_word(w)

    def add_word(self, word: str):
        if word in self.word2index:
            # 안전 증가
            self.word2count[word] = self.word2count.get(word, 0) + 1
        else:
            idx = self.n_words
            self.word2index[word] = idx
            self.index2word[idx] = word
            self.word2count[word] = 1
            self.n_words += 1

# ===== Pipeline Steps =====
def build_tokenizers():
    """형태소/단어 토크나이저 반환."""
    ensure_nltk()
    tokenizer_ko = Okt().morphs
    tokenizer_en = word_tokenize
    return tokenizer_ko, tokenizer_en

def prepare_data(
    ko_sentences: List[str],
    en_sentences: List[str],
    tokenizer_ko,
    tokenizer_en,
) -> Tuple[Lang, Lang, List[Tuple[str, str]]]:
    """어휘 구축 + (ko, en) 페어 생성."""
    assert len(ko_sentences) == len(en_sentences), "KO/EN 길이가 다릅니다."
    input_lang = Lang("ko")
    output_lang = Lang("en")
    pairs = list(zip(ko_sentences, en_sentences))
    for ko, en in pairs:
        input_lang.add_sentence(ko, tokenizer_ko)
        output_lang.add_sentence(en, tokenizer_en)
    return input_lang, output_lang, pairs

def tensor_from_sentence(
    lang: Lang,
    sentence: str,
    tokenizer,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    """문장을 인덱스 텐서로 변환 (SOS/EOS/PAD 포함, UNK 처리)."""
    idxs = [SOS_token]
    idxs += [lang.word2index.get(w, UNK_token) for w in tokenizer(sentence)[: max_length - 2]]
    idxs.append(EOS_token)
    if len(idxs) < max_length:
        idxs += [PAD_token] * (max_length - len(idxs))
    return torch.tensor(idxs[:max_length], dtype=torch.long, device=device)

def build_dataloader(
    input_lang: Lang,
    output_lang: Lang,
    pairs: List[Tuple[str, str]],
    tokenizer_ko,
    tokenizer_en,
    batch_size: int = 32,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: torch.device = None,
    use_random_sampler: bool = True,
) -> DataLoader:
    """(ko, en) 페어를 텐서로 바꿔 DataLoader 생성."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_tensors = [
        tensor_from_sentence(input_lang, ko, tokenizer_ko, max_length, device) for ko, _ in pairs
    ]
    target_tensors = [
        tensor_from_sentence(output_lang, en, tokenizer_en, max_length, device) for _, en in pairs
    ]

    input_tensors = torch.stack(input_tensors, dim=0)   # [N, L]
    target_tensors = torch.stack(target_tensors, dim=0) # [N, L]

    dataset = TensorDataset(input_tensors, target_tensors)
    if use_random_sampler:
        sampler = RandomSampler(dataset)
        dl = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    else:
        dl = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    return dl

# ===== High-level helper for main.py =====
def make_train_dataloader(
    train_json_path: str,
    max_samples: int = None,
    batch_size: int = 32,
    max_length: int = DEFAULT_MAX_LENGTH,
    src_key: str = "ko",
    tgt_key: str = "mt",
    seed: int = 42,
):
    """한 번에: JSON 로딩 → 토크나이저 → 어휘 구축 → DataLoader 생성."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = load_json(train_json_path, max_samples=max_samples)
    ko, en = split_fields(items, src_key=src_key, tgt_key=tgt_key)
    tok_ko, tok_en = build_tokenizers()
    in_lang, out_lang, pairs = prepare_data(ko, en, tok_ko, tok_en)
    dl = build_dataloader(
        in_lang, out_lang, pairs, tok_ko, tok_en,
        batch_size=batch_size, max_length=max_length, device=device
    )
    return dl, in_lang, out_lang, pairs