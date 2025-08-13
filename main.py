from pathlib import Path
import torch
from src.data_preprocessing import (
    load_json, split_fields, build_tokenizers,
    prepare_data, build_dataloader, make_train_dataloader
)
from src.models.bahdanau import Seq2SeqAttn, train_one_epoch, translate_one_batch_with_attention

import config as C

def pick_device():
    if C.DEVICE_POLICY == "cpu":
        return torch.device("cpu")
    if C.DEVICE_POLICY == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    device = pick_device()

    # ---- Train ----
    train_loader, input_lang, output_lang, train_pairs = make_train_dataloader(
        train_json_path=C.TRAIN_JSON,
        max_samples=C.TRAIN_MAX_SAMPLES,
        batch_size=C.BATCH_SIZE,
        max_length=C.MAX_LENGTH,
        src_key=C.SRC_KEY,
        tgt_key=C.TGT_KEY,
        seed=C.SEED,
    )

    # ---- Valid ----
    valid_items = load_json(C.VALID_JSON, max_samples=C.VALID_MAX_SAMPLES)
    ko_valid, en_valid = split_fields(valid_items, src_key=C.SRC_KEY, tgt_key=C.TGT_KEY)
    tokenizer_ko, tokenizer_en = build_tokenizers()
    valid_pairs = list(zip(ko_valid, en_valid))
    valid_loader = build_dataloader(
        input_lang, output_lang, valid_pairs,
        tokenizer_ko, tokenizer_en,
        batch_size=C.BATCH_SIZE, max_length=C.MAX_LENGTH
    )

    print("Device:", device)

    # (옵션) 배치 모양 한 번만 확인하고 싶다면 이 4줄만 두세요.
    # for xb, yb in train_loader:
    #     xb = xb.to(device); yb = yb.to(device)
    #     print("train batch shapes:", xb.shape, yb.shape)
    #     break
    
    # ---- Model ----
    # config.py에 아래 3개가 없다면 추가하세요: PAD_TOKEN=2, SOS_TOKEN=0, EOS_TOKEN=1
    pad_idx = getattr(C, "PAD_TOKEN", 2)
    sos_idx = getattr(C, "SOS_TOKEN", 0)
    eos_idx = getattr(C, "EOS_TOKEN", 1)
    
    bahdanau = Seq2SeqAttn(
        src_vocab=input_lang.n_words,
        tgt_vocab=output_lang.n_words,
        emb_dim=256,
        hidden=256,
        pad_idx=pad_idx,
        sos_idx=sos_idx,
        eos_idx=eos_idx
    ).to(device)
    
    # ---- Train for a few epochs ----
    EPOCHS = 100
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(bahdanau, train_loader, device, pad_idx)
        print(f"[Bahdanau][Epoch {epoch}/{EPOCHS}] Train loss: {loss:.4f}")
    
    # ---- Translation check ----
    translate_one_batch_with_attention(
        bahdanau,
        valid_loader,
        input_lang,
        output_lang,
        max_length=C.MAX_LENGTH,
        device=device,
        num_samples=3
    )



