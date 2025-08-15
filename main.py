from pathlib import Path
import sys
import torch
import torch.nn as nn

from src.data_preprocessing import (
    load_json, split_fields, build_tokenizers,
    prepare_data, build_dataloader, make_train_dataloader
)

# Bahdanau
from src.models.bahdanau import (
    Seq2SeqAttn, train_one_epoch,  # 학습용
    ids_to_text, DEFAULT_PAD, DEFAULT_SOS, DEFAULT_EOS, DEFAULT_UNK
)

# Transformer (파일명: src/models/tf.py)
from src.models.tf import (
    build_model,
    train_one_epoch_transformer,
    evaluate_transformer,
    greedy_decode_transformer,
)

import config as C


# -----------------------------
# Utils
# -----------------------------
def safe_load(path, map_location):
    """PyTorch 2.4+ weights_only 안전 로더 (하위버전 자동 폴백)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def pick_device():
    if getattr(C, "DEVICE_POLICY", None) == "cpu":
        return torch.device("cpu")
    if getattr(C, "DEVICE_POLICY", None) == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def encode_sentence_ko(sentence, tokenizer_ko, input_lang, max_length, sos_idx, eos_idx, pad_idx):
    """
    한국어 raw 문장을 토크나이즈 → 인덱스 → [SOS ... EOS] → PAD로 max_length까지 패딩하여 [1, L] LongTensor 반환
    """
    tokens = tokenizer_ko(sentence)
    ids = [sos_idx] + [input_lang.word2index.get(tok, DEFAULT_UNK) for tok in tokens] + [eos_idx]
    if len(ids) < max_length:
        ids = ids + [pad_idx] * (max_length - len(ids))
    else:
        ids = ids[:max_length]
        ids[-1] = eos_idx
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # [1, L]


def evaluate_bahdanau(model: Seq2SeqAttn, loader, device: torch.device, pad_idx: int) -> float:
    """
    Bahdanau용 검증 루프 (평균 토큰 크로스엔트로피)
    """
    model.eval()
    crit = nn.CrossEntropyLoss(ignore_index=pad_idx)
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, target = model(xb, yb)  # [B,L,V], [B,L]
            loss = crit(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            ntoks = target.numel() - target.eq(pad_idx).sum().item()
            total_loss += loss.item() * max(1, ntoks)
            total_tokens += max(1, ntoks)
    return total_loss / max(1, total_tokens)


def maybe_load_checkpoint_bahdanau(model, path: Path, device):
    """
    체크포인트가 있으면 로드. vocab 불일치 등으로 실패하면 False 반환(새로 학습).
    """
    if not path.exists():
        return False
    try:
        ckpt = safe_load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[Bahdanau] loaded checkpoint: {path}")
        return True
    except RuntimeError as e:
        print(f"[Bahdanau] checkpoint shape mismatch -> train new. ({e})")
        return False


def maybe_load_checkpoint_transformer(model, path: Path, device):
    """
    체크포인트가 있으면 로드. vocab 불일치 등으로 실패하면 False 반환(새로 학습).
    """
    if not path.exists():
        return False
    try:
        ckpt = safe_load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[Transformer] loaded checkpoint: {path}")
        return True
    except RuntimeError as e:
        print(f"[Transformer] checkpoint shape mismatch -> train new. ({e})")
        return False


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    device = pick_device()
    print("Device:", device.type)

    # ----------------- Data -----------------
    train_loader, input_lang, output_lang, train_pairs = make_train_dataloader(
        train_json_path=C.TRAIN_JSON,
        max_samples=C.TRAIN_MAX_SAMPLES,
        batch_size=C.BATCH_SIZE,
        max_length=C.MAX_LENGTH,
        src_key=C.SRC_KEY,
        tgt_key=C.TGT_KEY,
        seed=C.SEED,
    )

    # vocab 스냅샷 저장 (inference.py가 빠르게 뜨도록)
    Path("artifacts").mkdir(exist_ok=True)
    torch.save({
        "src_word2index": input_lang.word2index,
        "src_index2word": input_lang.index2word,
        "tgt_word2index": output_lang.word2index,
        "tgt_index2word": output_lang.index2word,
    }, "artifacts/vocab.pt")

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

    # ----------------- Tokens -----------------
    pad_idx = getattr(C, "PAD_TOKEN", DEFAULT_PAD)
    sos_idx = getattr(C, "SOS_TOKEN", DEFAULT_SOS)
    eos_idx = getattr(C, "EOS_TOKEN", DEFAULT_EOS)

    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    bah_path = ckpt_dir / "bahdanau_best.pt"
    trf_path = ckpt_dir / "transformer_best.pt"

    # =====================================================================
    # Bahdanau: train+val best 저장 또는 로드 후 스킵
    # =====================================================================
    bahdanau = Seq2SeqAttn(
        src_vocab=input_lang.n_words,
        tgt_vocab=output_lang.n_words,
        emb_dim=256,
        hidden=256,
        pad_idx=pad_idx,
        sos_idx=sos_idx,
        eos_idx=eos_idx
    ).to(device)

    loaded_bah = maybe_load_checkpoint_bahdanau(bahdanau, bah_path, device)
    if not loaded_bah:
        best_val = float("inf"); patience = 5; pat = 0
        EPOCHS = getattr(C, "EPOCHS_BAH", 100)
        for epoch in range(1, EPOCHS + 1):
            tr_loss = train_one_epoch(bahdanau, train_loader, device, pad_idx)
            val_loss = evaluate_bahdanau(bahdanau, valid_loader, device, pad_idx)
            print(f"[Bahdanau][Epoch {epoch}/{EPOCHS}] train {tr_loss:.4f} | valid {val_loss:.4f}")
            if val_loss < best_val - 1e-4:
                best_val = val_loss; pat = 0
                torch.save({
                    "model": bahdanau.state_dict(),
                    "vocab": {
                        "src_word2index": input_lang.word2index,
                        "tgt_word2index": output_lang.word2index,
                        "src_index2word": input_lang.index2word,
                        "tgt_index2word": output_lang.index2word,
                    }
                }, bah_path)
            else:
                pat += 1
                if pat >= patience:
                    print("[Bahdanau] early stop")
                    break
        # reload best
        maybe_load_checkpoint_bahdanau(bahdanau, bah_path, device)

    # =====================================================================
    # Transformer: train+val best 저장 또는 로드 후 스킵
    # =====================================================================
    trf_model, trf_cfg = build_model(
        src_vocab=input_lang.n_words,
        tgt_vocab=output_lang.n_words,
        src_pad_id=pad_idx,
        tgt_pad_id=pad_idx,
        sos_id=sos_idx,
        eos_id=eos_idx,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        label_smoothing=0.1,
        use_amp=True,
        max_norm=1.0,
    )

    loaded_trf = maybe_load_checkpoint_transformer(trf_model, trf_path, trf_cfg.device)
    if not loaded_trf:
        optimizer_trf = torch.optim.AdamW(trf_model.parameters(), lr=5e-4, betas=(0.9, 0.98), weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_trf, factor=0.5, patience=2, verbose=True)
        best_val = float("inf"); patience = 5; pat = 0
        EPOCHS = getattr(C, "EPOCHS_TRF", 100)
        for epoch in range(1, EPOCHS + 1):
            tr_loss = train_one_epoch_transformer(trf_model, train_loader, optimizer_trf, trf_cfg)
            val_loss = evaluate_transformer(trf_model, valid_loader, trf_cfg)
            scheduler.step(val_loss)
            print(f"[Transformer][Epoch {epoch}/{EPOCHS}] train {tr_loss:.4f} | valid {val_loss:.4f}")
            if val_loss < best_val - 1e-4:
                best_val = val_loss; pat = 0
                torch.save({
                    "model": trf_model.state_dict(),
                    "vocab": {
                        "src_word2index": input_lang.word2index,
                        "tgt_word2index": output_lang.word2index,
                        "src_index2word": input_lang.index2word,
                        "tgt_index2word": output_lang.index2word,
                    }
                }, trf_path)
            else:
                pat += 1
                if pat >= patience:
                    print("[Transformer] early stop")
                    break
        # reload best
        maybe_load_checkpoint_transformer(trf_model, trf_path, trf_cfg.device)

    # =====================================================================
    # 인터랙티브/단일 문장 테스트:
    #   - 한국어 문장을 입력하면 Bah/Trf/GT 동시 출력
    #   - GT는 valid set 내에 해당 문장이 '정확히' 있을 때만 표시
    # =====================================================================
    def infer_one_ko_sentence(sentence_ko: str):
        # 검색으로 GT 찾기 (정확 일치)
        gt_text = None
        try:
            idx = ko_valid.index(sentence_ko)
            gt_text = en_valid[idx]
        except ValueError:
            gt_text = None

        # 인코딩
        xb = encode_sentence_ko(sentence_ko, tokenizer_ko, input_lang, C.MAX_LENGTH, sos_idx, eos_idx, pad_idx).to(device)
        xb_trf = xb.to(trf_cfg.device)

        # Bahdanau
        bahdanau.eval()
        pred_bah_ids = bahdanau.greedy_decode(xb, max_len=C.MAX_LENGTH)  # [1, L]
        pred_bah = ids_to_text(pred_bah_ids[0].tolist(), output_lang.index2word,
                               pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx)

        # Transformer
        trf_model.eval()
        pred_trf_ids = greedy_decode_transformer(trf_model, xb_trf, trf_cfg, max_len=C.MAX_LENGTH)  # [1, L]
        pred_trf = ids_to_text(pred_trf_ids[0].tolist(), output_lang.index2word,
                               pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx)

        print("=" * 80)
        print("[SRC]", sentence_ko)
        if gt_text is not None:
            print("[GT ]", gt_text)
        else:
            print("[GT ] (없음; 검증셋에서 동일 문장 못 찾음)")
        print("[Bah]", pred_bah)
        print("[Trf]", pred_trf)

    # ----------------- 실행 모드 -----------------
    # 1) 명령행 인자에 한국어 문장을 주면 그 문장으로 인퍼런스
    #    예:  python main.py "저희 제품 문의 감사합니다."
    if len(sys.argv) > 1:
        infer_one_ko_sentence(" ".join(sys.argv[1:]))
    else:
        # 2) 아니면 검증셋에서 임의 3개 미리보기
        for s in ko_valid[:3]:
            infer_one_ko_sentence(s)