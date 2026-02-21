# ===============================================================
# RESEARCH-GRADE MULTI-TEACHER DISTILLATION PIPELINE
# English → Telugu Neural Machine Translation
#
# Teachers:
#   - facebook/nllb-200-distilled-600M
#   - ai4bharat/indictrans2-en-indic-1B
#
# Student:
#   - Custom 6-layer Transformer (Encoder-Decoder)
#
# Features:
#   - SentencePiece subword training
#   - Hash-based caching
#   - Deterministic training
#   - 3-phase training
#   - Multi-teacher equal-weight aggregation
#   - Proper KL distillation
#   - Beam search decoding
#   - Real sacreBLEU evaluation
#
# ===============================================================

import os
import sys
import time
import math
import random
import hashlib
from typing import List, Tuple

import numpy as np
import sacrebleu
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
# CONFIGURATION
# ===============================================================

DATA_PATH = "english-telugu.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42

# SentencePiece
SP_SRC_VOCAB = 12000
SP_TGT_VOCAB = 16000
SP_SRC_PREFIX = "sp_src"
SP_TGT_PREFIX = "sp_tgt"

MAX_SRC_LEN = 64
MAX_TGT_LEN = 100

PAD_ID = 0

# Training
EPOCHS_PHASE1 = 40
EPOCHS_PHASE2 = 30
EPOCHS_PHASE3 = 8

BATCH_SIZE = 128
DISTILL_BATCH_SIZE = 8

LR_PHASE1 = 2e-4
LR_PHASE2 = 1e-4
LR_PHASE3 = 2e-5

GRAD_CLIP = 1.0

TEMPERATURE = 2.0
ALPHA = 0.6  # KL weight

PATIENCE = 5

# Teacher models
TEACHER_1 = "facebook/nllb-200-distilled-600M"
TEACHER_2 = "ai4bharat/indictrans2-en-indic-1B"

# Aggregation (simple equal weighting)
TEACHER_WEIGHT_1 = 0.5
TEACHER_WEIGHT_2 = 0.5

# Beam search
BEAM_SIZE = 5
LENGTH_NORM_ALPHA = 0.7

# Cache
TEACHER1_CACHE = "teacher1_outputs.txt"
TEACHER2_CACHE = "teacher2_outputs.txt"
HASH_CACHE = "data_hash.txt"
SP_HASH_CACHE = "sp_hash.txt"

# ===============================================================
# SEED FIXING
# ===============================================================

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

fix_seed(SEED)

# ===============================================================
# LOGGING
# ===============================================================

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()

# ===============================================================
# HASH UTILITIES
# ===============================================================

def compute_hash(lines: List[str]) -> str:
    m = hashlib.md5()
    for s in lines:
        m.update(s.encode("utf-8"))
        m.update(b"\n")
    return m.hexdigest()

# ===============================================================
# DATA LOADING & CLEANING
# ===============================================================

def read_pairs(path: str) -> List[Tuple[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "++++$++++" not in line:
                continue
            s, t = line.strip().split("++++$++++")
            pairs.append((s.strip(), t.strip()))
    return pairs


def clean_text(s: str) -> str:
    s = "".join(ch for ch in s if ch.isprintable())
    s = s.replace("\u200b", "")
    s = " ".join(s.split())
    return s


def is_telugu(s: str) -> bool:
    total = len(s)
    if total == 0:
        return False
    tel = sum(1 for ch in s if 0x0C00 <= ord(ch) <= 0x0C7F)
    return tel / total >= 0.25


def clean_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    out = []
    seen = set()

    for s, t in pairs:
        s = clean_text(s)
        t = clean_text(t)

        if not s or not t:
            continue
        if not is_telugu(t):
            continue
        if len(s.split()) > MAX_SRC_LEN:
            continue
        if len(t.split()) > MAX_TGT_LEN:
            continue

        key = (s.lower(), t)
        if key in seen:
            continue
        seen.add(key)
        out.append((s, t))

    log(f"After cleaning: {len(out)} pairs")
    return out

# ===============================================================
# SENTENCEPIECE TRAINING / LOADING
# ===============================================================

def write_list(path: str, lst: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        for x in lst:
            f.write(x + "\n")


def train_sp(input_file: str, prefix: str, vocab_size: int):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        hard_vocab_limit=False,
    )
    sp = spm.SentencePieceProcessor()
    sp.load(prefix + ".model")
    return sp


def load_or_train_sp(src_texts, tgt_texts):
    combined_hash = compute_hash(src_texts + tgt_texts)
    combined_hash += f"|{SP_SRC_VOCAB}|{SP_TGT_VOCAB}"

    if (
        os.path.exists(SP_HASH_CACHE)
        and os.path.exists(SP_SRC_PREFIX + ".model")
        and os.path.exists(SP_TGT_PREFIX + ".model")
    ):
        with open(SP_HASH_CACHE, "r") as f:
            old_hash = f.read().strip()
        if old_hash == combined_hash:
            log("Reusing SentencePiece models")
            sp_src = spm.SentencePieceProcessor()
            sp_tgt = spm.SentencePieceProcessor()
            sp_src.load(SP_SRC_PREFIX + ".model")
            sp_tgt.load(SP_TGT_PREFIX + ".model")
            return sp_src, sp_tgt

    log("Training SentencePiece...")
    write_list("tmp_src.txt", src_texts)
    write_list("tmp_tgt.txt", tgt_texts)

    sp_src = train_sp("tmp_src.txt", SP_SRC_PREFIX, SP_SRC_VOCAB)
    sp_tgt = train_sp("tmp_tgt.txt", SP_TGT_PREFIX, SP_TGT_VOCAB)

    os.remove("tmp_src.txt")
    os.remove("tmp_tgt.txt")

    with open(SP_HASH_CACHE, "w") as f:
        f.write(combined_hash)

    return sp_src, sp_tgt

# ===============================================================
# TRANSFORMER STUDENT MODEL (6-LAYER)
# ===============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=768,
        n_heads=8,
        n_layers=6,
        ff_dim=3072,
        dropout=0.2,
        pad_id=0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

    def forward(self, src, src_key_padding_mask):
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_enc(x)
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=768,
        n_heads=8,
        n_layers=6,
        ff_dim=3072,
        dropout=0.2,
        pad_id=0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask,
        tgt_key_padding_mask,
        memory_key_padding_mask,
    ):
        x = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_enc(x)

        return self.decoder(
            x,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab,
        tgt_vocab,
        d_model=768,
        n_heads=8,
        n_layers=6,
        ff_dim=3072,
        dropout=0.2,
        pad_id=0,
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            src_vocab, d_model, n_heads, n_layers, ff_dim, dropout, pad_id
        )

        self.decoder = TransformerDecoder(
            tgt_vocab, d_model, n_heads, n_layers, ff_dim, dropout, pad_id
        )

        self.output_proj = nn.Linear(d_model, tgt_vocab, bias=False)

        # Weight tying
        self.output_proj.weight = self.decoder.embedding.weight

        self.pad_id = pad_id
        self.d_model = d_model

    def make_src_key_padding_mask(self, src):
        return src == self.pad_id

    def make_tgt_key_padding_mask(self, tgt):
        return tgt == self.pad_id

    def make_causal_mask(self, size, device):
        return torch.triu(
            torch.ones(size, size, device=device), diagonal=1
        ).bool()

    def forward(self, src, tgt_in):
        src_mask = self.make_src_key_padding_mask(src)
        tgt_mask = self.make_tgt_key_padding_mask(tgt_in)

        causal_mask = self.make_causal_mask(tgt_in.size(1), tgt_in.device)

        memory = self.encoder(src, src_mask)

        dec_out = self.decoder(
            tgt_in,
            memory,
            causal_mask,
            tgt_mask,
            src_mask,
        )

        logits = self.output_proj(dec_out)
        return logits

# ===============================================================
# TOKENIZATION & DATA PREPARATION
# ===============================================================

def sp_encode(sp: spm.SentencePieceProcessor, text: str, max_len: int) -> List[int]:
    raw_ids = sp.encode(text, out_type=int)
    # Pyre strict typing fallback
    if not isinstance(raw_ids, list):
        raw_ids = list(raw_ids)
    
    bos: int = int(sp.bos_id())
    eos: int = int(sp.eos_id())
    ids: List[int] = [bos] + raw_ids + [eos]

    if len(ids) > max_len:
        ids_trunc: List[int] = [ids[i] for i in range(max_len)]
        ids = ids_trunc
        ids[-1] = eos

    return ids


def pad_sequences(list_of_ids: List[List[int]], max_len: int) -> torch.Tensor:
    arr = torch.full((len(list_of_ids), max_len), PAD_ID, dtype=torch.long)
    for i, seq in enumerate(list_of_ids):
        arr[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return arr


class SeqDataset(Dataset):
    def __init__(self, src_arr: torch.Tensor, tgt_in_arr: torch.Tensor, tgt_out_arr: torch.Tensor):
        self.src_arr = src_arr
        self.tgt_in_arr = tgt_in_arr
        self.tgt_out_arr = tgt_out_arr

    def __len__(self) -> int:
        return len(self.src_arr)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.src_arr[idx],
            self.tgt_in_arr[idx],
            self.tgt_out_arr[idx],
        )


def prepare_datasets(pairs: List[Tuple[str, str]], sp_src: spm.SentencePieceProcessor, sp_tgt: spm.SentencePieceProcessor) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], List[str], torch.Tensor, torch.Tensor]:

    random.shuffle(pairs)

    src_all = [s for s, _ in pairs]
    tgt_all = [t for _, t in pairs]

    N = len(pairs)
    val_size = min(10000, max(1000, int(0.07 * N)))
    train_size = N - val_size

    train_src: List[str] = [src_all[i] for i in range(train_size)]
    train_tgt: List[str] = [tgt_all[i] for i in range(train_size)]
    val_src: List[str] = [src_all[i] for i in range(train_size, N)]
    val_tgt: List[str] = [tgt_all[i] for i in range(train_size, N)]

    log(f"Train: {len(train_src)} | Val: {len(val_src)}")

    # Encode training
    train_src_ids: List[List[int]] = []
    train_tin_ids: List[List[int]] = []
    train_tout_ids: List[List[int]] = []
    
    val_src_ids: List[List[int]] = []
    val_tin_ids: List[List[int]] = []
    val_tout_ids: List[List[int]] = []

    max_sl, max_tl = 0, 0

    for s, t in zip(train_src, train_tgt):
        s_i = sp_encode(sp_src, s, MAX_SRC_LEN)
        t_i = sp_encode(sp_tgt, t, MAX_TGT_LEN)

        t_in: List[int] = [t_i[k] for k in range(len(t_i) - 1)]
        t_out: List[int] = [t_i[k] for k in range(1, len(t_i))]

        train_src_ids.append(s_i)
        train_tin_ids.append(t_in)
        train_tout_ids.append(t_out)

        max_sl = max(max_sl, len(s_i))
        max_tl = max(max_tl, len(t_i) - 1)

    for s, t in zip(val_src, val_tgt):
        s_i = sp_encode(sp_src, s, MAX_SRC_LEN)
        t_i = sp_encode(sp_tgt, t, MAX_TGT_LEN)

        t_in: List[int] = [t_i[k] for k in range(len(t_i) - 1)]
        t_out: List[int] = [t_i[k] for k in range(1, len(t_i))]

        val_src_ids.append(s_i)
        val_tin_ids.append(t_in)
        val_tout_ids.append(t_out)

        max_sl = max(max_sl, len(s_i))
        max_tl = max(max_tl, len(t_i) - 1)

    train_src_arr = pad_sequences(train_src_ids, max_sl)
    train_tin_arr = pad_sequences(train_tin_ids, max_tl)
    train_tout_arr = pad_sequences(train_tout_ids, max_tl)

    val_src_arr = pad_sequences(val_src_ids, max_sl)
    val_tin_arr = pad_sequences(val_tin_ids, max_tl)
    val_tout_arr = pad_sequences(val_tout_ids, max_tl)

    train_ds = SeqDataset(train_src_arr, train_tin_arr, train_tout_arr)
    val_ds = SeqDataset(val_src_arr, val_tin_arr, val_tout_arr)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(DEVICE == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=(DEVICE == "cuda"),
    )

    # Distillation loader (smaller batch)
    distill_loader = DataLoader(
        train_ds,
        batch_size=DISTILL_BATCH_SIZE,
        shuffle=True,
        pin_memory=(DEVICE == "cuda"),
    )

    return (
        train_loader,
        val_loader,
        distill_loader,
        val_src,
        val_tgt,
        train_src_arr,
        val_src_arr,
    )

# ===============================================================
# MULTI-TEACHER LOADING
# ===============================================================

def load_teacher_nllb():
    log("Loading NLLB teacher...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_1)
    model = AutoModelForSeq2SeqLM.from_pretrained(TEACHER_1).to(DEVICE)
    model.eval()
    return tokenizer, model


def load_teacher_indic():
    log("Loading IndicTrans2 teacher...")
    tokenizer = AutoTokenizer.from_pretrained(
        TEACHER_2,
        trust_remote_code=True,
    )
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "tel_Telu"

    model = AutoModelForSeq2SeqLM.from_pretrained(
        TEACHER_2,
        trust_remote_code=True,
    ).to(DEVICE)

    model.eval()
    return tokenizer, model


# ===============================================================
# MULTI-TEACHER LOGIT AGGREGATION (EQUAL WEIGHT)
# ===============================================================

@torch.no_grad()
def compute_teacher_logits(
    src_batch,
    tgt_in_batch,
    teacher1,
    teacher2,
):
    tok1, model1 = teacher1
    tok2, model2 = teacher2

    # Teacher 1 (NLLB)
    logits1 = model1(
        input_ids=src_batch,
        attention_mask=(src_batch != PAD_ID),
        decoder_input_ids=tgt_in_batch,
    ).logits.detach()

    # Teacher 2 (IndicTrans2)
    logits2 = model2(
        input_ids=src_batch,
        attention_mask=(src_batch != PAD_ID),
        decoder_input_ids=tgt_in_batch,
    ).logits.detach()

    # Equal weighting
    teacher_logits = (
        TEACHER_WEIGHT_1 * logits1 +
        TEACHER_WEIGHT_2 * logits2
    )

    return teacher_logits

# ===============================================================
# TRAINING UTILITIES
# ===============================================================

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    B, T, V = logits.shape
    return nn.functional.cross_entropy(
        logits.view(B*T, V),
        targets.view(B*T),
        ignore_index=PAD_ID,
        label_smoothing=0.1,
    )


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    B, T, V = student_logits.shape

    # Hard CE
    ce = nn.functional.cross_entropy(
        student_logits.view(B*T, V),
        targets.view(B*T),
        ignore_index=PAD_ID,
        label_smoothing=0.1,
    )

    # Soft KL
    student_log_probs = nn.functional.log_softmax(
        student_logits / TEMPERATURE,
        dim=-1,
    )

    teacher_probs = nn.functional.softmax(
        teacher_logits / TEMPERATURE,
        dim=-1,
    )

    kl = nn.functional.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (TEMPERATURE ** 2)

    return ALPHA * kl + (1 - ALPHA) * ce


# ===============================================================
# PHASE 1 — SUPERVISED TRAINING
# ===============================================================

def train_phase1(model, train_loader, val_loader):

    log("Starting Phase 1 — Supervised Training")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_PHASE1,
        betas=(0.9, 0.98),
        weight_decay=0.01,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS_PHASE1 + 1):

        model.train()
        total_loss = 0

        for src_b, tin_b, tout_b in tqdm(train_loader, desc=f"P1 Epoch {epoch}"):

            src_b = src_b.to(DEVICE)
            tin_b = tin_b.to(DEVICE)
            tout_b = tout_b.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(DEVICE == "cuda")):
                logits = model(src_b, tin_b)
                loss = cross_entropy_loss(logits.float(), tout_b)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # Validation
        val_loss = evaluate_loss(model, val_loader)

        scheduler.step(val_loss)

        log(f"P1 Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < float(best_val_loss) - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "student_phase1_best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                log("Early stopping Phase 1")
                break

    model.load_state_dict(torch.load("student_phase1_best.pt"))
    return model


# ===============================================================
# PHASE 2 — MULTI-TEACHER DISTILLATION
# ===============================================================

def train_phase2(model, distill_loader, teacher1, teacher2):

    log("Starting Phase 2 — Multi-Teacher Distillation")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_PHASE2,
        betas=(0.9, 0.98),
        weight_decay=0.01,
    )

    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    for epoch in range(1, EPOCHS_PHASE2 + 1):

        model.train()
        total_loss = 0

        for src_b, tin_b, tout_b in tqdm(distill_loader, desc=f"P2 Epoch {epoch}"):

            src_b = src_b.to(DEVICE)
            tin_b = tin_b.to(DEVICE)
            tout_b = tout_b.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            # Compute teacher logits
            teacher_logits = compute_teacher_logits(
                src_b,
                tin_b,
                teacher1,
                teacher2,
            )

            with autocast(enabled=(DEVICE == "cuda")):
                student_logits = model(src_b, tin_b)
                loss = distillation_loss(
                    student_logits.float(),
                    teacher_logits.float(),
                    tout_b,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        log(f"P2 Epoch {epoch} | Loss: {total_loss / len(distill_loader):.4f}")

        torch.save(model.state_dict(), "student_phase2_latest.pt")

    return model


# ===============================================================
# PHASE 3 — HUMAN FINE-TUNING
# ===============================================================

def train_phase3(model, train_loader):

    log("Starting Phase 3 — Human Fine-Tuning")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_PHASE3,
    )

    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    for epoch in range(1, EPOCHS_PHASE3 + 1):

        model.train()
        total_loss = 0

        for src_b, tin_b, tout_b in tqdm(train_loader, desc=f"P3 Epoch {epoch}"):

            src_b = src_b.to(DEVICE)
            tin_b = tin_b.to(DEVICE)
            tout_b = tout_b.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(DEVICE == "cuda")):
                logits = model(src_b, tin_b)
                loss = cross_entropy_loss(logits.float(), tout_b)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        log(f"P3 Epoch {epoch} | Loss: {total_loss / len(train_loader):.4f}")

    return model


# ===============================================================
# LOSS EVALUATION
# ===============================================================

@torch.no_grad()
def evaluate_loss(model, loader):
    model.eval()
    total = 0

    for src_b, tin_b, tout_b in loader:

        src_b = src_b.to(DEVICE)
        tin_b = tin_b.to(DEVICE)
        tout_b = tout_b.to(DEVICE)

        logits = model(src_b, tin_b)
        loss = cross_entropy_loss(logits.float(), tout_b)

        total += loss.item()

    return total / len(loader)


# ===============================================================
# BEAM SEARCH DECODING
# ===============================================================

@torch.no_grad()
def beam_search_decode(model, sp_tgt, src_seq_np):
    model.eval()

    src = torch.tensor(src_seq_np, dtype=torch.long, device=DEVICE).unsqueeze(0)
    src_mask = src == PAD_ID

    memory = model.encoder(src, src_mask)

    bos = sp_tgt.bos_id()
    eos = sp_tgt.eos_id()

    beams = [(0.0, [bos])]

    for _ in range(MAX_TGT_LEN):

        new_beams = []

        for score, seq in beams:

            if seq[-1] == eos:
                new_beams.append((score, seq))
                continue

            tgt = torch.tensor(seq, device=DEVICE).unsqueeze(0)

            tgt_mask = model.make_tgt_key_padding_mask(tgt)
            causal_mask = model.make_causal_mask(tgt.size(1), DEVICE)

            dec_out = model.decoder(
                tgt,
                memory,
                causal_mask,
                tgt_mask,
                src_mask,
            )

            logits = model.output_proj(dec_out[:, -1])
            log_probs = torch.log_softmax(logits, dim=-1)

            topk = torch.topk(log_probs, BEAM_SIZE)

            for i in range(BEAM_SIZE):
                tok = topk.indices[0, i].item()
                new_score = score + topk.values[0, i].item()
                new_beams.append((new_score, seq + [tok]))

        def get_score(x: Tuple[float, List[int]]) -> float:
            return x[0] / (len(x[1]) ** LENGTH_NORM_ALPHA)
            
        sorted_beams = sorted(
            new_beams,
            key=get_score,
            reverse=True,
        )
        
        out_beams: List[Tuple[float, List[int]]] = []
        for i in range(min(BEAM_SIZE, len(sorted_beams))):
            out_beams.append(sorted_beams[i])
        beams = out_beams

    best_seq: List[int] = beams[0][1]
    best_len: int = len(best_seq)
    best: List[int] = [best_seq[i] for i in range(1, best_len)]

    if eos in best:
        eos_idx = best.index(eos)
        best_truncated: List[int] = [best[i] for i in range(eos_idx)]
        best = best_truncated

    return sp_tgt.decode(best)


# ===============================================================
# BLEU EVALUATION
# ===============================================================

def evaluate_bleu(model, sp_tgt, val_src_arr, val_refs, n_eval=500):

    preds = []

    for i in tqdm(range(min(n_eval, len(val_src_arr))), desc="Decoding"):
        pred = beam_search_decode(model, sp_tgt, val_src_arr[i])
        preds.append(pred)

    trunc_refs: List[str] = [val_refs[i] for i in range(len(preds))]
    bleu = sacrebleu.corpus_bleu(preds, [trunc_refs])
    chrf = sacrebleu.corpus_chrf(preds, [trunc_refs])

    log(f"BLEU: {bleu.score:.2f} | chrF: {chrf.score:.2f}")

    return bleu.score, chrf.score


# ===============================================================
# MAIN PIPELINE
# ===============================================================

def main():

    log("Loading dataset...")
    raw_pairs = read_pairs(DATA_PATH)
    pairs = clean_pairs(raw_pairs)

    src_texts = [s for s, _ in pairs]
    tgt_texts = [t for _, t in pairs]

    # Train SentencePiece
    sp_src, sp_tgt = load_or_train_sp(src_texts, tgt_texts)

    # Prepare datasets
    (
        train_loader,
        val_loader,
        distill_loader,
        val_src_text,
        val_refs,
        train_src_arr,
        val_src_arr,
    ) = prepare_datasets(pairs, sp_src, sp_tgt)

    Vsrc = sp_src.get_piece_size()
    Vtgt = sp_tgt.get_piece_size()

    # Build student model
    model = TransformerSeq2Seq(
        Vsrc,
        Vtgt,
        d_model=768,
        n_heads=8,
        n_layers=6,
        ff_dim=3072,
        dropout=0.2,
        pad_id=PAD_ID,
    ).to(DEVICE)

    # Phase 1
    model = train_phase1(model, train_loader, val_loader)

    log("Evaluating after Phase 1...")
    evaluate_bleu(model, sp_tgt, val_src_arr, val_refs)

    # Load teachers
    teacher1 = load_teacher_nllb()
    teacher2 = load_teacher_indic()

    # Phase 2
    model = train_phase2(model, distill_loader, teacher1, teacher2)

    log("Evaluating after Phase 2...")
    evaluate_bleu(model, sp_tgt, val_src_arr, val_refs)

    # Phase 3
    model = train_phase3(model, train_loader)

    log("Final Evaluation...")
    evaluate_bleu(model, sp_tgt, val_src_arr, val_refs)

    log("Training complete.")


if __name__ == "__main__":
    main()