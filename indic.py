# main.py
# ===============================================================
# English -> Telugu NMT (Teacher: Indic, Student: TRANSFORMER)
#
# FULL RESEARCH-GRADE PIPELINE

import os
import sys
import time
import random
import hashlib
from typing import List, Tuple
import math

import numpy as np
from tqdm import tqdm
import sacrebleu

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sentencepiece as spm

# ===============================================================
# CONFIG
# ===============================================================

DATA_PATH = "english-telugu.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

SEED = 42

# -------------------------------
# Teacher (NLLB)
# -------------------------------
TEACHER_MODEL = "ai4bharat/indictrans2-en-indic-1B"
TEACHER_BATCH = 32
TEACHER_MAX_LEN = 64
TEACHER_BEAMS = 1

# -------------------------------
# Student training
# -------------------------------
EPOCHS = 50
BATCH_SIZE = 128

LR = 2e-4
GRAD_CLIP = 1.0

# Scheduled teacher forcing (kept for compatibility)
BASE_TEACHER_FORCING = 0.8
MIN_TEACHER_FORCING = 0.4
TF_DECAY = 0.97

# -------------------------------
# SentencePiece
# -------------------------------
SP_SRC_VOCAB = 12000
SP_TGT_VOCAB = 16000

SP_SRC_PREFIX = "sp_src"
SP_TGT_PREFIX = "sp_tgt"

MAX_SRC_LEN = 64
MAX_TGT_LEN = 100

PAD_ID = 0

# -------------------------------
# Evaluation
# -------------------------------
VAL_SAMPLES_FOR_BLEU = 500

# -------------------------------
# Early stopping
# -------------------------------
PATIENCE = 5

# -------------------------------
# Cache paths
# -------------------------------
TEACHER_OUT_PATH = "teacher_outputs.txt"
TEACHER_HASH_PATH = "teacher_hash.txt"
SP_HASH_PATH = "sp_hash.txt"

# -------------------------------
# Beam search (used later)
# -------------------------------
BEAM_SIZE = 5
LENGTH_NORM_ALPHA = 0.7

# -------------------------------
# Distillation (Phase 2)
# -------------------------------
DISTILL_TEMPERATURE = 2.0
DISTILL_ALPHA = 1 
DISTILL_EPOCHS = 40

TEACHER_BATCH_DISTILL = 8  # MUST be small
DISTILL_BATCH_SIZE = 8  # IMPORTANT: keep this small for IndicTrans2

# ===============================================================
# SEED / DETERMINISM
# ===============================================================

def fix_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

fix_seed()

# ===============================================================
# LOGGING
# ===============================================================

def log(msg: str):
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
# DATA LOADING + CLEANING
# ===============================================================

def read_pairs(path: str) -> List[Tuple[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    eng, tel = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "++++$++++" not in line:
                continue
            a, b = line.strip().split("++++$++++")
            eng.append(a.strip())
            tel.append(b.strip())
    return list(zip(eng, tel))


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
# TEACHER TRANSLATION CACHING (NLLB)
# ===============================================================

def load_or_run_teacher(src_texts: List[str]) -> Tuple[List[str], str]:
    """
    Returns teacher translations and hash.
    Uses cache if hash + length match.
    IndicTrans2 teacher (English -> Telugu).
    """
    current_hash = compute_hash(src_texts)

    # -----------------------------
    # Cache check
    # -----------------------------
    if os.path.exists(TEACHER_OUT_PATH) and os.path.exists(TEACHER_HASH_PATH):
        with open(TEACHER_HASH_PATH, "r", encoding="utf-8") as f:
            old_hash = f.read().strip()

        if old_hash == current_hash:
            with open(TEACHER_OUT_PATH, "r", encoding="utf-8") as f:
                outs = [x.rstrip("\n") for x in f]

            if len(outs) == len(src_texts):
                log("Reusing cached teacher translations.")
                return outs, current_hash
            else:
                log("Teacher cache length mismatch, recomputing.")

    # -----------------------------
    # Run IndicTrans2 teacher
    # -----------------------------
    log("Running IndicTrans2 teacher translations...")

    tokenizer = AutoTokenizer.from_pretrained(
        TEACHER_MODEL,
        trust_remote_code=True,
    )
    tokenizer.src_lang = "en"
    tokenizer.tgt_lang = "te"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        TEACHER_MODEL,
        trust_remote_code=True,
    ).to(DEVICE)

    model.eval()
    translations = []

    for i in tqdm(
        range(0, len(src_texts), TEACHER_BATCH),
        desc="Teacher decoding",
    ):
        batch = src_texts[i:i + TEACHER_BATCH]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=TEACHER_MAX_LEN,
        ).to(DEVICE)

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_length=TEACHER_MAX_LEN,
                num_beams=TEACHER_BEAMS,
                do_sample=False,
            )

        dec = tokenizer.batch_decode(gen, skip_special_tokens=True)
        translations.extend(dec)

    assert len(translations) == len(src_texts)

    # -----------------------------
    # Save cache
    # -----------------------------
    with open(TEACHER_OUT_PATH, "w", encoding="utf-8") as f:
        for line in translations:
            f.write(line.replace("\n", " ") + "\n")

    with open(TEACHER_HASH_PATH, "w", encoding="utf-8") as f:
        f.write(current_hash)

    log("Saved teacher outputs + hash.")
    return translations, current_hash

def load_teacher_for_logits():
    # Use the specific IndicTrans2 model name
    tokenizer = AutoTokenizer.from_pretrained(  
        TEACHER_MODEL,
        trust_remote_code=True,
    )
    
    # IndicTrans2 needs these specific attributes to not get confused
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "tel_Telu"
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        TEACHER_MODEL,
        trust_remote_code=True,
    ).to(DEVICE)

    model.eval()
    return tokenizer, model
# ===============================================================
# SENTENCEPIECE UTILITIES
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
        user_defined_symbols=[]
    )
    sp = spm.SentencePieceProcessor()
    sp.load(prefix + ".model")
    return sp


def load_or_train_sp(
    src_texts: List[str],
    teacher_tgt_texts: List[str],
    data_hash: str,
):
    """
    Train or reuse SentencePiece models.
    Cache key includes data hash + vocab sizes.
    """
    src_model_path = SP_SRC_PREFIX + ".model"
    tgt_model_path = SP_TGT_PREFIX + ".model"

    combined_hash = data_hash + f"|src_vocab={SP_SRC_VOCAB}|tgt_vocab={SP_TGT_VOCAB}"

    if (
        os.path.exists(src_model_path)
        and os.path.exists(tgt_model_path)
        and os.path.exists(SP_HASH_PATH)
    ):
        with open(SP_HASH_PATH, "r", encoding="utf-8") as f:
            old_hash = f.read().strip()

        if old_hash == combined_hash:
            sp_src = spm.SentencePieceProcessor()
            sp_tgt = spm.SentencePieceProcessor()
            sp_src.load(src_model_path)
            sp_tgt.load(tgt_model_path)
            log("Reusing cached SentencePiece models.")
            return sp_src, sp_tgt

    # Train new SentencePiece models
    log("Training SentencePiece models from scratch...")
    tmp_src = "tmp_sp_src.txt"
    tmp_tgt = "tmp_sp_tgt.txt"

    write_list(tmp_src, src_texts)
    write_list(tmp_tgt, teacher_tgt_texts)

    sp_src = train_sp(tmp_src, SP_SRC_PREFIX, SP_SRC_VOCAB)
    sp_tgt = train_sp(tmp_tgt, SP_TGT_PREFIX, SP_TGT_VOCAB)

    os.remove(tmp_src)
    os.remove(tmp_tgt)

    with open(SP_HASH_PATH, "w", encoding="utf-8") as f:
        f.write(combined_hash)

    log("Saved new SentencePiece models + hash.")
    return sp_src, sp_tgt


def sp_encode(sp, text: str, max_len: int) -> List[int]:
    ids = sp.encode(text, out_type=int)
    ids = [sp.bos_id()] + ids + [sp.eos_id()]

    if len(ids) > max_len:
        ids = ids[:max_len]
        ids[-1] = sp.eos_id()

    return ids


def pad_sequences(
    list_of_ids: List[List[int]],
    max_len: int,
    pad_id: int = PAD_ID,
) -> np.ndarray:
    arr = np.full((len(list_of_ids), max_len), pad_id, dtype=np.int64)
    for i, seq in enumerate(list_of_ids):
        arr[i, :len(seq)] = seq
    return arr

# ===============================================================
# DATASET
# ===============================================================

class SeqDataset(Dataset):
    def __init__(
        self,
        src_arr: np.ndarray,
        tgt_in_arr: np.ndarray,
        tgt_out_arr: np.ndarray,
    ):
        self.src_arr = src_arr
        self.tgt_in_arr = tgt_in_arr
        self.tgt_out_arr = tgt_out_arr

    def __len__(self):
        return len(self.src_arr)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.src_arr[idx], dtype=torch.long),
            torch.tensor(self.tgt_in_arr[idx], dtype=torch.long),
            torch.tensor(self.tgt_out_arr[idx], dtype=torch.long),
        )

# ===============================================================
# TRANSFORMER STUDENT MODEL
# ===============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_dim: int,
        dropout: float,
        pad_id: int,
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
            norm_first=True,  # PRE-NORM for stability
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor):
        emb = self.embedding(src) * np.sqrt(self.embedding.embedding_dim)
        emb = self.pos_enc(emb)

        memory = self.encoder(
            emb,
            src_key_padding_mask=src_key_padding_mask,
        )
        return memory


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_dim: int,
        dropout: float,
        pad_id: int,
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
            norm_first=True,  # PRE-NORM
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ):
        emb = self.embedding(tgt) * np.sqrt(self.embedding.embedding_dim)
        emb = self.pos_enc(emb)

        out = self.decoder(
            emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return out


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 6,
        ff_dim: int = 3072,
        dropout: float = 0.3,
        pad_id: int = PAD_ID,
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            src_vocab,
            d_model,
            n_heads,
            n_layers,
            ff_dim,
            dropout,
            pad_id,
        )

        self.decoder = TransformerDecoder(
            tgt_vocab,
            d_model,
            n_heads,
            n_layers,
            ff_dim,
            dropout,
            pad_id,
        )

        self.output_proj = nn.Linear(d_model, tgt_vocab, bias=False)

        # WEIGHT TYING (decoder embedding ↔ output projection)
        self.output_proj.weight = self.decoder.embedding.weight

        self.pad_id = pad_id
        self.d_model = d_model

    def make_src_key_padding_mask(self, src: torch.Tensor):
        return (src == self.pad_id)

    def make_tgt_key_padding_mask(self, tgt: torch.Tensor):
        return (tgt == self.pad_id)

    def make_causal_mask(self, tgt_len: int, device):
        return torch.triu(
            torch.ones(tgt_len, tgt_len, device=device), diagonal=1
        ).bool()

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_mask=None,
        teacher_forcing_ratio: float = 1.0,
    ):
        src_key_padding = self.make_src_key_padding_mask(src)
        tgt_key_padding = self.make_tgt_key_padding_mask(tgt_in)

        tgt_len = tgt_in.size(1)
        tgt_mask = self.make_causal_mask(tgt_len, tgt_in.device)

        memory = self.encoder(src, src_key_padding)

        dec_out = self.decoder(
            tgt_in,
            memory,
            tgt_mask,
            tgt_key_padding,
            src_key_padding,
        )

        logits = self.output_proj(dec_out)
        return logits

# ===============================================================
# GREEDY DECODING (TRANSFORMER)
# ===============================================================

@torch.no_grad()
def greedy_decode_transformer(model, sp_tgt, src_seq_np):
    model.eval()

    src = torch.tensor(src_seq_np, dtype=torch.long, device=DEVICE).unsqueeze(0)
    src_key_padding = (src == PAD_ID)

    memory = model.encoder(src, src_key_padding)

    bos = sp_tgt.bos_id()
    eos = sp_tgt.eos_id()

    tgt = torch.tensor([[bos]], device=DEVICE)

    for _ in range(MAX_TGT_LEN):
        tgt_key_padding = (tgt == PAD_ID)
        tgt_mask = model.make_causal_mask(tgt.size(1), DEVICE)

        dec_out = model.decoder(
            tgt,
            memory,
            tgt_mask,
            tgt_key_padding,
            src_key_padding,
        )

        logits = model.output_proj(dec_out[:, -1])
        next_tok = logits.argmax(dim=-1).item()

        tgt = torch.cat([tgt, torch.tensor([[next_tok]], device=DEVICE)], dim=1)

        if next_tok == eos:
            break

    toks = tgt[0, 1:].tolist()
    if eos in toks:
        toks = toks[:toks.index(eos)]

    return sp_tgt.decode(toks)

# ===============================================================
# TRANSFORMER BEAM SEARCH (LENGTH-NORMALIZED)
# ===============================================================

@torch.no_grad()
def transformer_beam_search_decode(
    model,
    sp_tgt,
    src_seq_np,
    beam_size=BEAM_SIZE,
    alpha=LENGTH_NORM_ALPHA,
):
    model.eval()

    src = torch.tensor(src_seq_np, dtype=torch.long, device=DEVICE).unsqueeze(0)
    src_key_padding = (src == PAD_ID)

    memory = model.encoder(src, src_key_padding)

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
            tgt_key_padding = (tgt == PAD_ID)
            tgt_mask = model.make_causal_mask(tgt.size(1), DEVICE)

            dec_out = model.decoder(
                tgt,
                memory,
                tgt_mask,
                tgt_key_padding,
                src_key_padding,
            )

            logits = model.output_proj(dec_out[:, -1])
            log_probs = torch.log_softmax(logits, dim=-1)

            topk = torch.topk(log_probs, beam_size, dim=-1)

            for i in range(beam_size):
                tok = topk.indices[0, i].item()
                new_score = score + topk.values[0, i].item()
                new_beams.append((new_score, seq + [tok]))

        beams = sorted(
            new_beams,
            key=lambda x: x[0] / (len(x[1]) ** alpha),
            reverse=True,
        )[:beam_size]

    best = beams[0][1][1:]
    if eos in best:
        best = best[:best.index(eos)]

    return sp_tgt.decode(best)

# ===============================================================
# TRAINING / EVALUATION UTILITIES
# ===============================================================

def get_teacher_forcing_ratio(epoch: int) -> float:
    ratio = BASE_TEACHER_FORCING * (TF_DECAY ** (epoch - 1))
    return max(ratio, MIN_TEACHER_FORCING)


def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch):
    model.train()
    total_loss = 0.0
    tf_ratio = get_teacher_forcing_ratio(epoch)
    log(f"Teacher forcing ratio: {tf_ratio:.3f}")

    use_amp = (DEVICE == "cuda")

    for src_b, tin_b, tout_b in tqdm(loader, desc="Train", leave=False):
        src_b = src_b.to(DEVICE)
        tin_b = tin_b.to(DEVICE)
        tout_b = tout_b.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(src_b, tin_b)
            logits = logits.float()
            B, T, V = logits.shape
            loss = criterion(
                logits.view(B * T, V),
                tout_b.view(B * T),
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)

def train_one_epoch_distill(
    student,
    teacher,
    loader,
    optimizer,
    scaler,
):
    student.train()
    teacher.eval()
    total_loss = 0.0
    use_amp = (DEVICE == "cuda")

    for src_b, tin_b, tout_b in tqdm(loader, desc="Distill", leave=False):
        src_b = src_b.to(DEVICE)
        tin_b = tin_b.to(DEVICE)
        tout_b = tout_b.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        # ---- Student forward
        with autocast(enabled=use_amp):
            student_logits = student(src_b, tin_b)

        # ---- Teacher forward (NO AMP, NO GRAD)
        with torch.no_grad():
            teacher_logits = teacher(
                input_ids=src_b,
                attention_mask=(src_b != PAD_ID),
                decoder_input_ids=tin_b,
            ).logits.detach()

        with autocast(enabled=use_amp):
            loss = distillation_loss(
                student_logits.float(),
                teacher_logits.float(),
                tout_b,
                PAD_ID,
                DISTILL_ALPHA,
                DISTILL_TEMPERATURE,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # 🔥 VERY IMPORTANT
        del teacher_logits
        torch.cuda.empty_cache()

    return total_loss / len(loader)

@torch.no_grad()
def eval_loss(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    for src_b, tin_b, tout_b in tqdm(loader, desc="Val Loss", leave=False):
        src_b = src_b.to(DEVICE)
        tin_b = tin_b.to(DEVICE)
        tout_b = tout_b.to(DEVICE)

        logits = model(src_b, tin_b)
        logits = logits.float()

        B, T, V = logits.shape
        loss = criterion(
            logits.view(B * T, V),
            tout_b.view(B * T),
        )
        total_loss += loss.item()

    return total_loss / len(loader)

def distillation_loss(
    student_logits,
    teacher_logits,
    targets,
    pad_id,
    alpha,
    temperature,
):
    """
    student_logits: [B, T, V]
    teacher_logits: [B, T, V]
    targets:        [B, T]
    """
    B, T, V = student_logits.shape

    # ---- CE loss (hard targets)
    ce_loss = nn.functional.cross_entropy(
        student_logits.view(B*T, V),
        targets.view(B*T),
        ignore_index=pad_id,
        label_smoothing=0.1,
    )

    return ce_loss

@torch.no_grad()
def beam_search_decode_val(
    model,
    sp_tgt,
    src_arr,
    n_eval,
):
    preds = []
    for i in tqdm(range(n_eval), desc="Beam decode val", leave=False):
        pred = transformer_beam_search_decode(
            model,
            sp_tgt,
            src_arr[i],
            beam_size=BEAM_SIZE,
            alpha=LENGTH_NORM_ALPHA,
        )
        preds.append(pred)
    return preds


# ===============================================================
# MAIN
# ===============================================================

def main():
    # -----------------------------
    # Load + clean data
    # -----------------------------
    log("Loading data...")
    raw_pairs = read_pairs(DATA_PATH)
    pairs = clean_pairs(raw_pairs)

    if len(pairs) == 0:
        raise RuntimeError("No usable pairs after cleaning.")

    random.shuffle(pairs)
    src_all = [s for s, _ in pairs]
    tgt_human_all = [t for _, t in pairs]

    N = len(pairs)
    val_size = min(10000, max(1000, int(0.07 * N)))
    train_size = N - val_size

    train_src = src_all[:train_size]
    train_tgt_human = tgt_human_all[:train_size]
    val_src = src_all[train_size:]
    val_tgt_human = tgt_human_all[train_size:]

    log(f"Total pairs: {N} | Train: {train_size} | Val: {val_size}")

    # -----------------------------
    # Teacher translations
    # -----------------------------
    teacher_out_all, teacher_hash = load_or_run_teacher(src_all)
    train_teacher = teacher_out_all[:train_size]
    val_teacher = teacher_out_all[train_size:]

    # -----------------------------
    # Teacher BLEU / chrF
    # -----------------------------
    n_teacher_eval = min(0, len(val_teacher))
    if n_teacher_eval > 0:
        log(f"Evaluating teacher on {n_teacher_eval} samples...")
        bleu = sacrebleu.corpus_bleu(
            val_teacher[:n_teacher_eval],
            [val_tgt_human[:n_teacher_eval]],
        )
        chrf = sacrebleu.corpus_chrf(
            val_teacher[:n_teacher_eval],
            [val_tgt_human[:n_teacher_eval]],
        )
        log(f"Teacher BLEU: {bleu.score:.2f} | chrF: {chrf.score:.2f}")

    # -----------------------------
    # SentencePiece
    # -----------------------------
    combined_hash = compute_hash(src_all + teacher_out_all)
    sp_src, sp_tgt = load_or_train_sp(src_all, teacher_out_all, combined_hash)

    Vsrc = sp_src.get_piece_size()
    Vtgt = sp_tgt.get_piece_size()
    log(f"Vocab sizes — src: {Vsrc}, tgt: {Vtgt}")

    # -----------------------------
    # Encode data (STUDENT TARGET = TEACHER)
    # -----------------------------
    log("Encoding with SentencePiece...")

    train_src_ids, train_tin_ids, train_tout_ids = [], [], []
    val_src_ids, val_tin_ids, val_tout_ids = [], [], []

    max_sl, max_tl = 0, 0

    for s, t in zip(train_src, train_teacher):
        s_i = sp_encode(sp_src, s, MAX_SRC_LEN)
        t_i = sp_encode(sp_tgt, t, MAX_TGT_LEN)
        train_src_ids.append(s_i)
        train_tin_ids.append(t_i[:-1])
        train_tout_ids.append(t_i[1:])
        max_sl = max(max_sl, len(s_i))
        max_tl = max(max_tl, len(t_i) - 1)

    for s, t in zip(val_src, val_teacher):
        s_i = sp_encode(sp_src, s, MAX_SRC_LEN)
        t_i = sp_encode(sp_tgt, t, MAX_TGT_LEN)
        val_src_ids.append(s_i)
        val_tin_ids.append(t_i[:-1])
        val_tout_ids.append(t_i[1:])
        max_sl = max(max_sl, len(s_i))
        max_tl = max(max_tl, len(t_i) - 1)

    train_src_arr = pad_sequences(train_src_ids, max_sl)
    train_tin_arr = pad_sequences(train_tin_ids, max_tl)
    train_tout_arr = pad_sequences(train_tout_ids, max_tl)

    val_src_arr = pad_sequences(val_src_ids, max_sl)
    val_tin_arr = pad_sequences(val_tin_ids, max_tl)
    val_tout_arr = pad_sequences(val_tout_ids, max_tl)


    train_human_tin, train_human_tout = [], []
    val_human_tin, val_human_tout = [], []

    for s, t in zip(train_src, train_tgt_human):
        t_i = sp_encode(sp_tgt, t, MAX_TGT_LEN)
        train_human_tin.append(t_i[:-1])
        train_human_tout.append(t_i[1:])

    for s, t in zip(val_src, val_tgt_human):
        t_i = sp_encode(sp_tgt, t, MAX_TGT_LEN)
        val_human_tin.append(t_i[:-1])
        val_human_tout.append(t_i[1:])

    train_human_ds = SeqDataset(
        train_src_arr,
        pad_sequences(train_human_tin, max_tl),
        pad_sequences(train_human_tout, max_tl),
    )

    val_human_ds = SeqDataset(
    val_src_arr,
    pad_sequences(val_human_tin, max_tl),
    pad_sequences(val_human_tout, max_tl),
    )

    train_human_loader = DataLoader(
    train_human_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=(DEVICE == "cuda"),
    )
    # -----------------------------
    # DataLoaders
    # -----------------------------
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

    # -----------------------------
    # DataLoader for DISTILLATION (small batch)
    distill_loader = DataLoader(
        train_ds,                 # SAME dataset as Phase 1
        batch_size=DISTILL_BATCH_SIZE,
        shuffle=True,
        pin_memory=(DEVICE == "cuda"),
    )

    val_human_loader = DataLoader(
    val_human_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=(DEVICE == "cuda"),
    )

    # -----------------------------
    # Build Transformer student
    # -----------------------------
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
    scaler = GradScaler(enabled=(DEVICE == "cuda"))



    # -----------------------------
    # Training loop (Phase 1)
    # -----------------------------
    best_val_loss = float("inf")
    epochs_no_improve = 0
    save_path = "student_transformer_best.pt"

    log("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        log(f"Epoch {epoch}/{EPOCHS}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, epoch
        )
        log(f"Train loss: {train_loss:.4f}")

        val_loss = eval_loss(model, val_loader, criterion)
        log(f"Val loss:   {val_loss:.4f}")
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        log(f"Current LR: {current_lr:.6f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            log(f"Saved new best model to {save_path}")
        else:
            epochs_no_improve += 1
            log(f"No improvement ({epochs_no_improve}/{PATIENCE})")

            if epochs_no_improve >= PATIENCE:
                log("Early stopping triggered.")
                break

        # -------------------------
        # BLEU / chrF (beam search vs HUMAN)
        # -------------------------
        if epoch < 5:
            continue
        n_eval = min(0, len(val_src_arr))

        if epoch % 5 == 0:
        n_eval = min(VAL_SAMPLES_FOR_BLEU, len(val_src_arr))
    
        preds = beam_search_decode_val(
            model,
            sp_tgt,
            val_src_arr,
            n_eval,
        )
    
        bleu = sacrebleu.corpus_bleu(
            preds,
            [val_tgt_human[:n_eval]],
        )
    
        chrf = sacrebleu.corpus_chrf(
            preds,
            [val_tgt_human[:n_eval]],
        )

        log(f"Val subset BLEU: {bleu.score:.2f} | chrF: {chrf.score:.2f}")
    
    # -----------------------------
    # Phase 2: KL Distillation
    # -----------------------------
    log("Starting Phase 2: KL distillation")

    teacher_tok, teacher_model = load_teacher_for_logits()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.98),
        weight_decay=0.01,
    )
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    for epoch in range(1, DISTILL_EPOCHS + 1):
        log(f"[Distill] Epoch {epoch}/{DISTILL_EPOCHS}")

        loss = train_one_epoch_distill(
            model,
            teacher_model,
            distill_loader,   # ✅ CORRECT
            optimizer,
            scaler,
        )

        log(f"Distill loss: {loss:.4f}")

        torch.save(model.state_dict(), "student_transformer_distilled.pt")


    # ===============================================================
    # PHASE 3 — HUMAN FINE-TUNING
    # ===============================================================
    log("Starting Phase 3: Fine-tuning on human Telugu")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    total_ft_epochs = 7
    
    for epoch in range(1, total_ft_epochs + 1):
        log(f"[Human FT] Epoch {epoch}/{total_ft_epochs}")
        
        loss = train_one_epoch(model, DataLoader(train_human_ds, batch_size=BATCH_SIZE, shuffle=True), optimizer, criterion, scaler, epoch)
        log(f"Human FT loss: {loss:.4f}")

    log("Running FULL validation BLEU (this may take time)...")

    n_eval = len(val_src_arr)

    preds = beam_search_decode_val(
        model,
        sp_tgt,
        val_src_arr,
        n_eval,
    )
    
    bleu = sacrebleu.corpus_bleu(
        preds,
        [val_tgt_human],
    )
    
    chrf = sacrebleu.corpus_chrf(
        preds,
        [val_tgt_human],
    )
    
    log(f"FULL VAL BLEU: {bleu.score:.2f} | chrF: {chrf.score:.2f}")
    
    log("Training complete.")
    log(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
