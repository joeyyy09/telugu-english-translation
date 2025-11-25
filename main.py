# ===============================================================
# English -> Telugu NMT (Teacher: NLLB, Student: BiGRU)
# FINAL VERSION WITH ALL IMPROVEMENTS:
#
#  - Data cleaning, filtering, deduplication
#  - SentencePiece tokenization
#  - Teacher beam-search translation (better quality)
#  - Temperature-scaled knowledge distillation
#  - Top-k soft target distillation (MUCH better KD)
#  - Label smoothing on hard CE loss
#  - Mixed precision (AMP) for faster training
#  - Increased student model capacity
#  - Stable training with grad-clipping
#  - Full logging to console + timestamped .log file
#
# DOES NOT change architecture type (still BiGRU encoder-decoder)
# ===============================================================

import os
import sys
import time
import math
import random
import logging
from typing import List, Tuple

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

DATA_PATH = "English Telugu Data.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Teacher config
TEACHER_MODEL = "facebook/nllb-200-distilled-600M"
TEACHER_BATCH = 32
TEACHER_MAX_LEN = 128
TEACHER_BEAMS = 4
KD_TEMPERATURE = 2.0
KD_TOPK = 8              # top-k soft logits from teacher

# Student config
EPOCHS = 5
BATCH_SIZE = 64
EMB = 512
HID = 512
LR = 0.0008
GRAD_CLIP = 1.0

# Tokenizer configs
SP_SRC_VOCAB = 32000
SP_TGT_VOCAB = 32000

SP_SRC_PREFIX = "sp_src"
SP_TGT_PREFIX = "sp_tgt"

MAX_SRC_LEN = 120
MAX_TGT_LEN = 120

PAD_ID = 0

# Label smoothing
LABEL_SMOOTH = 0.1


# ===============================================================
# LOGGING SETUP
# ===============================================================

TS = time.strftime("%Y%m%d-%H%M%S")
LOG_FILE = f"run_{TS}.log"

logger = logging.getLogger("nmt")
logger.setLevel(logging.DEBUG)

fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

fh = logging.FileHandler(LOG_FILE, "w", encoding="utf-8")
fh.setFormatter(fmt)
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
sh.setLevel(logging.INFO)
logger.addHandler(sh)

logger.info("=== FINAL NMT RUN STARTED ===")


# ===============================================================
# SEED FIX
# ===============================================================
def fix_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

fix_seed()


# ===============================================================
# DATA LOADING + CLEANING
# ===============================================================
def read_pairs(path):
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

def clean_text(s):
    s = "".join(ch for ch in s if ch.isprintable())
    s = s.replace("\u200b", "")
    s = " ".join(s.split())
    return s

def is_telugu(s):
    total = len(s)
    if total == 0: return False
    tel = sum(1 for ch in s if 0x0C00 <= ord(ch) <= 0x0C7F)
    return tel / total >= 0.25

def clean_pairs(pairs):
    out = []
    seen = set()
    for s, t in pairs:
        s = clean_text(s)
        t = clean_text(t)
        if not s or not t:
            continue
        if not is_telugu(t):
            continue
        if len(s.split()) > MAX_SRC_LEN: continue
        if len(t.split()) > MAX_TGT_LEN: continue
        key = (s.lower(), t)
        if key in seen: continue
        seen.add(key)
        out.append((s, t))
    logger.info("After cleaning: %d pairs", len(out))
    return out


# ===============================================================
# SENTENCEPIECE
# ===============================================================
def write_list(path, lst):
    with open(path, "w", encoding="utf-8") as f:
        for x in lst:
            f.write(x + "\n")

def train_sp(input_file, prefix, vocab):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=prefix,
        vocab_size=vocab,
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


# ===============================================================
# STUDENT MODEL
# ===============================================================
class Encoder(nn.Module):
    def __init__(self, vocab, emb, hid):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hid, batch_first=True, bidirectional=True)

    def forward(self, x):
        e = self.emb(x)
        out, h = self.gru(e)
        h = torch.cat([h[0], h[1]], dim=1)  # [B, 2H]
        return out, h


class Decoder(nn.Module):
    def __init__(self, vocab, emb, hid):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hid, batch_first=True)
        self.lin = nn.Linear(hid, vocab)

    def forward(self, tgt, h0):
        e = self.emb(tgt)
        out, h1 = self.gru(e, h0.unsqueeze(0))
        logits = self.lin(out)
        return logits, h1.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, src, tgt, teacher_forcing=True):
        _, h = self.enc(src)
        logits, _ = self.dec(tgt, h)
        return logits


# ===============================================================
# DISTILLATION + LABEL SMOOTHING
# ===============================================================
def label_smooth_loss(logits, targets, eps=0.1, ignore_idx=0):
    """
    Standard label smoothing CE.
    """
    B, T, V = logits.size()
    logits = logits.view(B*T, V)
    targets = targets.view(B*T)

    prob = torch.softmax(logits, dim=1)
    log_prob = torch.log(prob + 1e-9)

    n_classes = V

    true_dist = torch.zeros_like(prob)
    true_dist.fill_(eps / (n_classes - 1))
    mask = targets != ignore_idx
    true_dist[mask, targets[mask]] = 1 - eps

    loss = -(true_dist * log_prob).sum(dim=1)
    loss = loss[mask].mean()
    return loss


def kd_soft_loss(student_logits, teacher_logits, temp):
    """
    teacher_logits and student_logits are [B,T,V]
    """
    p = torch.log_softmax(student_logits / temp, dim=-1)
    q = torch.softmax(teacher_logits / temp, dim=-1)
    loss = torch.nn.functional.kl_div(p, q, reduction="batchmean") * (temp*temp)
    return loss


# ===============================================================
# TEACHER TOP-K LOGITS EXTRACTION
# ===============================================================
def build_teacher_soft_targets(teacher_model, teacher_tok, sentences, max_len, k=8):
    """
    Returns a list of T x V tensors with top-k logits for each token.
    """
    teacher_model.eval()
    soft_targets = []

    with torch.no_grad():
        enc = teacher_tok(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len
        ).to(DEVICE)

        out = teacher_model(
            **enc,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True
        )

        logits = out.logits  # [B, L, V_teacher]

        # convert to list-of-tensors with only top-k kept
        for i in range(logits.size(0)):
            row = logits[i]  # [L, V]
            L, V = row.size()
            topk_vals, topk_idx = torch.topk(row, k=k, dim=1)
            # create dense logits in student vocab shape later (we convert)
            soft_targets.append((topk_idx, topk_vals))

    return soft_targets


# ===============================================================
# MAIN PIPELINE
# ===============================================================
def main():

    # ===============================================================
    # LOAD + CLEAN DATA
    # ===============================================================
    logger.info("Loading data...")
    raw_pairs = read_pairs(DATA_PATH)
    pairs = clean_pairs(raw_pairs)

    src_texts = [s for s, _ in pairs]
    tgt_texts = [t for _, t in pairs]

    # ===============================================================
    # TEACHER MODEL
    # ===============================================================
    logger.info("Loading teacher NLLB...")
    teacher_tok = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    teacher_model = AutoModelForSeq2SeqLM.from_pretrained(TEACHER_MODEL).to(DEVICE)
    teacher_model.eval()

    # ===============================================================
    # TEACHER TRANSLATION
    # ===============================================================
    logger.info("Teacher translating with beam search...")
    teacher_tgt_texts = []

    for i in tqdm(range(0, len(src_texts), TEACHER_BATCH)):
        batch = src_texts[i:i+TEACHER_BATCH]
        enc = teacher_tok(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        gen = teacher_model.generate(
            **enc,
            max_length=TEACHER_MAX_LEN,
            num_beams=TEACHER_BEAMS,
            do_sample=False
        )
        dec = teacher_tok.batch_decode(gen, skip_special_tokens=True)
        teacher_tgt_texts.extend(dec)

    assert len(teacher_tgt_texts) == len(src_texts)

    # ===============================================================
    # TRAIN SENTENCEPIECE MODELS
    # ===============================================================
    tmp_src = "tmp_src.txt"
    tmp_tgt = "tmp_tgt.txt"
    write_list(tmp_src, src_texts)
    write_list(tmp_tgt, teacher_tgt_texts)

    logger.info("Training SentencePiece...")
    sp_src = train_sp(tmp_src, SP_SRC_PREFIX, SP_SRC_VOCAB)
    sp_tgt = train_sp(tmp_tgt, SP_TGT_PREFIX, SP_TGT_VOCAB)

    os.remove(tmp_src)
    os.remove(tmp_tgt)

    # ===============================================================
    # ENCODE DATA WITH SP
    # ===============================================================
    def sp_encode(sp, text, max_len):
        ids = sp.encode(text, out_type=int)
        ids = [sp.bos_id()] + ids + [sp.eos_id()]
        if len(ids) > max_len:
            ids = ids[:max_len]
            ids[-1] = sp.eos_id()
        return ids

    src_ids, tgt_in_ids, tgt_out_ids = [], [], []
    max_sl, max_tl = 0, 0

    for s, t in zip(src_texts, teacher_tgt_texts):
        s_i = sp_encode(sp_src, s, MAX_SRC_LEN)
        t_i = sp_encode(sp_tgt, t, MAX_TGT_LEN)
        dec_in = t_i[:-1]
        dec_out = t_i[1:]
        src_ids.append(s_i)
        tgt_in_ids.append(dec_in)
        tgt_out_ids.append(dec_out)
        max_sl = max(max_sl, len(s_i))
        max_tl = max(max_tl, len(dec_in))

    def pad_batch(lst, mx):
        arr = np.full((len(lst), mx), PAD_ID, dtype=np.int64)
        for i, seq in enumerate(lst):
            arr[i, :len(seq)] = seq
        return arr

    src_arr = pad_batch(src_ids, max_sl)
    tgt_in_arr = pad_batch(tgt_in_ids, max_tl)
    tgt_out_arr = pad_batch(tgt_out_ids, max_tl)

    # ===============================================================
    # DATASET
    # ===============================================================
    class SeqDataset(Dataset):
        def __init__(self, s, ti, to):
            self.s, self.ti, self.to = s, ti, to
        def __len__(self):
            return len(self.s)
        def __getitem__(self, i):
            return self.s[i], self.ti[i], self.to[i]

    ds = SeqDataset(src_arr, tgt_in_arr, tgt_out_arr)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    # ===============================================================
    # STUDENT MODEL
    # ===============================================================
    Vsrc = sp_src.get_piece_size()
    Vtgt = sp_tgt.get_piece_size()
    enc = Encoder(Vsrc, EMB, HID).to(DEVICE)
    dec = Decoder(Vtgt, EMB, HID*2).to(DEVICE)
    model = Seq2Seq(enc, dec).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    # ===============================================================
    # TRAINING LOOP
    # ===============================================================
    logger.info("Training student...")

    best_loss = float("inf")

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        total_kd = 0
        total_smooth = 0
        batches = 0

        for src_b, tin_b, tout_b in tqdm(dl, desc=f"Epoch {epoch}"):

            src_b = src_b.to(DEVICE)
            tin_b = tin_b.to(DEVICE)
            tout_b = tout_b.to(DEVICE)

            opt.zero_grad()

            with autocast():

                # 1) Student forward
                student_logits = model(src_b, tin_b, teacher_forcing=True)

                # 2) Teacher soft targets (top-k)
                # decode text again (needed for teacher tokenizer)
                batch_txt = [sp_src.decode([x for x in row if x != 0]) for row in src_b.cpu().numpy()]
                enc = teacher_tok(batch_txt, return_tensors="pt", padding=True).to(DEVICE)
                with torch.no_grad():
                    output = teacher_model(**enc)
                    teach_logits = output.logits  # [B,L,V_teacher]

                # convert teacher top-k to full student vocab distribution
                teach_logits_student = torch.full(
                    (teach_logits.size(0), student_logits.size(1), Vtgt),
                    -10.0, device=DEVICE
                )

                for bi in range(teach_logits.size(0)):
                    row = teach_logits[bi]  # [L,V_teacher]
                    L = min(student_logits.size(1), row.size(0))
                    tk_vals, tk_idx = torch.topk(row[:L], k=min(KD_TOPK, row.size(1)), dim=1)

                    for t in range(L):
                        for j in range(tk_idx.size(1)):
                            teacher_token_id = tk_idx[t,j].item()
                            teacher_token_piece = teacher_tok.convert_ids_to_tokens(teacher_token_id)
                            try:
                                st_id = sp_tgt.piece_to_id(teacher_token_piece)
                            except:
                                continue
                            if st_id < Vtgt:
                                teach_logits_student[bi, t, st_id] = tk_vals[t,j]

                # 3) Combined loss: KD + label smoothing CE
                kd_loss = kd_soft_loss(student_logits, teach_logits_student, KD_TEMPERATURE)
                ls_loss = label_smooth_loss(student_logits, tout_b, LABEL_SMOOTH, ignore_idx=PAD_ID)

                loss = 0.5 * kd_loss + 0.5 * ls_loss

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            total_kd += kd_loss.item()
            total_smooth += ls_loss.item()
            batches += 1

        avg = total_loss / batches
        logger.info(f"Epoch {epoch} | Loss={avg:.4f} | KD={total_kd/batches:.4f} | SM={total_smooth/batches:.4f}")

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), "best_student.pt")
            logger.info("Saved new BEST model.")

    logger.info("Training complete.")
    logger.info(f"Logs written to {LOG_FILE}")


if __name__ == "__main__":
    main()
