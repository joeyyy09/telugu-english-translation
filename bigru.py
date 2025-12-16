# main.py
# ===============================================================
# English -> Telugu NMT (Teacher: NLLB, Student: BiGRU + Attention)
#
# IMPROVED BASELINE VERSION:
#  - Stronger student:
#       * EMB_DIM      : 768
#       * ENC_HID_DIM  : 384 per direction (BiGRU → 768)
#       * DEC_HID_DIM  : 768
#       * Dropout in encoder/decoder
#  - Better tokenization:
#       * SentencePiece vocab: 16k / 16k (src / tgt)
#       * SP cache hash now includes vocab sizes (prevents stale models)
#  - Training:
#       * Scheduled teacher forcing (0.8 → 0.4 with decay)
#       * Label smoothing (0.1)
#       * AMP + grad clipping
#       * Early stopping
#  - Inference:
#       * Beam search decoding (beam=5) with length normalization
#       * BLEU/chrF vs HUMAN Telugu on validation subset
#  - Caching:
#       * Teacher translations (NLLB) with hash
#       * SentencePiece models with hash
#
#  This is designed to beat the ~25 BLEU / 56 chrF baseline.
# ===============================================================

import os
import sys
import time
import random
import hashlib
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
print("Using device:", DEVICE)

SEED = 42

# Teacher config
TEACHER_MODEL = "facebook/nllb-200-distilled-600M"
TEACHER_BATCH = 32
TEACHER_MAX_LEN = 100
TEACHER_BEAMS = 4

# Student config
EPOCHS = 35
BATCH_SIZE = 64

EMB_DIM = 768
ENC_HID_DIM = 384            # per direction → 768 biGRU output
DEC_HID_DIM = 768

ENC_DROPOUT = 0.3
DEC_DROPOUT = 0.3

LR = 8e-4
GRAD_CLIP = 1.0

# Scheduled teacher forcing
BASE_TEACHER_FORCING = 0.8
MIN_TEACHER_FORCING = 0.4
TF_DECAY = 0.97

# Tokenizer configs
SP_SRC_VOCAB = 12000
SP_TGT_VOCAB = 16000

SP_SRC_PREFIX = "sp_src"
SP_TGT_PREFIX = "sp_tgt"

MAX_SRC_LEN = 100
MAX_TGT_LEN = 100

PAD_ID = 0

# Evaluation subset
VAL_SAMPLES_FOR_BLEU = 500

# Early stopping
PATIENCE = 5          # epochs without improvement before stopping

# Cache files
TEACHER_OUT_PATH = "teacher_outputs.txt"
TEACHER_HASH_PATH = "teacher_hash.txt"
SP_HASH_PATH = "sp_hash.txt"

# Beam search
BEAM_SIZE = 5
LENGTH_NORM_ALPHA = 0.7   # length-normalization exponent

# ===============================================================
# SEED / CUDNN
# ===============================================================

def fix_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

fix_seed()

# ===============================================================
# LOGGING
# ===============================================================

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()

# ===============================================================
# HASH UTILS
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
# TEACHER TRANSLATION CACHING
# ===============================================================

def load_or_run_teacher(src_texts: List[str]) -> Tuple[List[str], str]:
    """
    Returns teacher translations for all src_texts and the hash.
    Uses cache if possible.
    """
    current_hash = compute_hash(src_texts)

    if os.path.exists(TEACHER_OUT_PATH) and os.path.exists(TEACHER_HASH_PATH):
        with open(TEACHER_HASH_PATH, "r", encoding="utf-8") as f:
            old_hash = f.read().strip()
        if old_hash == current_hash:
            with open(TEACHER_OUT_PATH, "r", encoding="utf-8") as f:
                lines = [x.rstrip("\n") for x in f]
            if len(lines) == len(src_texts):
                log("Reusing cached teacher translations (hash + length match).")
                return lines, current_hash
            else:
                log("Cached teacher outputs length mismatch. Will recompute.")

    # Need to run teacher
    log("Running teacher NLLB for translations (no valid cache).")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(TEACHER_MODEL).to(DEVICE)
    model.eval()

    translations = []
    tokenizer.src_lang = "eng_Latn"
    try:
        forced_bos_token_id = tokenizer.convert_tokens_to_ids("tel_Telu")
    except Exception:
        forced_bos_token_id = None

    for i in tqdm(range(0, len(src_texts), TEACHER_BATCH), desc="Teacher beam search"):
        batch = src_texts[i:i + TEACHER_BATCH]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=TEACHER_MAX_LEN,
        ).to(DEVICE)
        gen_kwargs = dict(
            max_length=TEACHER_MAX_LEN,
            num_beams=TEACHER_BEAMS,
            do_sample=False,
        )
        if forced_bos_token_id is not None:
            gen_kwargs["forced_bos_token_id"] = forced_bos_token_id
        with torch.no_grad():
            gen = model.generate(**enc, **gen_kwargs)
        dec = tokenizer.batch_decode(gen, skip_special_tokens=True)
        translations.extend(dec)

    assert len(translations) == len(src_texts)

    # Save cache
    with open(TEACHER_OUT_PATH, "w", encoding="utf-8") as f:
        for line in translations:
            f.write(line.replace("\n", " ") + "\n")
    with open(TEACHER_HASH_PATH, "w", encoding="utf-8") as f:
        f.write(current_hash)

    log("Saved teacher outputs + hash for future runs.")
    return translations, current_hash

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

def load_or_train_sp(src_texts: List[str],
                     teacher_tgt_texts: List[str],
                     data_hash: str):
    """
    Cache SentencePiece models based on data_hash + vocab sizes.
    If sp_src.model / sp_tgt.model + sp_hash.txt exist and hash matches, reuse.
    Otherwise retrain.
    """
    src_model_path = SP_SRC_PREFIX + ".model"
    tgt_model_path = SP_TGT_PREFIX + ".model"

    # Include vocab sizes in hash so changing vocab retrains SP
    combined = data_hash + f"|src_vocab={SP_SRC_VOCAB}|tgt_vocab={SP_TGT_VOCAB}"

    if (
        os.path.exists(src_model_path)
        and os.path.exists(tgt_model_path)
        and os.path.exists(SP_HASH_PATH)
    ):
        with open(SP_HASH_PATH, "r", encoding="utf-8") as f:
            old_hash = f.read().strip()
        if old_hash == combined:
            sp_src = spm.SentencePieceProcessor()
            sp_tgt = spm.SentencePieceProcessor()
            sp_src.load(src_model_path)
            sp_tgt.load(tgt_model_path)
            log("Reusing cached SentencePiece models (hash + vocab match).")
            return sp_src, sp_tgt

    # Retrain SP
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
        f.write(combined)

    log("Saved new SentencePiece models and updated sp_hash.txt")
    return sp_src, sp_tgt

def sp_encode(sp, text, max_len):
    ids = sp.encode(text, out_type=int)
    ids = [sp.bos_id()] + ids + [sp.eos_id()]
    if len(ids) > max_len:
        ids = ids[:max_len]
        ids[-1] = sp.eos_id()
    return ids

def pad_sequences(list_of_ids: List[List[int]], max_len: int, pad_id: int = 0):
    arr = np.full((len(list_of_ids), max_len), pad_id, dtype=np.int64)
    for i, seq in enumerate(list_of_ids):
        arr[i, :len(seq)] = seq
    return arr

# ===============================================================
# DATASET
# ===============================================================

class SeqDataset(Dataset):
    def __init__(self, src_arr, tgt_in_arr, tgt_out_arr):
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
# MODEL: BiGRU + Bahdanau Attention
# ===============================================================

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hid_dim, pad_idx=0, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            emb_dim,
            enc_hid_dim,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, src):  # src: [B, S]
        embedded = self.embedding(src)       # [B, S, E]
        embedded = self.dropout(embedded)
        outputs, hidden = self.gru(embedded) # outputs: [B, S, 2*H], hidden: [2, B, H]
        return outputs, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim * 2 + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, src_mask):
        """
        hidden: [B, dec_hid_dim]
        encoder_outputs: [B, S, 2*enc_hid_dim]
        src_mask: [B, S] (bool)
        """
        B, S, _ = encoder_outputs.shape
        hidden_rep = hidden.unsqueeze(1).repeat(1, S, 1)  # [B, S, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden_rep, encoder_outputs), dim=2)))  # [B, S, dec_hid_dim]
        scores = self.v(energy).squeeze(2)  # [B, S]
        mask_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~src_mask, mask_value)
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [B, 2*enc_hid_dim]
        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, pad_idx=0, dropout=0.0):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(enc_hid_dim, dec_hid_dim)
        self.gru = nn.GRU(
            emb_dim + enc_hid_dim * 2,
            dec_hid_dim,
            batch_first=True
        )
        self.fc_out = nn.Linear(dec_hid_dim + enc_hid_dim * 2, output_dim)

    def forward_step(self, input_tokens, hidden, encoder_outputs, src_mask):
        """
        input_tokens: [B] (token ids)
        hidden: [B, dec_hid_dim]
        encoder_outputs: [B, S, 2*enc_hid_dim]
        src_mask: [B, S]
        """
        embedded = self.embedding(input_tokens)  # [B, E]
        embedded = self.dropout(embedded)
        context, attn_weights = self.attention(hidden, encoder_outputs, src_mask)  # context: [B, 2*enc_hid_dim]

        rnn_input = torch.cat((embedded, context), dim=1).unsqueeze(1)  # [B, 1, E+2H]
        hidden_in = hidden.unsqueeze(0)  # [1, B, dec_hid_dim]

        output, hidden_out = self.gru(rnn_input, hidden_in)  # output: [B,1,dec_hid_dim]
        output = output.squeeze(1)       # [B, dec_hid_dim]
        hidden_new = hidden_out.squeeze(0)  # [B, dec_hid_dim]

        pred_logits = self.fc_out(torch.cat((output, context), dim=1))  # [B, output_dim]
        return pred_logits, hidden_new, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bridge = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self, src, trg_in, src_mask, teacher_forcing_ratio=1.0):
        """
        src: [B, S]
        trg_in: [B, T] (teacher input sequence, starting with BOS)
        src_mask: [B, S]
        Returns logits: [B, T, V]
        """
        B, T = trg_in.shape
        V = self.decoder.output_dim
        outputs = torch.zeros(B, T, V, device=src.device)

        encoder_outputs, hidden_enc = self.encoder(src)  # hidden_enc: [2, B, enc_hid_dim]
        # concat forward + backward
        hidden_cat = torch.cat((hidden_enc[-2], hidden_enc[-1]), dim=1)  # [B, 2*enc_hid_dim]
        hidden_dec = torch.tanh(self.bridge(hidden_cat))                 # [B, dec_hid_dim]

        input_tokens = trg_in[:, 0]  # BOS

        for t in range(T):
            logits_step, hidden_dec, _ = self.decoder.forward_step(
                input_tokens, hidden_dec, encoder_outputs, src_mask
            )
            outputs[:, t, :] = logits_step

            if t + 1 < T:
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = logits_step.argmax(dim=1)
                next_in = trg_in[:, t + 1] if teacher_force else top1
                input_tokens = next_in

        return outputs

# ===============================================================
# TRAINING / EVAL
# ===============================================================

def get_teacher_forcing_ratio(epoch: int) -> float:
    ratio = BASE_TEACHER_FORCING * (TF_DECAY ** (epoch - 1))
    return max(ratio, MIN_TEACHER_FORCING)

def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch):
    model.train()
    total_loss = 0.0
    tf_ratio = get_teacher_forcing_ratio(epoch)
    log(f"Teacher forcing ratio this epoch: {tf_ratio:.3f}")

    use_amp = (DEVICE == "cuda")

    for src_b, tin_b, tout_b in tqdm(loader, desc="Train", leave=False):
        src_b = src_b.to(DEVICE)
        tin_b = tin_b.to(DEVICE)
        tout_b = tout_b.to(DEVICE)
        src_mask = (src_b != PAD_ID).bool().to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(src_b, tin_b, src_mask, teacher_forcing_ratio=tf_ratio)
            B, T, V = logits.shape
            loss = criterion(logits.view(B * T, V), tout_b.view(B * T))

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def eval_loss(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    for src_b, tin_b, tout_b in tqdm(loader, desc="Val Loss", leave=False):
        src_b = src_b.to(DEVICE)
        tin_b = tin_b.to(DEVICE)
        tout_b = tout_b.to(DEVICE)
        src_mask = (src_b != PAD_ID).bool().to(DEVICE)

        logits = model(src_b, tin_b, src_mask, teacher_forcing_ratio=1.0)
        B, T, V = logits.shape
        loss = criterion(logits.view(B * T, V), tout_b.view(B * T))
        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def greedy_decode_batch(model, sp_tgt, src_batch_np):
    """
    src_batch_np: [B, S] numpy
    Return: list of decoded strings (student predictions).
    """
    model.eval()
    src = torch.tensor(src_batch_np, dtype=torch.long, device=DEVICE)
    src_mask = (src != PAD_ID).bool().to(DEVICE)

    encoder_outputs, hidden_enc = model.encoder(src)
    hidden_cat = torch.cat((hidden_enc[-2], hidden_enc[-1]), dim=1)
    hidden_dec = torch.tanh(model.bridge(hidden_cat))

    B = src.shape[0]
    bos_id = sp_tgt.bos_id()
    eos_id = sp_tgt.eos_id()

    input_tokens = torch.full((B,), bos_id, dtype=torch.long, device=DEVICE)
    finished = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    sequences = [[] for _ in range(B)]

    for _ in range(MAX_TGT_LEN):
        logits_step, hidden_dec, _ = model.decoder.forward_step(
            input_tokens, hidden_dec, encoder_outputs, src_mask
        )
        next_tokens = logits_step.argmax(dim=1)  # [B]

        for i in range(B):
            if finished[i]:
                continue
            tid = next_tokens[i].item()
            if tid == eos_id:
                finished[i] = True
            elif tid != PAD_ID:
                sequences[i].append(tid)

        input_tokens = next_tokens
        if finished.all():
            break

    # Decode sequences
    texts = []
    for seq in sequences:
        if len(seq) == 0:
            texts.append("")
        else:
            texts.append(sp_tgt.decode(seq))
    return texts

@torch.no_grad()
def beam_search_decode_one(model, sp_tgt, src_seq_np, beam_size=BEAM_SIZE, alpha=LENGTH_NORM_ALPHA):
    """
    Beam search decode a single source sequence with length normalization.
    src_seq_np: [S] numpy array
    """
    model.eval()
    src = torch.tensor(src_seq_np, dtype=torch.long, device=DEVICE).unsqueeze(0)  # [1,S]
    src_mask = (src != PAD_ID).bool().to(DEVICE)

    encoder_outputs, hidden_enc = model.encoder(src)
    hidden_cat = torch.cat((hidden_enc[-2], hidden_enc[-1]), dim=1)  # [1,2H]
    hidden_dec = torch.tanh(model.bridge(hidden_cat))[0]             # [dec_hid_dim]

    bos_id = sp_tgt.bos_id()
    eos_id = sp_tgt.eos_id()

    # Beam elements: (log_prob, length, seq, hidden)
    beams = [(0.0, 1, [bos_id], hidden_dec)]
    max_steps = MAX_TGT_LEN

    for _ in range(max_steps):
        new_beams = []
        all_finished = True

        for logp, length, seq, h in beams:
            if seq[-1] == eos_id:
                new_beams.append((logp, length, seq, h))
                continue

            all_finished = False
            input_token = torch.tensor([seq[-1]], dtype=torch.long, device=DEVICE)  # [1]
            h_batch = h.unsqueeze(0)  # [1,H]
            logits_step, h_new_batch, _ = model.decoder.forward_step(
                input_token,
                h_batch,
                encoder_outputs,
                src_mask,
            )
            h_new = h_new_batch[0]  # [H]
            log_probs = torch.log_softmax(logits_step[0], dim=-1)  # [V]
            topk_vals, topk_idx = torch.topk(log_probs, beam_size)

            for k in range(beam_size):
                nid = topk_idx[k].item()
                new_logp = logp + topk_vals[k].item()
                new_seq = seq + [nid]
                new_len = length + 1
                new_beams.append((new_logp, new_len, new_seq, h_new))

        if all_finished:
            break

        # sort beams by length-normalized score
        def norm_score(b):
            lp, ln, _, _ = b
            return lp / (ln ** alpha)
        new_beams.sort(key=norm_score, reverse=True)
        beams = new_beams[:beam_size]

    # choose best beam by normalized score
    def norm_score(b):
        lp, ln, _, _ = b
        return lp / (ln ** alpha)
    beams.sort(key=norm_score, reverse=True)
    best_seq = beams[0][2]

    # remove BOS + cut at EOS
    toks = best_seq[1:]
    if eos_id in toks:
        toks = toks[:toks.index(eos_id)]
    if len(toks) == 0:
        return ""
    return sp_tgt.decode(toks)

@torch.no_grad()
def beam_search_decode_val(model, sp_tgt, src_arr, n_eval, beam_size=BEAM_SIZE):
    preds = []
    for i in tqdm(range(n_eval), desc="Beam decode val", leave=False):
        pred = beam_search_decode_one(model, sp_tgt, src_arr[i], beam_size=beam_size)
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

    # Shuffle and split train/val
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
    # Teacher translations (all)
    # -----------------------------
    teacher_out_all, teacher_hash = load_or_run_teacher(src_all)
    train_teacher = teacher_out_all[:train_size]
    val_teacher = teacher_out_all[train_size:]

        # -----------------------------
    # (NEW) Evaluate teacher translations vs human tgt on validation subset
    # -----------------------------
    # compute teacher scores on same subset size we use for the student evaluation
    n_teacher_eval = min(VAL_SAMPLES_FOR_BLEU, len(val_teacher), len(val_tgt_human))
    if n_teacher_eval > 0:
        log(f"Computing teacher BLEU/chrF on {n_teacher_eval} val samples...")
        teacher_preds_subset = val_teacher[:n_teacher_eval]
        teacher_refs_subset = val_tgt_human[:n_teacher_eval]

        # sacrebleu expects list of references (each ref list), so pass [refs]
        teacher_bleu = sacrebleu.corpus_bleu(teacher_preds_subset, [teacher_refs_subset])
        teacher_chrf = sacrebleu.corpus_chrf(teacher_preds_subset, [teacher_refs_subset])

        log(f"Teacher (NLLB) val subset ({n_teacher_eval}) BLEU: {teacher_bleu.score:.2f} | chrF: {teacher_chrf.score:.2f}")

        # optional: save to file for later analysis / reproducibility
        try:
            with open("teacher_scores.txt", "w", encoding="utf-8") as f:
                f.write(f"n={n_teacher_eval}\n")
                f.write(f"BLEU: {teacher_bleu.score:.6f}\n")
                f.write(f"chrF: {teacher_chrf.score:.6f}\n")
            log("Saved teacher scores to teacher_scores.txt")
        except Exception as e:
            log(f"Warning: could not write teacher_scores.txt: {e}")
    else:
        log("No validation samples available for teacher scoring.")

    # Hash for SP models: include both src and teacher outputs
    combined_hash = compute_hash(src_all + teacher_out_all)

    # -----------------------------
    # SentencePiece (cached)
    # -----------------------------
    sp_src, sp_tgt = load_or_train_sp(src_all, teacher_out_all, combined_hash)

    Vsrc = sp_src.get_piece_size()
    Vtgt = sp_tgt.get_piece_size()
    log(f"Vocab sizes - src: {Vsrc}, tgt: {Vtgt}")
    assert sp_src.pad_id() == PAD_ID and sp_tgt.pad_id() == PAD_ID, "SP pad_id must be 0."

    # -----------------------------
    # Encode train / val (STUDENT TARGET = TEACHER TRANSLATION)
    # -----------------------------
    log("Encoding with SentencePiece...")

    train_src_ids, train_tin_ids, train_tout_ids = [], [], []
    val_src_ids, val_tin_ids, val_tout_ids = [], [], []

    max_sl, max_tl = 0, 0

    # Train
    for s, t in zip(train_src, train_teacher):
        s_i = sp_encode(sp_src, s, MAX_SRC_LEN)
        t_i = sp_encode(sp_tgt, t, MAX_TGT_LEN)
        dec_in = t_i[:-1]
        dec_out = t_i[1:]
        train_src_ids.append(s_i)
        train_tin_ids.append(dec_in)
        train_tout_ids.append(dec_out)
        max_sl = max(max_sl, len(s_i))
        max_tl = max(max_tl, len(dec_in))

    # Val
    for s, t in zip(val_src, val_teacher):
        s_i = sp_encode(sp_src, s, MAX_SRC_LEN)
        t_i = sp_encode(sp_tgt, t, MAX_TGT_LEN)
        dec_in = t_i[:-1]
        dec_out = t_i[1:]
        val_src_ids.append(s_i)
        val_tin_ids.append(dec_in)
        val_tout_ids.append(dec_out)
        max_sl = max(max_sl, len(s_i))
        max_tl = max(max_tl, len(dec_in))

    log(f"Max src len: {max_sl} | Max tgt len: {max_tl}")

    train_src_arr = pad_sequences(train_src_ids, max_sl, PAD_ID)
    train_tin_arr = pad_sequences(train_tin_ids, max_tl, PAD_ID)
    train_tout_arr = pad_sequences(train_tout_ids, max_tl, PAD_ID)

    val_src_arr = pad_sequences(val_src_ids, max_sl, PAD_ID)
    val_tin_arr = pad_sequences(val_tin_ids, max_tl, PAD_ID)
    val_tout_arr = pad_sequences(val_tout_ids, max_tl, PAD_ID)

    # -----------------------------
    # Dataloaders
    # -----------------------------
    train_ds = SeqDataset(train_src_arr, train_tin_arr, train_tout_arr)
    val_ds = SeqDataset(val_src_arr, val_tin_arr, val_tout_arr)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,  
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )

    # -----------------------------
    # Build model
    # -----------------------------
    encoder = Encoder(Vsrc, EMB_DIM, ENC_HID_DIM, pad_idx=PAD_ID, dropout=ENC_DROPOUT)
    decoder = Decoder(Vtgt, EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, pad_idx=PAD_ID, dropout=DEC_DROPOUT)
    model = Seq2Seq(encoder, decoder, ENC_HID_DIM, DEC_HID_DIM).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    # -----------------------------
    # Training loop with early stopping
    # -----------------------------
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    save_path = "student_biGRU_attn_improved.pt"

    log("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        log(f"Epoch {epoch}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, epoch)
        log(f"Train loss: {train_loss:.4f}")

        val_loss = eval_loss(model, val_loader, criterion)
        log(f"Val loss:   {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss - 1e-4:  # small delta to avoid noise
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            log(f"New best model saved to {save_path}")
        else:
            epochs_no_improve += 1
            log(f"No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= PATIENCE:
                log(f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}")
                break

        # -------------------------
        # BLEU + chrF on subset (beam=5 decoding vs HUMAN Telugu)
        # -------------------------
        n_eval = min(VAL_SAMPLES_FOR_BLEU, len(val_src_arr))
        if n_eval > 0:
            refs = val_tgt_human[:n_eval]
            sys_preds = beam_search_decode_val(model, sp_tgt, val_src_arr, n_eval, beam_size=BEAM_SIZE)

            sys_preds = sys_preds[:n_eval]
            refs = refs[:n_eval]

            bleu = sacrebleu.corpus_bleu(sys_preds, [refs])
            chrf = sacrebleu.corpus_chrf(sys_preds, [refs])

            log(f"Val subset ({n_eval}) [beam={BEAM_SIZE}] BLEU: {bleu.score:.2f} | chrF: {chrf.score:.2f}")

    log("Training complete.")
    log(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")

if __name__ == "__main__":
    main()
