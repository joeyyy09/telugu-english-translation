# FILE: experiment_entropy.py
import os
import sys
import time
import math
import random
import hashlib
import types
from typing import List, Tuple

# --- THE ONNX BUG FIX ---
# This tricks IndicTrans2 into loading smoothly on newer transformers versions
if 'transformers.onnx' not in sys.modules:
    dummy_onnx = types.ModuleType('transformers.onnx')
    dummy_onnx.OnnxConfig = object
    dummy_onnx.OnnxSeq2SeqConfigWithPast = object
    sys.modules['transformers.onnx'] = dummy_onnx
# ------------------------

import numpy as np
import sacrebleu
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch.amp

# ===============================================================
# CONFIGURATION
# ===============================================================
DATA_PATH = "english-telugu.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
PAD_ID = 0
MAX_SRC_LEN, MAX_TGT_LEN = 64, 100
BATCH_SIZE, DISTILL_BATCH_SIZE = 128, 8
EPOCHS_PHASE1, EPOCHS_PHASE2, EPOCHS_PHASE3 = 40, 30, 8
TEACHER_1 = "facebook/nllb-200-distilled-600M"
TEACHER_2 = "ai4bharat/indictrans2-en-indic-1B"
BEAM_SIZE = 5

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
    sys.stdout.flush()

# ===============================================================
# DATA & TOKENIZATION (Simplified for space, exact same logic)
# ===============================================================
def read_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "++++$++++" in line:
                s, t = line.strip().split("++++$++++")
                pairs.append((s.strip(), t.strip()))
    return pairs

def clean_pairs(pairs):
    out = []
    for s, t in pairs:
        if s and t and len(s.split()) <= MAX_SRC_LEN and len(t.split()) <= MAX_TGT_LEN:
            out.append((s, t))
    return out

def train_sp(input_file, prefix, vocab_size):
    spm.SentencePieceTrainer.train(
        input=input_file, model_prefix=prefix, vocab_size=vocab_size, model_type="unigram",
        pad_id=0, unk_id=1, bos_id=2, eos_id=3)
    sp = spm.SentencePieceProcessor()
    sp.load(prefix + ".model")
    return sp

# ===============================================================
# MODEL ARCHITECTURE (Standard Transformer)
# ===============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2048):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=768, n_heads=8, n_layers=6, ff_dim=3072, dropout=0.2):
        super().__init__()
        self.pad_id = PAD_ID
        self.enc_emb = nn.Embedding(src_vocab, d_model, padding_idx=PAD_ID)
        self.dec_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD_ID)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout, batch_first=True, norm_first=True), n_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, n_heads, ff_dim, dropout, batch_first=True, norm_first=True), n_layers)
        self.output_proj = nn.Linear(d_model, tgt_vocab, bias=False)
        self.output_proj.weight = self.dec_emb.weight

    def forward(self, src, tgt_in):
        src_mask = (src == self.pad_id)
        tgt_mask = (tgt_in == self.pad_id)
        causal_mask = torch.triu(torch.ones(tgt_in.size(1), tgt_in.size(1), device=src.device), diagonal=1).bool()
        
        enc_out = self.encoder(self.pos_enc(self.enc_emb(src) * math.sqrt(768)), src_key_padding_mask=src_mask)
        dec_out = self.decoder(self.pos_enc(self.dec_emb(tgt_in) * math.sqrt(768)), enc_out, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=src_mask)
        return self.output_proj(dec_out)

# ===============================================================
# ENTROPY LOGIC (The Core Experiment)
# ===============================================================
@torch.no_grad()
def compute_teacher_logits(src_batch, tgt_in_batch, teacher1, teacher2):
    model1 = teacher1[1]
    model2 = teacher2[1]

    l1 = model1(input_ids=src_batch, attention_mask=(src_batch != PAD_ID), decoder_input_ids=tgt_in_batch).logits.detach()
    l2 = model2(input_ids=src_batch, attention_mask=(src_batch != PAD_ID), decoder_input_ids=tgt_in_batch).logits.detach()

    p1, lp1 = F.softmax(l1, dim=-1), F.log_softmax(l1, dim=-1)
    p2, lp2 = F.softmax(l2, dim=-1), F.log_softmax(l2, dim=-1)

    ent1 = -(p1 * lp1).sum(dim=-1, keepdim=True) + 1e-9
    ent2 = -(p2 * lp2).sum(dim=-1, keepdim=True) + 1e-9

    w1_raw, w2_raw = 1.0 / ent1, 1.0 / ent2
    w_total = w1_raw + w2_raw
    w1, w2 = w1_raw / w_total, w2_raw / w_total

    return (w1 * l1) + (w2 * l2)

# ===============================================================
# DATASET PREP
# ===============================================================
class SeqDataset(Dataset):
    def __init__(self, src, tin, tout): self.src, self.tin, self.tout = src, tin, tout
    def __len__(self): return len(self.src)
    def __getitem__(self, i): return self.src[i], self.tin[i], self.tout[i]

def pad_seqs(ids, max_l):
    arr = torch.full((len(ids), max_l), PAD_ID, dtype=torch.long)
    for i, s in enumerate(ids): arr[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return arr

# ===============================================================
# EVALUATION & SAVING TRANSLATIONS
# ===============================================================
@torch.no_grad()
def beam_decode(model, sp_tgt, src_seq):
    model.eval()
    src = torch.tensor(src_seq, dtype=torch.long, device=DEVICE).unsqueeze(0)
    src_mask = (src == PAD_ID)
    memory = model.encoder(model.pos_enc(model.enc_emb(src) * math.sqrt(768)), src_key_padding_mask=src_mask)
    
    beams = [(0.0, [sp_tgt.bos_id()])]
    for _ in range(MAX_TGT_LEN):
        new_beams = []
        for score, seq in beams:
            if seq[-1] == sp_tgt.eos_id():
                new_beams.append((score, seq))
                continue
            tgt = torch.tensor(seq, device=DEVICE).unsqueeze(0)
            tgt_mask = (tgt == PAD_ID)
            causal = torch.triu(torch.ones(tgt.size(1), tgt.size(1), device=DEVICE), diagonal=1).bool()
            dec_out = model.decoder(model.pos_enc(model.dec_emb(tgt) * math.sqrt(768)), memory, tgt_mask=causal, tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=src_mask)
            logits = model.output_proj(dec_out[:, -1])
            topk = torch.topk(torch.log_softmax(logits, dim=-1), BEAM_SIZE)
            for i in range(BEAM_SIZE):
                new_beams.append((score + topk.values[0, i].item(), seq + [topk.indices[0, i].item()]))
        beams = sorted(new_beams, key=lambda x: x[0] / (len(x[1])**0.7), reverse=True)[:BEAM_SIZE]
    
    best = beams[0][1][1:]
    if sp_tgt.eos_id() in best: best = best[:best.index(sp_tgt.eos_id())]
    return sp_tgt.decode(best)

def evaluate_and_save(model, sp_tgt, val_src_arr, val_refs, filename):
    preds = []
    for i in tqdm(range(min(500, len(val_src_arr))), desc=f"Decoding {filename}"):
        preds.append(beam_decode(model, sp_tgt, val_src_arr[i]))
    
    trunc_refs = val_refs[:len(preds)]
    bleu = sacrebleu.corpus_bleu(preds, [trunc_refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [trunc_refs]).score
    log(f"[{filename}] BLEU: {bleu:.2f} | chrF: {chrf:.2f}")

    # SAVE FOR STATISTICAL ANALYSIS LATER
    with open(filename, "w", encoding="utf-8") as f:
        for p in preds: f.write(p + "\n")
    return bleu, chrf

# ===============================================================
# TRAINING LOOPS
# ===============================================================
def train_loop(model, loader, epochs, lr, is_distill=False, t1=None, t2=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE=="cuda"))
    
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0
        for sb, tin, tout in tqdm(loader, desc=f"Epoch {ep}"):
            sb, tin, tout = sb.to(DEVICE), tin.to(DEVICE), tout.to(DEVICE)
            opt.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=(DEVICE=="cuda")):
                s_logits = model(sb, tin)
                if is_distill:
                    t_logits = compute_teacher_logits(sb, tin, t1, t2)
                    loss = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), tout.view(-1), ignore_index=PAD_ID) * 0.4
                    loss += F.kl_div(F.log_softmax(s_logits/2.0, dim=-1), F.softmax(t_logits/2.0, dim=-1), reduction="batchmean") * 4.0 * 0.6
                else:
                    loss = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), tout.view(-1), ignore_index=PAD_ID)
            
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item()
        log(f"Epoch {ep} Loss: {loss_sum/len(loader):.4f}")

# ===============================================================
# MAIN PIPELINE
# ===============================================================
def main():
    log("1. Preparing Data...")
    pairs = clean_pairs(read_pairs(DATA_PATH))
    random.shuffle(pairs)
    train_pairs, val_pairs = pairs[:-1000], pairs[-1000:]
    
    with open("tmp_s.txt","w") as f: f.write("\n".join([p[0] for p in train_pairs]))
    with open("tmp_t.txt","w") as f: f.write("\n".join([p[1] for p in train_pairs]))
    sp_src = train_sp("tmp_s.txt", "sp_src", 12000)
    sp_tgt = train_sp("tmp_t.txt", "sp_tgt", 16000)

    def encode_pairs(p_list):
        src_ids, tin_ids, tout_ids = [], [], []
        for s, t in p_list:
            si = [2] + sp_src.encode(s)[:MAX_SRC_LEN-2] + [3]
            ti = [2] + sp_tgt.encode(t)[:MAX_TGT_LEN-2] + [3]
            src_ids.append(si); tin_ids.append(ti[:-1]); tout_ids.append(ti[1:])
        return pad_seqs(src_ids, MAX_SRC_LEN), pad_seqs(tin_ids, MAX_TGT_LEN), pad_seqs(tout_ids, MAX_TGT_LEN)

    tr_s, tr_tin, tr_tout = encode_pairs(train_pairs)
    v_s, v_tin, v_tout = encode_pairs(val_pairs)
    val_refs = [p[1] for p in val_pairs]

    loader_p1 = DataLoader(SeqDataset(tr_s, tr_tin, tr_tout), batch_size=BATCH_SIZE, shuffle=True)
    loader_p2 = DataLoader(SeqDataset(tr_s, tr_tin, tr_tout), batch_size=DISTILL_BATCH_SIZE, shuffle=True)

    model = TransformerSeq2Seq(12000, 16000).to(DEVICE)

    # --- PHASE 1: SUPERVISED BASELINE ---
    log("2. Running Phase 1 (Supervised Baseline)...")
    train_loop(model, loader_p1, EPOCHS_PHASE1, 2e-4)
    evaluate_and_save(model, sp_tgt, v_s, val_refs, "preds_baseline_supervised.txt")
    torch.save(model.state_dict(), "student_phase1.pt")

    # --- PHASE 2: ENTROPY DISTILLATION ---
    log("3. Loading Teachers...")
    t1 = (None, AutoModelForSeq2SeqLM.from_pretrained(TEACHER_1).eval().to(DEVICE))
    t2 = (None, AutoModelForSeq2SeqLM.from_pretrained(TEACHER_2, trust_remote_code=True).eval().to(DEVICE))

    log("4. Running Phase 2 (Entropy Distillation)...")
    train_loop(model, loader_p2, EPOCHS_PHASE2, 1e-4, is_distill=True, t1=t1, t2=t2)
    
    # --- PHASE 3: FINE TUNING ---
    log("5. Running Phase 3 (Human Fine-Tuning)...")
    train_loop(model, loader_p1, EPOCHS_PHASE3, 2e-5)
    
    log("6. Final Evaluation...")
    evaluate_and_save(model, sp_tgt, v_s, val_refs, "preds_multi_entropy.txt")
    
    # Save the ground truth references so we can compare later!
    with open("refs_ground_truth.txt", "w", encoding="utf-8") as f:
        for r in val_refs[:500]: f.write(r + "\n")

    log("EXPERIMENT COMPLETE!")

if __name__ == "__main__":
    main()
