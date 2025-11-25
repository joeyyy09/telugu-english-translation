# main.py
# ===============================================================
# English -> Telugu NMT (Teacher: NLLB, Student: Bi-GRU in PyTorch)
# Tokenizers: SentencePiece (separate EN and TE models)
# ===============================================================

import os
import random
import argparse
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sentencepiece as spm
import sacrebleu

# -----------------------
# Config / Hyperparams
# -----------------------
DATA_PATH = "English Telugu Data.txt"
TEACHER_MODEL = "facebook/nllb-200-distilled-600M"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device on:", DEVICE)

# SentencePiece params
SPM_EN_MODEL = "spm_en"
SPM_TE_MODEL = "spm_te"
SPM_EN_VOCAB = 8000   # adjust if you want smaller/larger
SPM_TE_VOCAB = 8000

# Model/training params
BATCH_SIZE_TEACHER = 32    # teacher translation batch
BATCH_SIZE_TRAIN = 64      # training batch (reduce if OOM)
EPOCHS = 5
EMBED_DIM = 256
ENC_HID = 256              # per-direction hidden dim
DEC_HID = ENC_HID * 2      # decoder hidden must equal concat of both directions
LEARNING_RATE = 1e-3
MAX_ENC_LEN = 128
MAX_DEC_LEN = 128
TEACHER_FORCING_RATIO = 0.5
SAVE_PATH = "best_student_model.pt"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

# -----------------------
# Utilities
# -----------------------
def read_pairs(path: str) -> List[Tuple[str,str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Put dataset in same folder.")
    src, tgt = [], []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if "++++$++++" not in ln:
                continue
            a,b = ln.strip().split("++++$++++")
            src.append(a.strip())
            tgt.append(b.strip())
    return list(zip(src,tgt))

# -----------------------
# SentencePiece helpers
# -----------------------
def train_spm(input_texts: List[str], model_prefix: str, vocab_size: int, model_type="bpe"):
    # writes model_prefix.model and model_prefix.vocab
    temp_file = model_prefix + "_train.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        for line in input_texts:
            if line:
                f.write(line.replace("\n"," ") + "\n")
    spm_cmd = (
        f"--input={temp_file} --model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} --model_type={model_type} "
        f"--character_coverage=1.0 --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3"
    )
    spm.SentencePieceTrainer.Train(spm_cmd)
    os.remove(temp_file)
    print(f"Trained SentencePiece model: {model_prefix}.model")

def load_spm(model_prefix: str) -> spm.SentencePieceProcessor:
    model_file = model_prefix + ".model"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"{model_file} not found.")
    sp = spm.SentencePieceProcessor()
    sp.Load(model_file)
    return sp

def encode_with_sp(sp: spm.SentencePieceProcessor, texts: List[str], max_len: int, add_bos_eos=True):
    encs = []
    bos = sp.bos_id() if sp.bos_id() is not None else 2
    eos = sp.eos_id() if sp.eos_id() is not None else 3
    pad = sp.pad_id() if sp.pad_id() is not None else 0
    for t in texts:
        ids = sp.EncodeAsIds(t)
        if add_bos_eos:
            ids = [bos] + ids + [eos]
        if len(ids) > max_len:
            ids = ids[:max_len-1] + [eos]  # ensure eos at end
        # pad
        if len(ids) < max_len:
            ids = ids + [pad] * (max_len - len(ids))
        encs.append(ids)
    return np.array(encs, dtype=np.int64)

# -----------------------
# Teacher: batched translation (HF Transformers)
# -----------------------
def load_teacher(name=TEACHER_MODEL):
    print("Loading teacher:", name)
    tkn = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name).to(DEVICE)
    return tkn, model

def translate_with_teacher(tokenizer, model, texts: List[str], batch_size=BATCH_SIZE_TEACHER, max_length=MAX_DEC_LEN, src_lang="eng_Latn", tgt_lang="tel_Telu"):
    tokenizer.src_lang = src_lang
    results = []
    # try forcing target language token id if available
    try:
        forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    except Exception:
        forced_bos_id = None

    for i in tqdm(range(0, len(texts), batch_size), desc="Teacher Fast Batch"):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_ENC_LEN).to(DEVICE)
        gen_kwargs = dict(max_length=max_length, num_beams=1, do_sample=False)
        if forced_bos_id is not None:
            gen_kwargs["forced_bos_token_id"] = forced_bos_id
        with torch.no_grad():
            gen = model.generate(**enc, **gen_kwargs)
        dec = tokenizer.batch_decode(gen, skip_special_tokens=True)
        results.extend(dec)
    return results

# -----------------------
# Dataset & DataLoader
# -----------------------
class Seq2SeqDataset(Dataset):
    def __init__(self, enc_src, enc_tgt_in, enc_tgt_out):
        # all numpy arrays (N, L)
        self.enc_src = enc_src.astype(np.int64)
        self.enc_tgt_in = enc_tgt_in.astype(np.int64)
        self.enc_tgt_out = enc_tgt_out.astype(np.int64)
        assert len(self.enc_src) == len(self.enc_tgt_in) == len(self.enc_tgt_out)

    def __len__(self):
        return len(self.enc_src)

    def __getitem__(self, idx):
        return (self.enc_src[idx], self.enc_tgt_in[idx], self.enc_tgt_out[idx])

def collate_fn(batch):
    srcs, ins, outs = zip(*batch)
    return (torch.tensor(np.stack(srcs), dtype=torch.long),
            torch.tensor(np.stack(ins), dtype=torch.long),
            torch.tensor(np.stack(outs), dtype=torch.long))

# -----------------------
# PyTorch Student model
# -----------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True, batch_first=True)

    def forward(self, src):
        # src: [B, L]
        emb = self.embedding(src)                # [B, L, E]
        outputs, hidden = self.gru(emb)          # hidden: [2, B, enc_hid] (fwd, bwd)
        # concat forward/back hidden to shape [B, 2*enc_hid]
        h_fwd = hidden[0]   # [B, enc_hid]
        h_bwd = hidden[1]   # [B, enc_hid]
        h_cat = torch.cat((h_fwd, h_bwd), dim=1) # [B, 2*enc_hid]
        return outputs, h_cat

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, dec_hid_dim, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, dec_hid_dim, batch_first=True)
        self.out = nn.Linear(dec_hid_dim, output_dim)

    def forward(self, input_tokens, hidden):
        # input_tokens: [B, T]
        emb = self.embedding(input_tokens)  # [B, T, E]
        # hidden: [B, dec_hid] -> GRU expects [num_layers, B, H]
        outputs, hn = self.gru(emb, hidden.unsqueeze(0))
        logits = self.out(outputs)          # [B, T, V]
        return logits, hn.squeeze(0)        # return new hidden as [B, dec_hid]

class Seq2Seq(nn.Module):
    def __init__(self, enc: Encoder, dec: Decoder):
        super().__init__()
        self.encoder = enc
        self.decoder = dec

    def forward(self, src, trg_in, teacher_forcing=True):
        # src: [B, S], trg_in: [B, T]
        batch_size, trg_len = trg_in.size()
        device = src.device
        vocab = self.decoder.out.out_features
        outputs = torch.zeros(batch_size, trg_len, vocab, device=device)
        _, enc_hidden = self.encoder(src)     # enc_hidden: [B, dec_hid]
        dec_hidden = enc_hidden               # shape [B, dec_hid]
        if teacher_forcing:
            # single full pass: feed trg_in (teacher forcing)
            logits, _ = self.decoder(trg_in, dec_hidden)  # [B, T, V]
            return logits
        else:
            # step-by-step
            input_tok = trg_in[:, 0].unsqueeze(1)  # [B,1]
            hidden = dec_hidden
            for t in range(trg_len):
                logits_step, hidden = self.decoder(input_tok, hidden)  # logits_step: [B,1,V]
                out_step = logits_step[:, -1, :]                       # [B, V]
                outputs[:, t, :] = out_step
                # choose next token (greedy no teacher forcing here)
                top1 = out_step.argmax(dim=1).unsqueeze(1)
                input_tok = top1
            return outputs

# -----------------------
# Training & Eval
# -----------------------
def train_epoch(model, loader, optimizer, criterion, device, teacher_forcing_ratio=TEACHER_FORCING_RATIO):
    model.train()
    total_loss = 0.0
    for src_batch, dec_in_batch, dec_out_batch in tqdm(loader, desc="Train Batches"):
        src_batch = src_batch.to(device)
        dec_in_batch = dec_in_batch.to(device)
        dec_out_batch = dec_out_batch.to(device)
        optimizer.zero_grad()
        teacher = (random.random() < teacher_forcing_ratio)
        # choose full teacher pass or stepwise
        logits = model(src_batch, dec_in_batch, teacher_forcing=teacher)  # [B, T, V]
        B,T,V = logits.shape
        logits_flat = logits.view(B*T, V)
        targets_flat = dec_out_batch.view(B*T)
        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_bleu(model, enc_src, dec_in, raw_truths, sp_tgt, device, max_eval=200):
    model.eval()
    preds = []
    bos = sp_tgt.bos_id(); eos = sp_tgt.eos_id(); pad = sp_tgt.pad_id()
    for i in range(min(max_eval, len(enc_src))):
        src = torch.tensor(enc_src[i:i+1], dtype=torch.long).to(device)
        # greedy decode
        _, enc_hidden = model.encoder(src)
        hidden = enc_hidden
        tok = torch.tensor([[bos]], dtype=torch.long).to(device)
        words = []
        for _ in range(MAX_DEC_LEN):
            logits_step, hidden = model.decoder(tok, hidden)
            next_logits = logits_step[:, -1, :]
            top1 = next_logits.argmax(dim=1)
            t = top1.item()
            if t == eos:
                break
            piece = sp_tgt.IdToPiece(t) if hasattr(sp_tgt, "IdToPiece") else sp_tgt.id_to_piece(t)
            # SentencePiece piece may be raw piece; we'll decode later but append
            words.append(piece)
            tok = top1.unsqueeze(1)
        # reconstruct sentence from pieces
        pred_text = sp_tgt.DecodePieces(words) if hasattr(sp_tgt, "DecodePieces") else sp_tgt.decode_pieces(words)
        preds.append(pred_text)
    refs = [raw_truths[:len(preds)]]
    bleu = sacrebleu.corpus_bleu(preds, refs)
    return bleu.score, preds

# -----------------------
# Main
# -----------------------
def main():
    print("Reading dataset...")
    pairs = read_pairs(DATA_PATH)
    src_texts = [p[0] for p in pairs]
    tgt_texts = [p[1] for p in pairs]
    N = len(pairs)
    print("Total pairs:", N)

    # ---------------- Teacher
    tokenizer, teacher_model = load_teacher(TEACHER_MODEL)
    print("Running teacher to produce distilled targets (this may take time)...")
    teacher_outs = translate_with_teacher(tokenizer, teacher_model, src_texts, batch_size=BATCH_SIZE_TEACHER, max_length=MAX_DEC_LEN)

    assert len(teacher_outs) == N

    # ---------------- SentencePiece training/loading
    # Train if model files not present
    if not (os.path.exists(SPM_EN_MODEL + ".model") and os.path.exists(SPM_TE_MODEL + ".model")):
        print("Training SentencePiece models (this may take a while)...")
        # For encoder we train on source (English)
        train_spm(src_texts, SPM_EN_MODEL, vocab_size=SPM_EN_VOCAB, model_type="bpe")
        # For decoder we train on teacher outputs (Telugu), so student learns teacher's distribution
        train_spm(teacher_outs, SPM_TE_MODEL, vocab_size=SPM_TE_VOCAB, model_type="bpe")
    else:
        print("Found existing SentencePiece models. Loading...")

    sp_en = load_spm(SPM_EN_MODEL)
    sp_te = load_spm(SPM_TE_MODEL)

    src_vocab = sp_en.get_piece_size()
    tgt_vocab = sp_te.get_piece_size()
    pad_en = sp_en.pad_id(); pad_te = sp_te.pad_id()
    bos_te = sp_te.bos_id(); eos_te = sp_te.eos_id()
    print("Vocab sizes - en:", src_vocab, "te:", tgt_vocab, "pad_te:", pad_te, "bos:", bos_te, "eos:", eos_te)

    # ---------------- Encode with SPM and pad to max lengths (cap lengths)
    max_enc_len = min(MAX_ENC_LEN, max(len(sp_en.EncodeAsIds(s)) + 2 for s in src_texts))  # +2 for bos/eos if used later
    max_dec_len = min(MAX_DEC_LEN, max(len(sp_te.EncodeAsIds(s)) + 2 for s in teacher_outs))
    # enforce caps
    max_enc_len = max(8, max_enc_len)
    max_dec_len = max(8, max_dec_len)
    print("Max lengths - enc:", max_enc_len, "dec:", max_dec_len)

    enc_src = encode_with_sp(sp_en, src_texts, max_enc_len, add_bos_eos=True)
    enc_tgt_all = encode_with_sp(sp_te, teacher_outs, max_dec_len, add_bos_eos=True)

    # Prepare decoder input (shifted)
    # dec_in: <bos> w1 w2 ... (length = max_dec_len)
    # dec_out: w1 w2 ... <eos> (length = max_dec_len)
    bos_id = sp_te.bos_id(); eos_id = sp_te.eos_id(); pad_id = sp_te.pad_id()
    assert pad_id == 0, "We trained spm with pad_id=0; expected pad_id 0. If not, adjust code."

    dec_in = np.copy(enc_tgt_all)
    dec_out = np.zeros_like(enc_tgt_all, dtype=np.int64)
    # shift left: target output at position t is decoder input at t+1
    dec_out[:, :-1] = enc_tgt_all[:, 1:]
    dec_out[:, -1] = pad_id  # last token (if truncated) -> pad

    # ensure any id < vocab
    if enc_src.max() >= src_vocab or enc_tgt_all.max() >= tgt_vocab:
        raise ValueError("Unexpected token id >= vocab size. Check SentencePiece models and IDs.")

    # ---------------- Dataloader
    dataset = Seq2SeqDataset(enc_src, dec_in, dec_out)
    # On Windows keep num_workers=0 to avoid issues
    loader = DataLoader(dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=(DEVICE=="cuda"))

    # ---------------- Build models
    encoder = Encoder(input_dim=src_vocab, emb_dim=EMBED_DIM, enc_hid_dim=ENC_HID, pad_idx=pad_en)
    decoder = Decoder(output_dim=tgt_vocab, emb_dim=EMBED_DIM, dec_hid_dim=DEC_HID, pad_idx=pad_te)
    model = Seq2Seq(encoder, decoder).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_te)

    # ---------------- Training loop
    best_loss = float("inf")
    print("Starting training on device:", DEVICE)
    try:
        for ep in range(1, EPOCHS+1):
            print(f"Epoch {ep}/{EPOCHS}")
            train_loss = train_epoch(model, loader, optimizer, criterion, DEVICE)
            print(f"Train loss: {train_loss:.4f}")
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(model.state_dict(), SAVE_PATH)
                print("Saved", SAVE_PATH)
            # quick BLEU sample
            bleu, preds = evaluate_bleu(model, enc_src, dec_in, tgt_texts, sp_te, DEVICE, max_eval=200)
            print(f"Quick BLEU (sample) : {bleu:.2f}")
    except RuntimeError as e:
        # often CUDA assert; give debugging hint
        print("RuntimeError during training:", e)
        if "device-side assert" in str(e) or "CUDA error" in str(e):
            print("Device-side assert detected. Possible causes:")
            print("- target token indices >= decoder vocabulary size")
            print("- wrong pad_id mismatch between SentencePiece and model")
            print("To debug locally set environment variable: CUDA_LAUNCH_BLOCKING=1 python main.py")
        raise

    # ---------------- Examples
    print("\nExamples:")
    for i in range(min(5, len(enc_src))):
        src = enc_src[i:i+1]
        teacher = teacher_outs[i]
        # greedy decode
        with torch.no_grad():
            src_t = torch.tensor(src, dtype=torch.long).to(DEVICE)
            _, enc_hidden = model.encoder(src_t)
            hidden = enc_hidden
            tok = torch.tensor([[bos_id]], dtype=torch.long).to(DEVICE)
            pieces = []
            for _ in range(max_dec_len):
                logits_step, hidden = model.decoder(tok, hidden)
                next_logits = logits_step[:, -1, :]
                top1 = next_logits.argmax(dim=1)
                t = top1.item()
                if t == eos_id:
                    break
                piece = sp_te.IdToPiece(t) if hasattr(sp_te, "IdToPiece") else sp_te.id_to_piece(t)
                pieces.append(piece)
                tok = top1.unsqueeze(1)
            pred = sp_te.DecodePieces(pieces) if hasattr(sp_te, "DecodePieces") else sp_te.decode_pieces(pieces)
        print(f"Src: {src_texts[i]}\nTchr: {teacher}\nStu: {pred}\n---")

if __name__ == "__main__":
    main()

    
