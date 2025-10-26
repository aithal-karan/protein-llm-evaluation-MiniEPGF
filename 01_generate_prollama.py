# 01_generate_prollama.py
# Generate controllable protein sequences with ProLLaMA, with clear progress + FASTA output.
# Usage: python 01_generate_prollama.py

import os, re, sys, time, json, random, logging
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging

# -------------------- CONFIG --------------------
MODEL_ID = "GreatCaptainNemo/ProLLaMA"   
OUT_DIR = Path("data")
OUT_FASTA = OUT_DIR / "gen_raw_prollama.fasta"

SUPERFAMILIES = [
    "CheY-like superfamily",
    "Thioredoxin-like superfamily",
    
]
N_PER_FAMILY = 150

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.8
TOP_P = 0.9
REPETITION_PENALTY = 1.05

MIN_LEN = 90
MAX_LEN = 480
SEED = 1234

DEVICE_MAP = "auto"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

AA20 = set("ACDEFGHIKLMNPQRSTVWY")

try:
    from tqdm import tqdm as _tqdm
    def trange(n, **kw): return _tqdm(range(n), **kw)
except Exception:
    def trange(n, **kw): return range(n)

# ---------- Logging / HF verbosity ----------
hf_logging.set_verbosity_info()          
hf_logging.enable_explicit_format()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("prollama")

def say(msg: str):
    print(msg, flush=True)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --------------- Prompt helpers ---------------
def prompt_generate(superfamily: str) -> str:
    return (
        "Instruction:[Generate by superfamily]\n"
        f"Input:Superfamily=<{superfamily}>\n"
        "Output:Seq=<"
    )

def prompt_classify(seq_text: str) -> str:
    
    return (
        "Instruction:[Determine superfamily]\n"
        f"Input:{seq_text}\n"
        "Output:Superfamily=<"
    )

# --------------- Tokenizer / Model ---------------
def load_model():
    
    try:
        tok = AutoTokenizer.from_pretrained(
            MODEL_ID, use_fast=True, trust_remote_code=True
        )
    except Exception as e_fast:
        say(f"[warn] fast tokenizer failed: {repr(e_fast)}")
        say("[info] falling back to slow tokenizer (tip: pip install tiktoken sentencepiece)")
        tok = AutoTokenizer.from_pretrained(
            MODEL_ID, use_fast=False, trust_remote_code=True
        )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=DEVICE_MAP,
        torch_dtype=DTYPE,
        trust_remote_code=True
    ).eval()
    return tok, mdl

# --------------- Text → Sequence parsing ---------------
def clean_seq(seq: str) -> str:
    seq = re.sub(r"[^A-Z]", "", seq)
    return "".join([c for c in seq if c in AA20])

def parse_seq_from_text(txt: str) -> str:
    flat = txt.replace("\n", " ")
    m = re.search(r"Seq\s*=\s*<\s*([A-Z]+?)\s*>", flat)
    if m:
        return clean_seq(m.group(1))
    # fallback: take the longest AA20 chunk length ≥ 20
    chunks = re.findall(r"[ACDEFGHIKLMNPQRSTVWY]{20,}", flat)
    return clean_seq(max(chunks, key=len)) if chunks else ""

# --------------- Generation ---------------
@torch.no_grad()
def generate_once(tokenizer, model, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def generate_for_superfamily(tokenizer, model, fam: str, n: int) -> List[Tuple[str, str, str]]:
    recs: List[Tuple[str, str, str]] = []
    pbar = trange(n, desc=f"[{fam}] generating", ncols=100)
    for i in pbar:
        txt = generate_once(tokenizer, model, prompt_generate(fam))
        seq = parse_seq_from_text(txt)
        ok = (MIN_LEN <= len(seq) <= MAX_LEN)
        if ok:
            recs.append((f"{fam.replace(' ','_')}_{i:03d}", fam, seq))
        
        try:
            pbar.set_postfix_str(f"kept={len(recs)}/{i+1}, last_len={len(seq) if seq else 0}")
        except Exception:
            pass
        # short preview
        if (i + 1) % 25 == 0:
            preview = txt.replace("\n", " ")[:140]
            say(f"[{fam}] sample {i+1}: kept={len(recs)} | last_len={len(seq)} | preview: {preview}...")
    return recs

def write_fasta(records: List[Tuple[str, str, str]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rid, fam, seq in records:
            f.write(f">{rid}|{fam}\n{seq}\n")

def quick_kmer_stat(seq: str, k=3) -> float:
    from collections import Counter
    if len(seq) < k: return 0.0
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    c = Counter(kmers)
    return round(c.most_common(1)[0][1] / max(1, len(kmers)), 3)

def classify_superfamily(tokenizer, model, seq: str) -> str:
    txt_in = f"Seq=<{seq}>"
    txt = generate_once(tokenizer, model, prompt_classify(txt_in))
    m = re.search(r"Superfamily=<\s*([^>]+?)\s*>", txt.replace("\n"," "))
    return m.group(1).strip() if m else "(no match)"

# --------------- Main ---------------
if __name__ == "__main__":
    set_seed(SEED)
    say(">> loading tokenizer/model …")
    tokenizer, model = load_model()
    say(">> model loaded; starting generation.")

    all_records: List[Tuple[str, str, str]] = []
    t_start = time.time()

    for fam in SUPERFAMILIES:
        t0 = time.time()
        recs = generate_for_superfamily(tokenizer, model, fam, N_PER_FAMILY)
        dt = time.time() - t0
        say(f"[{fam}] kept {len(recs)} / {N_PER_FAMILY} in {dt:.1f}s")
        all_records.extend(recs)

    write_fasta(all_records, OUT_FASTA)
    total_dt = time.time() - t_start
    say(f">> wrote {len(all_records)} sequences -> {OUT_FASTA} (total {total_dt:.1f}s)")

    
    if all_records:
        rid, fam, seq = all_records[0]
        info = {
            "id": rid,
            "fam": fam,
            "len": len(seq),
            "aa_valid": all(c in AA20 for c in seq),
            "top3mer_fraction": quick_kmer_stat(seq, 3)
        }
        say(f">> first record: {json.dumps(info)}")
        pred = classify_superfamily(tokenizer, model, seq)
        say(f">> predicted superfamily for first-gen seq: {pred}")

    say(">> done.")
