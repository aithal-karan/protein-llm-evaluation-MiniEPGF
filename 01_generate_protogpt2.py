# 01_generate_protgpt2.py  — 
import re, sys, time, json, random
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging

# ---------------- CONFIG ----------------
MODEL = "nferruz/ProtGPT2"
OUT_DIR = Path("data")
OUT_FASTA = OUT_DIR / "gen_raw_protgpt2.fasta"


FAMILIES = [
  "CheY-like superfamily",
  "Thioredoxin-like superfamily",
]
N_PER_FAMILY = 150

# ProtGPT2 sampling defaults from paper repos: high top_k, nucleus, mild rep-penalty
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
TOP_P = 0.94
TOP_K = 950
REPETITION_PENALTY = 1.2

# Acceptance bounds
MIN_LEN = 90
MAX_LEN = 480
MAX_RETRIES = 4

SEED = 42
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
AA20 = set("ACDEFGHIKLMNPQRSTVWY")

try:
    from tqdm import tqdm as _tqdm
    def trange(n, **kw): return _tqdm(range(n), **kw)
except Exception:
    def trange(n, **kw): return range(n)

hf_logging.set_verbosity_info()
hf_logging.enable_explicit_format()

def say(msg): print(msg, flush=True)

def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL, use_safetensors=True, torch_dtype=DTYPE
    ).eval()
    if torch.cuda.is_available(): mdl = mdl.cuda()
    return tok, mdl

def clean_to_AA20(txt: str) -> str:
    # Strip everything to the 20 canonical AAs, keep order
    return "".join(ch for ch in txt if ch in AA20)

def parse_seq(txt: str) -> str:
    
    flat = clean_to_AA20(txt)
    
    chunks = re.findall(r"[ACDEFGHIKLMNPQRSTVWY]+", flat)
    return max(chunks, key=len) if chunks else ""

@torch.no_grad()
def sample_once(tok, mdl, prompt_text: str) -> str:
    inps = tok(prompt_text, return_tensors="pt")
    if torch.cuda.is_available(): inps = {k: v.cuda() for k,v in inps.items()}
    out = mdl.generate(
        **inps,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_new_tokens=MAX_NEW_TOKENS,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,  
    )
    return tok.decode(out[0], skip_special_tokens=True)

def generate_valid_seq(tok, mdl) -> str:
    # seed with "M" so many outputs start with Met and look protein-like
    prompt = "M"
    for _ in range(MAX_RETRIES):
        txt = sample_once(tok, mdl, prompt)
        seq = parse_seq(txt)
        if MIN_LEN <= len(seq) <= MAX_LEN:
            return seq
        
        global TOP_P, TEMPERATURE
        TEMPERATURE = min(1.2, TEMPERATURE + 0.05)
        TOP_P = min(0.98, TOP_P + 0.01)
        prompt = "M" * random.randint(1, 3)
    return ""  

def write_fasta(records: List[Tuple[str,str,str]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rid, fam, seq in records:
            f.write(f">{rid}|{fam}\n{seq}\n")

if __name__ == "__main__":
    set_seed(SEED)
    say(">> loading ProtGPT2 …")
    tok, mdl = load_model()
    say(">> model ready; generating.")

    all_recs: List[Tuple[str,str,str]] = []
    t0 = time.time()
    for fam in FAMILIES:
        kept = 0
        for i in trange(N_PER_FAMILY, desc=f"[ProtGPT2:{fam}]"):
            seq = generate_valid_seq(tok, mdl)
            if seq:
                kept += 1
                all_recs.append((f"ProtGPT2_{fam.replace(' ','_')}_{i:03d}", fam, seq))
            if (i+1) % 25 == 0:
                say(f"[ProtGPT2:{fam}] @ {i+1}: kept={kept} (last_len={len(seq) if seq else 0})")
        say(f"[ProtGPT2:{fam}] kept {kept} / {N_PER_FAMILY}")

    write_fasta(all_recs, OUT_FASTA)
    say(f">> wrote {len(all_recs)} sequences -> {OUT_FASTA} (total {time.time()-t0:.1f}s)")
    if all_recs:
        rid, fam, seq = all_recs[0]
        from collections import Counter
        kmers = [seq[i:i+3] for i in range(len(seq)-2)]
        kmax = round(Counter(kmers).most_common(1)[0][1] / max(1, len(kmers)), 3) if kmers else 0.0
        say(json.dumps({"id": rid, "fam": fam, "len": len(seq), "aa_valid": all(c in AA20 for c in seq), "top3mer_fraction": kmax}))
    say(">> done.")
