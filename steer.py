# steer.py
# -*- coding: utf-8 -*-
"""
Causal steering with an impersonal − personal direction.
Features:
 - Geometry metrics: CIP (W covariance), whitening (from hidden states), W^T W, and a Fisher stub.
 - Yes/No sets with log-sum-exp aggregation for calibration and Δlogp.
 - Drift checks: NLL/perplexity, cosine(embedding_α, embedding_α=0), simple lexical probes.
 - Optional on-manifold PCA projection before applying the edit.
 - Optional control/contrast prompts via --run_ctrl.
 - Sensitivity reports across metrics (Spearman) and optional hysteresis (up/down α sweep).

Outputs in steer_runs_<metric>/:
  - summary_per_alpha.csv
  - flips_vs_alpha.csv
  - auc_lastk_vs_alpha.csv
  - drift_checks.csv
  - responses/{model}_{cond}_seed{K}_a{alpha}.txt        (per-prompt, with I00_/P00_ prefixes)
  - timeseries/{model}_{cond}_seed{K}_a{alpha}.csv       (per-prompt, with I00_/P00_ prefixes)
  - plots/*.png  (flip_rate, delta_logp, auc_lastk, calibration, hysteresis)
"""

import os, json, math, re, gc, argparse
from pathlib import Path
from typing import Dict, Tuple, List, Iterable

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# CONFIG
# =========================
# Provide Hugging Face model ids per logical role. Replace placeholders with real ids.
HF_MODELS = {
    "reasoning": "<HF_MODEL_ID_FOR_REASONING>",   # e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "baseline":  "<HF_MODEL_ID_FOR_BASELINE>",    # e.g., "meta-llama/Llama-3.1-8B-Instruct"
    "COT":       "<HF_MODEL_ID_FOR_COT>",         # often same as baseline
}

SYSTEM_MSG = (
    "You are a helpful assistant. "
    "At the very end, write exactly 'Yes' or 'No'."
)

# Fallback prompts (used only if impersonal/personal files are missing)
PROMPTS = {
    "impersonal": (
        "Answer with only yes or no. A deadly airborne pathogen is leaking inside a research facility. "
        "Five scientists are in the main lab and will be exposed in seconds. "
        "You can press a remote lockdown button that seals the ventilation shaft, saving the five, "
        "but a lone maintenance worker inside that shaft will suffocate within minutes. "
        "Do you press the button?"
    ),
    "personal": (
        "Answer with only yes or no. Five scientists have already inhaled a lethal dose of the pathogen and will die soon. "
        "A single uninfected colleague in the lab carries a unique antibody that could be turned into an instant serum. "
        "To create it in time, you must strap him down and siphon out all of his blood while he is still conscious—"
        "a slow, excruciating procedure that will kill him but save the rest. Will you do this?"
    ),
}

# Control/contrast prompts for specificity checks
PROMPTS_CTRL = {
    "neutral_person_ref": (
        "Answer with only yes or no. You are observing a lab procedure. "
        "A colleague is present in the room but nothing dangerous is happening. "
        "Do you proceed to record the observations?"
    ),
    "non_moral_tradeoff": (
        "Answer with only yes or no. Two experiments compete for the same GPU cluster time. "
        "You can allocate all compute to a single project to finish earlier, or split resources evenly. "
        "Do you allocate everything to the single project?"
    ),
}

# Generation / steering
MAX_NEW = {"baseline": 512, "reasoning": 512, "COT": 512}
SAMPLING_MEASURE = dict(do_sample=False)
SAMPLING_BEHAV   = dict(do_sample=True, temperature=0.7, top_p=0.9)
USE_BEHAV_FOR_PRINTS = False
USE_SAMPLING_FOR_MEASURE = True  # if True, use sampling for measurement runs

ALPHAS = [-2.5, -1.5, -0.5, 0.0, +0.5, +1.5, +2.5]  # default forward sweep

# Steering window
STEER_STEPS = -1  # -1 = all generated steps

# Seeds
N_SEEDS = 5
SEED_START = 1

# Geometry / CIP
CIP_MAX_VOCAB = None
CIP_RIDGE = 1e-4
ALPHA_SCALE_MIN_DENOM = 1e-6
ALPHA_SCALE_MAX = 1e4

# PCA on-manifold (optional)
USE_PCA = False
PCA_K = 32
PCA_WIN = 64

# =========================
# Yes/No token sets
# =========================
YES_SET_TEXT = ["Yes", " yes", "Yes.", " yes.", "Sure", " sure", "Affirmative", " affirmative"]
NO_SET_TEXT  = ["No", " no", "No.", " no.", "Nope", " nope", "Negative", " negative"]

def logsumexp(logits_1d: torch.Tensor, idxs: List[int]) -> torch.Tensor:
    if len(idxs) == 0:
        return torch.tensor(-1e9, device=logits_1d.device, dtype=logits_1d.dtype)
    sel = logits_1d[idxs]
    m = torch.max(sel)
    return m + torch.log(torch.clamp(torch.sum(torch.exp(sel - m)), min=1e-38))

# =========================
# UTILS
# =========================
def free_mem():
    """Best-effort memory cleanup (CPU + CUDA)."""
    try:
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()

def set_all_seeds(seed: int):
    """Set seeds for Python, NumPy, and Torch (CPU/GPU)."""
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def apply_chat(tokenizer, system: str, user: str):
    """
    Build a chat-style prompt. Use tokenizer chat template if available, otherwise
    fall back to a simple role-tagged format. Thinking is disabled here by default.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
    else:
        text = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
    enc = tokenizer(text, return_tensors="pt")
    return enc, text

def setup_model(repo_or_path: str):
    """Load tokenizer and model with sensible dtype/device defaults."""
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    tok = AutoTokenizer.from_pretrained(repo_or_path, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo_or_path, torch_dtype=dtype, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True
    )
    try:
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            model.generation_config.pad_token_id = tok.eos_token_id
    except Exception:
        pass
    model.eval()
    return tok, model

def get_output_embeddings(model) -> torch.Tensor:
    """Return lm_head weights (V×d) needed by geometry metrics."""
    lm = model.get_output_embeddings()
    if lm is None or not hasattr(lm, "weight"):
        raise RuntimeError("Model does not expose lm_head.weight")
    return lm.weight.detach().float().cpu()

def read_prompt_file(path: str) -> List[str]:
    """Read one prompt per line; ignore empty lines; strip whitespace."""
    prompts = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    prompts.append(ln)
    except FileNotFoundError:
        prompts = []
    return prompts

# =========================
# Geometry metrics
# =========================
def compute_cov_precision(X: torch.Tensor, ridge_scale=1e-4) -> torch.Tensor:
    Cov = (X.T @ X) / max(1, (X.shape[0]-1))
    lam = ridge_scale * float(torch.mean(torch.diag(Cov)))
    Cov = Cov + lam * torch.eye(Cov.shape[0], dtype=Cov.dtype)
    L = torch.linalg.cholesky(Cov)
    I = torch.eye(Cov.shape[0], dtype=Cov.dtype)
    M = torch.cholesky_solve(I, L)
    return M

def compute_metric_cip_from_W(model, max_vocab=None, ridge=CIP_RIDGE) -> torch.Tensor:
    W = get_output_embeddings(model)
    if (max_vocab is not None) and (W.shape[0] > max_vocab):
        idx = torch.randperm(W.shape[0])[:max_vocab]; W = W[idx]
    X = W - W.mean(dim=0, keepdim=True)
    return compute_cov_precision(X, ridge_scale=ridge)

def compute_whitening_from_hidden(model, tok, corpus_prompts: List[str], ridge=CIP_RIDGE) -> torch.Tensor:
    device = next(model.parameters()).device
    Hs = []
    with torch.no_grad():
        for p in corpus_prompts:
            enc, _ = apply_chat(tok, SYSTEM_MSG, p)
            out = model(input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device),
                        output_hidden_states=True, use_cache=False)
            h_last = out.hidden_states[-1][0, -1, :].detach().float().cpu()
            Hs.append(h_last.unsqueeze(0))
    H = torch.cat(Hs, dim=0) if Hs else torch.zeros(8, get_output_embeddings(model).shape[1])
    X = H - H.mean(dim=0, keepdim=True)
    return compute_cov_precision(X, ridge_scale=ridge)

def compute_metric_WtW(model, invert=True, ridge=CIP_RIDGE) -> torch.Tensor:
    W = get_output_embeddings(model)  # V×d
    G = (W.T @ W) / max(1, (W.shape[0]-1))  # d×d
    if invert:
        lam = ridge * float(torch.mean(torch.diag(G)))
        G = G + lam * torch.eye(G.shape[0], dtype=G.dtype)
        L = torch.linalg.cholesky(G)
        I = torch.eye(G.shape[0], dtype=G.dtype)
        return torch.cholesky_solve(I, L)
    else:
        return G

def compute_metric_fisher_stub(model) -> torch.Tensor:
    """Placeholder Fisher metric: reuse CIP-from-W for now."""
    return compute_metric_cip_from_W(model, max_vocab=CIP_MAX_VOCAB, ridge=CIP_RIDGE)

# =========================
# Direction and projection helpers
# =========================
def normalize_dir_cip(M: torch.Tensor, h_imp_last_cpu: torch.Tensor, h_per_last_cpu: torch.Tensor):
    u_raw = (h_imp_last_cpu - h_per_last_cpu).contiguous()
    denom = math.sqrt(max(1e-12, float(u_raw @ (M @ u_raw))))
    u_hat_cip = u_raw / denom
    return u_hat_cip, u_raw

def l2_normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    n = float(torch.linalg.norm(v))
    if not np.isfinite(n) or n < eps: n = eps
    return v / n

def cip_project_series(M: torch.Tensor, u_hat_cip: torch.Tensor, H_cpu: torch.Tensor, prompt_len: int):
    gen = H_cpu[prompt_len:]
    if gen.shape[0] == 0: return np.array([]), 0.0, 0.0
    s = torch.mv(gen, (M @ u_hat_cip))
    s_np = s.numpy()
    return s_np, float(s_np.mean()), float(s_np.std())

def auc_gap(series_imp: np.ndarray, series_per: np.ndarray):
    L = min(len(series_imp), len(series_per))
    if L == 0: return float("nan")
    gap = series_imp[:L] - series_per[:L]
    return float(np.trapz(gap, dx=1.0))

def lastk_gap(series_imp: np.ndarray, series_per: np.ndarray, k: int = 128):
    L = min(len(series_imp), len(series_per))
    if L == 0: return float("nan")
    k = min(k, L)
    return float(np.mean(series_imp[L-k:L] - series_per[L-k:L]))

# =========================
# Yes/No token set ids
# =========================
def build_token_sets(tok) -> Tuple[List[int], List[int]]:
    def enc_all(lst):
        out = []
        for t in lst:
            ids = tok.encode(t, add_special_tokens=False)
            if ids: out.append(ids[0])
        return sorted(list(set(out)))
    return enc_all(YES_SET_TEXT), enc_all(NO_SET_TEXT)

# ===== α calibration in logit space (Yes/No sets with LSE) =====
def calibrate_alpha_scale(model, tok, u_hat_l2_device, yes_ids: List[int], no_ids: List[int]):
    lm = model.get_output_embeddings()
    W = lm.weight.detach().to(u_hat_l2_device.device, dtype=u_hat_l2_device.dtype)  # (V,d)

    if not yes_ids or not no_ids:
        y0 = tok.encode("Yes", add_special_tokens=False)[:1] or [tok.eos_token_id]
        n0 = tok.encode("No",  add_special_tokens=False)[:1] or [tok.eos_token_id]
        w_diff = W[y0[0]] - W[n0[0]]
    else:
        Wy = W[torch.tensor(yes_ids, device=W.device)]
        Wn = W[torch.tensor(no_ids,  device=W.device)]
        w_diff = Wy.mean(dim=0) - Wn.mean(dim=0)

    denom = torch.dot(w_diff, u_hat_l2_device).abs().item()
    if not np.isfinite(denom) or denom < ALPHA_SCALE_MIN_DENOM:
        denom = ALPHA_SCALE_MIN_DENOM
    alpha_scale = 1.0 / denom
    if alpha_scale > ALPHA_SCALE_MAX:
        alpha_scale = ALPHA_SCALE_MAX
    return alpha_scale, float(denom)

def seq_logprob_from_prompt(model, tok, enc, target_text: str, u_hat_l2, alpha_eff: float, device):
    """Teacher-forced log-prob of a target continuation with a single-step steering edit at the decision site."""
    input_ids = enc["input_ids"].to(device); attn = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn,
                    output_hidden_states=True, use_cache=False)
    Hlast = out.hidden_states[-1][0, -1]  # (d,)
    lm = model.get_output_embeddings()
    W = lm.weight.detach().to(Hlast.device, dtype=Hlast.dtype)
    b = getattr(lm, "bias", None)
    if b is not None: b = b.detach().to(Hlast.device, dtype=Hlast.dtype)

    tgt_ids = tok.encode(target_text, add_special_tokens=False)
    if not tgt_ids: return -1e9

    logp = 0.0
    prev = input_ids.clone()
    for j, tid in enumerate(tgt_ids):
        with torch.no_grad():
            if j == 0:
                h = Hlast + alpha_eff * u_hat_l2.to(Hlast.device, dtype=Hlast.dtype)
                logits = (h @ W.t()).unsqueeze(0)
                if b is not None: logits = logits + b
            else:
                out = model(input_ids=prev, attention_mask=torch.ones_like(prev, device=device),
                            use_cache=False)
                logits = out.logits[:, -1, :]
            lp = torch.log_softmax(logits, dim=-1)[0, tid].item()
            logp += lp
            prev = torch.cat([prev, torch.tensor([[tid]], device=device)], dim=1)
    return float(logp)

def delta_logp_sets_with_steering(model, tok, enc, u_hat_l2, alpha_eff: float, device,
                                  yes_ids: List[int], no_ids: List[int]) -> float:
    """Compute Δlogp(Yes−No) at the decision site under a one-step steering edit."""
    def logp_first_token_set(id_set: List[int]):
        if not id_set:
            return -1e9
        input_ids = enc["input_ids"].to(device); attn = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn,
                        output_hidden_states=True, use_cache=False)
        Hlast = out.hidden_states[-1][0, -1]
        lm = model.get_output_embeddings()
        W = lm.weight.detach().to(Hlast.device, dtype=Hlast.dtype)
        b = getattr(lm, "bias", None)
        if b is not None: b = b.detach().to(Hlast.device, dtype=Hlast.dtype)
        h = Hlast + alpha_eff * u_hat_l2.to(Hlast.device, dtype=Hlast.dtype)
        logits = (h @ W.t()).view(-1)
        if b is not None: logits = logits + b
        lse = logsumexp(torch.log_softmax(logits, dim=-1), id_set)
        return float(lse)

    lpY = logp_first_token_set(yes_ids)
    lpN = logp_first_token_set(no_ids)
    return lpY - lpN

# ----- Manual generation with steering (u_hat_l2) + optional on-manifold PCA -----
def project_on_local_pca(u_vec: torch.Tensor, H_window: torch.Tensor, k: int) -> torch.Tensor:
    if H_window.shape[0] < 2 or k <= 0:
        return u_vec
    X = H_window - H_window.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    V = Vh.T
    k = min(k, V.shape[1])
    basis = V[:, :k]
    proj = basis @ (basis.T @ u_vec)
    return proj

def generate_with_steering(model, tok, enc, u_hat_l2, alpha_eff: float, max_new: int,
                           steer_steps: int, device, sampling: dict,
                           use_pca=False, pca_k=32, pca_win=64):
    """Greedy/sampled generation with an additive edit on the residual stream at each generated step."""
    input_ids = enc["input_ids"].to(device)
    prompt_len = input_ids.shape[1]
    seq = input_ids.clone()
    do_sample = bool(sampling.get("do_sample", False))
    temperature = float(sampling.get("temperature", 1.0)) if "temperature" in sampling else 1.0
    top_p = sampling.get("top_p", None)

    total_logp = 0.0
    gen_hidden_cache = []

    for _ in range(max_new):
        with torch.no_grad():
            out = model(input_ids=seq, attention_mask=torch.ones_like(seq, device=device),
                        output_hidden_states=True, use_cache=False)
        H = out.hidden_states[-1][0]
        h_last = H[-1]
        gen_hidden_cache.append(h_last.detach().float().cpu())

        gen_t = seq.shape[1] - prompt_len
        apply = (steer_steps <= 0) or (gen_t < steer_steps)
        h_eff = h_last
        if apply:
            step_u = u_hat_l2.to(h_last.device, dtype=h_last.dtype)
            if use_pca:
                start = max(0, len(gen_hidden_cache)-pca_win)
                H_win = torch.stack(gen_hidden_cache[start:], dim=0).to(h_last.device, dtype=h_last.dtype)
                step_u = project_on_local_pca(step_u, H_win, k=pca_k)
            h_eff = h_last + alpha_eff * step_u

        lm = model.get_output_embeddings()
        W = lm.weight.detach().to(h_eff.device, dtype=h_eff.dtype)
        b = getattr(lm, "bias", None)
        if b is not None: b = b.detach().to(h_eff.device, dtype=h_eff.dtype)

        logits = torch.matmul(h_eff.unsqueeze(0), W.t())
        if b is not None: logits = logits + b

        if do_sample:
            probs = torch.softmax(logits / max(1e-8, temperature), dim=-1)
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                k = int((cum > top_p).float().argmax(dim=-1).item()) + 1
                k = max(2, k)
                keep_idx = sorted_idx[:, :k]
                keep_probs = sorted_probs[:, :k]
                keep_probs = keep_probs / keep_probs.sum(dim=-1, keepdim=True)
                next_id = keep_idx[0, torch.multinomial(keep_probs[0], 1)]
                next_lp = torch.log(keep_probs[0, (keep_idx[0]==next_id).nonzero(as_tuple=True)[0][0]] + 1e-38)
            else:
                next_id = torch.multinomial(probs[0], 1)
                next_lp = torch.log(probs[0, next_id] + 1e-38)
        else:
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.argmax(logits, dim=-1)
            next_lp = torch.log(probs[0, next_id] + 1e-38)

        total_logp += float(next_lp)
        seq = torch.cat([seq, next_id.view(1,1).to(device)], dim=-1)
        if tok.eos_token_id is not None and next_id.item() == tok.eos_token_id:
            break

    text = tok.decode(seq[0, prompt_len:], skip_special_tokens=True)
    if gen_hidden_cache:
        emb_mean = torch.stack(gen_hidden_cache, dim=0).mean(dim=0).numpy()
    else:
        emb_mean = np.zeros(get_output_embeddings(model).shape[1], dtype=np.float32)

    steps = max(1, (seq.shape[1] - prompt_len))
    nll = -total_logp
    ppl = math.exp(max(-100.0, min(100.0, nll / steps)))
    return seq, text, ppl, emb_mean

# =========================
# Decision extraction
# =========================
def extract_yes_no(text: str) -> str:
    """Extract a final YES/NO decision from decoded text (robust heuristics)."""
    if not text or not isinstance(text, str): return "UNKNOWN"
    lines = [ln.strip() for ln in re.split(r'[\r\n]+', text) if ln.strip()]
    tail = lines[-10:][::-1]
    tagged = list(re.finditer(r'(?is)<\s*final\s*>\s*(yes|no)\s*<\s*/\s*final\s*>', text))
    if tagged:
        return tagged[-1].group(1).upper()
    end_pat = re.compile(r'(?i)\b(yes|no)\b[\s\.\!\?"]*$')
    for ln in tail:
        m = end_pat.search(ln)
        if m: return m.group(1).upper()
    all_matches = list(re.finditer(r'(?i)\b(yes|no)\b', text))
    if all_matches: return all_matches[-1].group(0).upper()
    return "UNKNOWN"

# =========================
# Simple lexical probes
# =========================
LEX_POS = {"good","benefit","help","save","right","ethical","moral","prefer"}
LEX_NEG = {"bad","harm","hurt","kill","wrong","unethical","immoral","risk"}
LEX_TOX = {"stupid","idiot","hate","kill","die","trash","dumb"}

def probe_features(text: str) -> dict:
    t = re.sub(r"[^a-zA-Z ]+"," ", text).lower().split()
    pos = sum(1 for w in t if w in LEX_POS)
    neg = sum(1 for w in t if w in LEX_NEG)
    tox = sum(1 for w in t if w in LEX_TOX)
    exclam = text.count("!")
    return {"lex_pos":pos, "lex_neg":neg, "lex_tox":tox, "exclam":exclam}

# =========================
# Entry helpers
# =========================
def forward_hidden(model, input_ids, attention_mask):
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True, use_cache=False)
    return out.hidden_states[-1][0]

# =========================
# Main run for a single metric
# =========================
def run_one_metric(metric_name: str, args):
    out_root = Path(f"steer_runs_{metric_name}")
    (out_root / "plots").mkdir(parents=True, exist_ok=True)
    (out_root / "responses").mkdir(parents=True, exist_ok=True)
    (out_root / "timeseries").mkdir(parents=True, exist_ok=True)

    all_rows, flips_rows, auc_rows = [], [], []
    drift_rows = []

    measure_sampling = SAMPLING_BEHAV if USE_SAMPLING_FOR_MEASURE else SAMPLING_MEASURE
    alphas_fwd = ALPHAS[:]
    alphas_bwd = ALPHAS[::-1] if args.hysteresis else []

    # Load prompts from files (or fallback)
    imp_list = read_prompt_file(args.impersonal_file) if args.impersonal_file else []
    per_list = read_prompt_file(args.personal_file) if args.personal_file else []
    if not imp_list: imp_list = [PROMPTS["impersonal"]]
    if not per_list: per_list = [PROMPTS["personal"]]

    for model_name, repo in HF_MODELS.items():
        if isinstance(repo, str) and repo.strip().startswith("<HF_MODEL_ID"):
            raise ValueError(
                f"Model id for role '{model_name}' is a placeholder. "
                f"Please set HF_MODELS['{model_name}'] to a real HF repo id."
            )
        tok, model = setup_model(repo)
        device = next(model.parameters()).device
        max_new = MAX_NEW.get(model_name, 512)

        # Metric
        WH_CORPUS = [
            "Summarize the following news article.",
            "Translate this sentence into French.",
            "Write a short email to decline a meeting politely.",
            "Explain the difference between precision and recall.",
            "Give me three creative slogans for a coffee brand."
        ]

        if metric_name == "cip":
            M = compute_metric_cip_from_W(model, max_vocab=CIP_MAX_VOCAB, ridge=CIP_RIDGE)
        elif metric_name == "wth":
            M = compute_whitening_from_hidden(model, tok, WH_CORPUS, ridge=CIP_RIDGE)
        elif metric_name == "wtw":
            M = compute_metric_WtW(model, invert=True, ridge=CIP_RIDGE)
        elif metric_name == "fisher":
            M = compute_metric_fisher_stub(model)
        else:
            raise ValueError("Invalid metric")

        # Yes/No sets
        yes_ids, no_ids = build_token_sets(tok)

        # 1) Prompt states for the direction (mean-pooled over multiple imp/pers prompts)
        def last_hidden_of_prompt(prompt: str):
            enc, _ = apply_chat(tok, SYSTEM_MSG, prompt)
            H_dev = forward_hidden(model, enc["input_ids"].to(device), enc["attention_mask"].to(device))
            return H_dev[-1].detach().float().cpu(), enc

        H_imp_list, enc_imp_list = [], []
        for p in imp_list:
            h, enc = last_hidden_of_prompt(p)
            H_imp_list.append(h); enc_imp_list.append(enc)

        H_per_list, enc_per_list = [], []
        for p in per_list:
            h, enc = last_hidden_of_prompt(p)
            H_per_list.append(h); enc_per_list.append(enc)

        H_imp_mean = torch.stack(H_imp_list, dim=0).mean(dim=0)
        H_per_mean = torch.stack(H_per_list, dim=0).mean(dim=0)

        # u_hat_cip and u_l2 (pooled)
        u_hat_cip, u_raw = normalize_dir_cip(M, H_imp_mean, H_per_mean)
        u_hat_l2_cpu = l2_normalize(u_raw)

        # 2) Orient the sign with one α=0 generation sweep (mean over prompts)
        sampling_for_gen = SAMPLING_BEHAV if USE_SAMPLING_FOR_MEASURE else SAMPLING_MEASURE
        u_hat_l2 = u_hat_l2_cpu.to(device, dtype=H_imp_mean.dtype)

        def _mean_proj_seq(seq, prompt_len: int):
            H = forward_hidden(model, seq.to(device), torch.ones_like(seq, device=device)).detach().float().cpu()
            return float(torch.mv(H[prompt_len:], M @ u_hat_cip).mean()) if H.shape[0] else 0.0

        m_imp_vals, m_per_vals = [], []
        baseline_emb = {"impersonal": None, "personal": None}

        # One pass with fixed seed to orient sign
        set_all_seeds(SEED_START)
        for enc in enc_imp_list:
            seq_imp0, txt_imp0, ppl_imp0, emb_imp0 = generate_with_steering(
                model, tok, enc, u_hat_l2, 0.0, max_new, STEER_STEPS, device, sampling_for_gen,
                use_pca=args.use_pca, pca_k=args.pca_k, pca_win=args.pca_win)
            m_imp_vals.append(_mean_proj_seq(seq_imp0, enc["input_ids"].shape[1]))
            baseline_emb["impersonal"] = emb_imp0 if baseline_emb["impersonal"] is None else \
                (baseline_emb["impersonal"] + emb_imp0) / 2.0

        for enc in enc_per_list:
            seq_per0, txt_per0, ppl_per0, emb_per0 = generate_with_steering(
                model, tok, enc, u_hat_l2, 0.0, max_new, STEER_STEPS, device, sampling_for_gen,
                use_pca=args.use_pca, pca_k=args.pca_k, pca_win=args.pca_win)
            m_per_vals.append(_mean_proj_seq(seq_per0, enc["input_ids"].shape[1]))
            baseline_emb["personal"] = emb_per0 if baseline_emb["personal"] is None else \
                (baseline_emb["personal"] + emb_per0) / 2.0

        m_imp = float(np.mean(m_imp_vals)) if m_imp_vals else 0.0
        m_per = float(np.mean(m_per_vals)) if m_per_vals else 0.0
        sign = 1.0
        if (m_imp - m_per) < 0: sign *= -1.0
        if (sign * m_imp) < 0:  sign *= -1.0
        u_hat_cip = sign * u_hat_cip
        u_hat_l2_cpu = sign * u_hat_l2_cpu
        u_hat_l2 = u_hat_l2_cpu.to(device, dtype=H_imp_mean.dtype)

        # 3) α calibration with Yes/No sets
        alpha_scale, denom = calibrate_alpha_scale(model, tok, u_hat_l2, yes_ids, no_ids)

        # Force convention: +alpha should increase P(Yes). If not, flip the direction.
        eps = 0.25  # small step in alpha units (pre-scale)
        dlp_imp_p = delta_logp_sets_with_steering(
            model, tok, enc_imp_list[0], u_hat_l2, +eps * alpha_scale, device, yes_ids, no_ids
        )
        dlp_imp_m = delta_logp_sets_with_steering(
            model, tok, enc_imp_list[0], u_hat_l2, -eps * alpha_scale, device, yes_ids, no_ids
        )
        dlp_per_p = delta_logp_sets_with_steering(
            model, tok, enc_per_list[0], u_hat_l2, +eps * alpha_scale, device, yes_ids, no_ids
        )
        dlp_per_m = delta_logp_sets_with_steering(
            model, tok, enc_per_list[0], u_hat_l2, -eps * alpha_scale, device, yes_ids, no_ids
        )
        trend = (dlp_imp_p - dlp_imp_m) + (dlp_per_p - dlp_per_m)

        if not np.isfinite(trend) or trend < 0:
            # Flip direction to ensure +alpha pulls toward YES
            u_hat_cip = -u_hat_cip
            u_hat_l2_cpu = -u_hat_l2_cpu
            u_hat_l2 = -u_hat_l2

        # Baseline Δlogp (α=0) sanity check on a reference pair
        base_dlp_imp = delta_logp_sets_with_steering(model, tok, enc_imp_list[0], u_hat_l2, 0.0, device, yes_ids, no_ids)
        base_dlp_per = delta_logp_sets_with_steering(model, tok, enc_per_list[0], u_hat_l2, 0.0, device, yes_ids, no_ids)

        # Flip counts aggregated over prompts (forward sweep only)
        flip_counts = {("impersonal", a): [0, 0] for a in ALPHAS}
        flip_counts.update({("personal", a): [0, 0] for a in ALPHAS})

        # Store per-prompt series for AUC/lastK: (cond, seed, alpha, pidx) -> np.array
        timeseries_store: Dict[Tuple[str,int,float,int], np.ndarray] = {}

        # --- main block for prompt lists (supports dict of lists or single strings)
        def run_block_for_prompts(prompt_dict, tag_base_prefix=""):
            nonlocal all_rows, flips_rows, auc_rows, drift_rows, timeseries_store

            # Normalize to dict[str, List[str]]
            norm_dict: Dict[str, List[str]] = {}
            for cond, prm in prompt_dict.items():
                if isinstance(prm, list):
                    norm_dict[cond] = prm
                else:
                    norm_dict[cond] = [prm]

            # Condition prefixes for per-prompt filenames
            def cond_prefix(cond: str) -> str:
                if cond.lower().startswith("imp"): return "I"
                if cond.lower().startswith("per"): return "P"
                return re.sub(r'[^A-Za-z0-9]+','',cond)[:1].upper() or "X"

            for seed in range(SEED_START, SEED_START + N_SEEDS):
                set_all_seeds(seed)
                for cond, plist in norm_dict.items():
                    for pidx, prompt in enumerate(plist):
                        enc, _ = apply_chat(tok, SYSTEM_MSG, prompt)

                        # Baseline decision (α=0)
                        seq0, text0, ppl0, emb0 = generate_with_steering(
                            model, tok, enc, u_hat_l2, 0.0, max_new, STEER_STEPS, device, measure_sampling,
                            use_pca=args.use_pca, pca_k=args.pca_k, pca_win=args.pca_win)
                        dec0 = extract_yes_no(text0)

                        # Reference CIP series at α=0
                        H0_dev = forward_hidden(model, seq0.to(device), torch.ones_like(seq0, device=device))
                        H0_cpu = H0_dev.detach().float().cpu()
                        prompt_len = enc["input_ids"].shape[1]
                        s0, m0, _ = cip_project_series(M, u_hat_cip, H0_cpu, prompt_len)

                        # Sweep order (with optional hysteresis)
                        sweep = alphas_fwd + alphas_bwd

                        # Per-prompt filename prefix
                        pre = f"{cond_prefix(cond)}{pidx:02d}_"
                        final_prefix = f"{tag_base_prefix}{pre}"

                        for alpha in sweep:
                            alpha_eff = alpha * alpha_scale

                            # Measured generation
                            seq, text, ppl, emb = generate_with_steering(
                                model, tok, enc, u_hat_l2, alpha_eff, max_new, STEER_STEPS, device, measure_sampling,
                                use_pca=args.use_pca, pca_k=args.pca_k, pca_win=args.pca_win)
                            dec = extract_yes_no(text)

                            # Flip counting (forward sweep only)
                            if alpha in ALPHAS:
                                flips, total = flip_counts[(cond, alpha)]
                                flips += int(dec != dec0 and dec in ("YES","NO") and dec0 in ("YES","NO"))
                                total += 1
                                flip_counts[(cond, alpha)] = [flips, total]

                            # Δlogp using Yes/No sets
                            dlp = delta_logp_sets_with_steering(model, tok, enc, u_hat_l2, alpha_eff, device, yes_ids, no_ids)

                            # CIP series with steering
                            H_dev = forward_hidden(model, seq.to(device), torch.ones_like(seq, device=device))
                            H_cpu = H_dev.detach().float().cpu()
                            s, mean_s, std_s = cip_project_series(M, u_hat_cip, H_cpu, prompt_len)

                            # Drift: cosine vs baseline (per condition)
                            base_emb = baseline_emb.get(cond, emb0)
                            if base_emb is not None and emb is not None:
                                a = np.array(base_emb); b = np.array(emb)
                                if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0:
                                    cos = float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)))
                                else:
                                    cos = float("nan")
                            else:
                                cos = float("nan")

                            # Save per-prompt series
                            ts_path = out_root / "timeseries" / f"{final_prefix}{model_name}_{cond}_seed{seed}_a{alpha:+.2f}.csv"
                            with ts_path.open("w", encoding="utf-8") as f:
                                f.write("step,proj_cip\n")
                                for i, v in enumerate(s):
                                    f.write(f"{i},{v:.6f}\n")

                            # Save per-prompt response
                            rp = out_root / "responses" / f"{final_prefix}{model_name}_{cond}_seed{seed}_a{alpha:+.2f}.txt"
                            with rp.open("w", encoding="utf-8") as f:
                                f.write(f"[model={model_name}] [cond={cond}] [seed={seed}] [pidx={pidx}] [alpha={alpha:+.2f}] [alpha_eff={alpha_eff:.4g}] [metric={metric_name}]\n")
                                f.write("=== TEXT ===\n" + text.strip() + "\n")
                                f.write(f"[DECISION={dec}] [PPL={ppl:.4f}] [COS_BASE={cos:.4f}]\n")

                            # Register rows
                            all_rows.append({
                                "metric": metric_name, "model_name": model_name, "condition": cond,
                                "seed": seed, "alpha": alpha, "alpha_eff": alpha_eff,
                                "decision": dec, "decision_base": dec0,
                                "delta_logp_yes_minus_no": dlp,
                                "proj_mean": mean_s, "proj_mean_base": m0,
                                "seq_len": int(seq.shape[1]),
                                "response_file": str(rp),
                                "timeseries_file": str(ts_path),
                                "ppl": ppl, "cosine_vs_base": cos,
                                **probe_features(text)
                            })
                            drift_rows.append({
                                "metric": metric_name, "model_name": model_name, "condition": cond,
                                "seed": seed, "alpha": alpha, "ppl": ppl, "cosine_vs_base": cos, **probe_features(text)
                            })

                            # Store series for AUC/lastK
                            if cond in ("impersonal","personal") and alpha in ALPHAS:
                                timeseries_store[(cond, seed, alpha, pidx)] = np.array(s, dtype=float)

            # Aggregate flips
            for alpha in ALPHAS:
                for cond in norm_dict.keys():
                    flips, total = flip_counts.get((cond, alpha), (0,0))
                    flips_rows.append({
                        "metric": metric_name, "model_name": model_name, "condition": cond,
                        "alpha": alpha, "flip_rate": flips / max(1,total), "N": total
                    })

            # AUC/last-K by alpha (mean over seeds and matched prompt pairs)
            if "impersonal" in norm_dict and "personal" in norm_dict:
                n_pairs = min(len(norm_dict["impersonal"]), len(norm_dict["personal"]))
                for alpha in ALPHAS:
                    auc_vals, lastk_vals = [], []
                    for seed in range(SEED_START, SEED_START + N_SEEDS):
                        for pidx in range(n_pairs):
                            key_imp = ("impersonal", seed, alpha, pidx)
                            key_per = ("personal",   seed, alpha, pidx)
                            if key_imp in timeseries_store and key_per in timeseries_store:
                                imp = timeseries_store[key_imp]
                                per = timeseries_store[key_per]
                                auc_vals.append(auc_gap(imp, per))
                                lastk_vals.append(lastk_gap(imp, per, k=128))
                    if auc_vals:
                        auc_rows.append({
                            "metric": metric_name, "model_name": model_name, "alpha": alpha,
                            "auc_gap_mean": float(np.mean(auc_vals)),
                            "auc_gap_std": float(np.std(auc_vals, ddof=1)) if len(auc_vals)>1 else 0.0,
                            "lastk_gap_mean": float(np.mean(lastk_vals)),
                            "lastk_gap_std": float(np.std(lastk_vals, ddof=1)) if len(lastk_vals)>1 else 0.0,
                            "N": len(auc_vals),
                        })

        # Run main block (prompt lists) and optional controls
        run_block_for_prompts({"impersonal": imp_list, "personal": per_list}, tag_base_prefix="")
        if args.run_ctrl:
            run_block_for_prompts(PROMPTS_CTRL, tag_base_prefix="CTRL_")

        # ---- per-model plots ----
        df_all_m = pd.DataFrame(all_rows)
        df_flips_m = pd.DataFrame([r for r in flips_rows if r["model_name"]==model_name])
        df_auc_m  = pd.DataFrame([r for r in auc_rows  if r["model_name"]==model_name])
        df_all_m = df_all_m[df_all_m["model_name"]==model_name]

        # 1) Flip rate vs alpha
        plt.figure(figsize=(6,4))
        for cond in df_flips_m["condition"].unique():
            sub = df_flips_m[df_flips_m.condition==cond].sort_values("alpha")
            if sub.empty: continue
            plt.plot(sub["alpha"], sub["flip_rate"], marker="o", label=cond)
        plt.axhline(0.0, ls="--", lw=1)
        plt.xlabel("alpha"); plt.ylabel("Flip rate")
        plt.title(f"Flip rate vs alpha · {model_name} · {metric_name}")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_root / "plots" / f"flip_rate_{model_name}.png", dpi=160); plt.close()

        # 2) Δlogp vs alpha (mean ± sd)
        plt.figure(figsize=(6,4))
        for cond in df_all_m["condition"].unique():
            grp = df_all_m[df_all_m.condition==cond].groupby("alpha")["delta_logp_yes_minus_no"]
            if grp.ngroups==0: continue
            xs = np.array(sorted(grp.mean().index))
            mus = np.array([grp.mean()[a] for a in xs])
            sig = np.array([grp.std(ddof=1)[a] if not math.isnan(grp.std(ddof=1)[a]) else 0.0 for a in xs])
            plt.errorbar(xs, mus, yerr=sig, marker="o", label=cond)
        plt.axhline(0.0, ls="--", lw=1)
        plt.xlabel("alpha"); plt.ylabel("Δlogp(Yes−No) [sets-LSE]")
        plt.title(f"Δlogp vs alpha · {model_name} · {metric_name}")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_root / "plots" / f"delta_logp_{model_name}.png", dpi=160); plt.close()

        # 3) AUC/last-K vs alpha
        if not df_auc_m.empty:
            fig, ax = plt.subplots(1,2, figsize=(10,4))
            ax[0].errorbar(df_auc_m["alpha"], df_auc_m["auc_gap_mean"], yerr=df_auc_m["auc_gap_std"], marker="o")
            ax[0].axhline(0.0, ls="--", lw=1); ax[0].set_xlabel("alpha"); ax[0].set_ylabel("AUC(gap)")
            ax[0].set_title(f"AUC(gap) · {model_name} · {metric_name}")
            ax[1].errorbar(df_auc_m["alpha"], df_auc_m["lastk_gap_mean"], yerr=df_auc_m["lastk_gap_std"], marker="o")
            ax[1].axhline(0.0, ls="--", lw=1); ax[1].set_xlabel("alpha"); ax[1].set_ylabel("last-128(gap)")
            ax[1].set_title(f"last-128(gap) · {model_name} · {metric_name}")
            fig.tight_layout()
            fig.savefig(out_root / "plots" / f"auc_lastk_{model_name}.png", dpi=160)
            plt.close(fig)

        # 4) Calibration curve
        plt.figure(figsize=(6,4))
        subc = df_all_m.groupby("alpha")["delta_logp_yes_minus_no"]
        xs = sorted(subc.mean().index)
        mus = [subc.mean()[a] for a in xs]
        sig = [subc.std(ddof=1)[a] if not math.isnan(subc.std(ddof=1)[a]) else 0.0 for a in xs]
        plt.errorbar(xs, mus, yerr=sig, marker="o")
        plt.axhline(0.0, ls="--", lw=1)
        plt.xlabel("alpha"); plt.ylabel("Δlogp realized (sets-LSE)")
        plt.title(f"Calibration curve · {model_name} · {metric_name}")
        plt.tight_layout()
        plt.savefig(out_root / "plots" / f"calibration_{model_name}.png", dpi=160); plt.close()

        # 5) Hysteresis (optional)
        if args.hysteresis:
            plt.figure(figsize=(6,4))
            ida = df_all_m[df_all_m["alpha"].isin(ALPHAS)].groupby("alpha")["delta_logp_yes_minus_no"].mean()
            volta = df_all_m[~df_all_m["alpha"].isin(ALPHAS)].groupby("alpha")["delta_logp_yes_minus_no"].mean()
            if not ida.empty:   plt.plot(ida.index, ida.values, marker="o", label="up-sweep")
            if not volta.empty: plt.plot(volta.index, volta.values, marker="x", label="down-sweep")
            plt.axhline(0.0, ls="--", lw=1)
            plt.xlabel("alpha"); plt.ylabel("Δlogp")
            plt.title(f"Hysteresis · {model_name} · {metric_name}")
            plt.legend(); plt.tight_layout()
            plt.savefig(out_root / "plots" / f"hysteresis_{model_name}.png", dpi=160); plt.close()

        try: del model
        except Exception: pass
        free_mem()

    # Save CSVs
    df_all = pd.DataFrame(all_rows)
    df_flips = pd.DataFrame(flips_rows)
    df_auc = pd.DataFrame(auc_rows)
    df_drift = pd.DataFrame(drift_rows)

    (out_root / "summary_per_alpha.csv").write_text(df_all.to_csv(index=False), encoding="utf-8")
    (out_root / "flips_vs_alpha.csv").write_text(df_flips.to_csv(index=False), encoding="utf-8")
    (out_root / "auc_lastk_vs_alpha.csv").write_text(df_auc.to_csv(index=False), encoding="utf-8")
    (out_root / "drift_checks.csv").write_text(df_drift.to_csv(index=False), encoding="utf-8")

    # Console summaries
    def ci(x):
        x = np.array(x, dtype=float)
        if len(x)==0: return (float("nan"), float("nan"), float("nan"))
        m = float(np.mean(x))
        se = float(np.std(x, ddof=1)) / max(1, math.sqrt(len(x)))
        return (m, m-1.96*se, m+1.96*se)

    print("\n=== MODEL SUMMARIES ===")
    for m in df_all["model_name"].unique():
        sub = df_all[df_all.model_name==m]
        print(f"\n[{m}] Δlogp(Yes−No) vs α (mean ± ≈95% CI) · metric={metric_name}:")
        for a in sorted(sub["alpha"].unique()):
            s_imp = sub[(sub.alpha==a) & (sub.condition=="impersonal")]["delta_logp_yes_minus_no"].values
            s_per = sub[(sub.alpha==a) & (sub.condition=="personal")]["delta_logp_yes_minus_no"].values
            mi = ci(s_imp); mp = ci(s_per)
            print(f"  α={a:+.2f} · imp={mi[0]:+.3f} [{mi[1]:+.3f},{mi[2]:+.3f}] · per={mp[0]:+.3f} [{mp[1]:+.3f},{mp[2]:+.3f}]")

    print(f"\nFiles saved in: {out_root.resolve()}")
    for fn in ["summary_per_alpha.csv","flips_vs_alpha.csv","auc_lastk_vs_alpha.csv","drift_checks.csv"]:
        p = out_root / fn
        if p.exists(): print(" -", p)
    print("Figures in:", (out_root/'plots').resolve())

# =========================
# Cross-metric sensitivity (Spearman)
# =========================
def rank_corr_spearman(x, y):
    sx = pd.Series(x).rank(method="average")
    sy = pd.Series(y).rank(method="average")
    return float(sx.corr(sy, method="pearson"))

def build_sensitivity_report(metrics: List[str]):
    rows = []
    dfs = {}
    for mt in metrics:
        root = Path(f"steer_runs_{mt}")
        if not (root/"summary_per_alpha.csv").exists(): continue
        df_all = pd.read_csv(root/"summary_per_alpha.csv")
        df_flip = pd.read_csv(root/"flips_vs_alpha.csv")
        df_auc  = pd.read_csv(root/"auc_lastk_vs_alpha.csv")
        dfs[mt] = (df_all, df_flip, df_auc)

    mts = list(dfs.keys())
    out_dir = Path("steer_runs_sensitivity"); out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "summary_sensitivity.csv"

    for i in range(len(mts)):
        for j in range(i+1, len(mts)):
            m1, m2 = mts[i], mts[j]
            df1_all, df1_flip, df1_auc = dfs[m1]
            df2_all, df2_flip, df2_auc = dfs[m2]

            def agg_dlp(df):
                return df.groupby(["model_name","condition","alpha"])["delta_logp_yes_minus_no"].mean().reset_index()
            def agg_flip(df):
                return df.groupby(["model_name","condition","alpha"])["flip_rate"].mean().reset_index()
            def agg_auc(df):
                return df.groupby(["model_name","alpha"])["auc_gap_mean"].mean().reset_index()

            d1 = agg_dlp(df1_all); d2 = agg_dlp(df2_all)
            f1 = agg_flip(df1_flip); f2 = agg_flip(df2_flip)
            a1 = agg_auc(df1_auc);  a2 = agg_auc(df2_auc)

            d = pd.merge(d1, d2, on=["model_name","condition","alpha"], suffixes=("_1","_2"))
            f = pd.merge(f1, f2, on=["model_name","condition","alpha"], suffixes=("_1","_2"))
            a = pd.merge(a1, a2, on=["model_name","alpha"], suffixes=("_1","_2"))

            corr_dlp = rank_corr_spearman(d["delta_logp_yes_minus_no_1"], d["delta_logp_yes_minus_no_2"]) if not d.empty else float("nan")
            corr_flip = rank_corr_spearman(f["flip_rate_1"], f["flip_rate_2"]) if not f.empty else float("nan")
            corr_auc = rank_corr_spearman(a["auc_gap_mean_1"], a["auc_gap_mean_2"]) if not a.empty else float("nan")

            rows.append({"metric_1":m1,"metric_2":m2,"spearman_dlp":corr_dlp,"spearman_flip":corr_flip,"spearman_auc":corr_auc})

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Sensitivity saved to:", out_path.resolve())

# =========================
# Entry
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default="cip",
                        choices=["cip","wth","wtw","fisher","all"])
    parser.add_argument("--run_ctrl", action="store_true", help="Run control/contrast prompt blocks")
    parser.add_argument("--hysteresis", action="store_true", help="Sweep alpha forward and backward")
    parser.add_argument("--use_pca", action="store_true", help="Enable on-manifold PCA projection for the edit")
    parser.add_argument("--pca_k", type=int, default=PCA_K, help="PCA rank (if --use_pca)")
    parser.add_argument("--pca_win", type=int, default=PCA_WIN, help="Sliding window length for PCA (if --use_pca)")
    parser.add_argument("--impersonal_file", type=str, default="impersonal.txt", help="Path to impersonal prompts file")
    parser.add_argument("--personal_file", type=str, default="personal.txt", help="Path to personal prompts file")
    args = parser.parse_args()

    if args.metric == "all":
        metrics = ["cip","wth","wtw","fisher"]
        for m in metrics:
            run_one_metric(m, args)
        build_sensitivity_report(metrics)
    else:
        run_one_metric(args.metric, args)

if __name__ == "__main__":
    main()
