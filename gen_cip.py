# gen_cip.py
# -*- coding: utf-8 -*-
"""
Generation + Geometry (CIP / whitening / W^T W) for three roles (baseline, reasoning, COT),
saving tokens and projection time series for downstream analysis.

Roles (logical profiles):
 - "baseline":  prompts from impersonal.txt / personal.txt
 - "COT":      prompts from impersonal2.txt / personal2.txt (same model as baseline by default)
 - "reasoning": may point to a model configured to produce brief rationales

This script:
 - runs generation for each role × condition (impersonal/personal) × pair × seed
 - saves tokens, responses, metadata
 - recomputes full hidden states and projects them on an oriented concept direction
 - dumps per-seed summaries and aggregated CSVs
 - computes cross-metric Spearman sensitivity for Δproj
"""

import os, gc, re, json, math, argparse, datetime as dt
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
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
# Optional per-role specific revisions (git commit / tag); use None for default.
HF_REVISIONS = {"reasoning": None, "baseline": None, "COT": None}

# System message used for chat-style templates (kept simple on purpose).
SYSTEM_MSG = "You are a helpful assistant"

# Default prompt files (one prompt per line; impersonal and personal matched by index).
IMPERSONAL_TXT  = "impersonal.txt"
PERSONAL_TXT    = "personal.txt"
IMPERSONAL2_TXT = "impersonal2.txt"
PERSONAL2_TXT   = "personal2.txt"

# Generation caps per role and a minimal retry plan to obtain a literal YES/NO when possible.
GEN_TOKENS_INIT = {"reasoning": 3072, "baseline": 3072, "COT": 3072}
RETRY_PLAN      = {"reasoning": [3072], "baseline": [3072, 3072], "COT": [3072, 3072]}

# Sampling parameters (kept constant across conditions).
SAMPLING    = {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": None, "repetition_penalty": None}
GEN_KW_BASE = dict(**SAMPLING)

# Seeds
N_SEEDS    = 5
SEED_START = 1

# Geometry (CIP) configuration.
CIP_MAX_VOCAB = None
CIP_RIDGE     = 1e-4

# Outputs (paths unchanged intentionally to preserve compatibility with existing tooling).
OUTDIR = Path("runs_trolleycorrect")
(OUTDIR / "tokens").mkdir(parents=True, exist_ok=True)
(OUTDIR / "meta").mkdir(parents=True, exist_ok=True)
(OUTDIR / "timeseries").mkdir(parents=True, exist_ok=True)
(OUTDIR / "plots").mkdir(parents=True, exist_ok=True)
(OUTDIR / "responses").mkdir(parents=True, exist_ok=True)
(OUTDIR / "sensitivity").mkdir(parents=True, exist_ok=True)
OFFLOAD_DIR = OUTDIR / "offload"; OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def apply_chat(tokenizer, system: str, user: str):
    """
    Build a chat-style prompt. If the tokenizer supports chat templates, use them;
    otherwise fall back to a simple role-tagged plain format.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            add_generation_prompt=True, tokenize=False
        )
    else:
        text = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
    enc = tokenizer(text, return_tensors="pt")
    return enc, text

def setup_model(model_path: str):
    """Load tokenizer and model with sensible dtype/device defaults."""
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    try:
        # Ensure pad token for generation config if missing.
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            model.generation_config.pad_token_id = tok.eos_token_id
    except Exception:
        pass
    model.config.return_dict_in_generate = True
    model.eval()
    return tok, model

def get_output_embeddings(model) -> torch.Tensor:
    """Return lm_head weights (V×d) as float CPU tensor; required for CIP/W^T W."""
    lm = model.get_output_embeddings()
    if lm is None or not hasattr(lm, "weight"):
        raise RuntimeError("Model does not expose lm_head.weight; required for geometry metrics.")
    return lm.weight.detach().float().cpu()

# ===== Geometry metrics =====
def compute_cov_precision(X: torch.Tensor, ridge=1e-4):
    """Compute (Cov + λI)^(-1) via Cholesky, with ridge scaled to mean diagonal."""
    Cov = (X.T @ X) / max(1, (X.shape[0]-1))
    lam = ridge * float(torch.mean(torch.diag(Cov)))
    Cov = Cov + lam * torch.eye(Cov.shape[0], dtype=Cov.dtype)
    L = torch.linalg.cholesky(Cov)
    I = torch.eye(Cov.shape[0], dtype=Cov.dtype)
    return torch.cholesky_solve(I, L)

def metric_cip_W(model, ridge=CIP_RIDGE):
    """CIP with row-covariance of W (centered)."""
    W = get_output_embeddings(model)
    X = W - W.mean(dim=0, keepdim=True)
    return compute_cov_precision(X, ridge)

def metric_whitening_from_hidden(model, tok, corpus_prompts: List[str], ridge=CIP_RIDGE):
    """Whitening metric from in-model hidden states on a small generic corpus."""
    device = next(model.parameters()).device
    Hs = []
    with torch.no_grad():
        for p in corpus_prompts:
            enc, _ = apply_chat(tok, SYSTEM_MSG, p)
            out = model(input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device),
                        output_hidden_states=True, use_cache=False)
            h = out.hidden_states[-1][0, -1, :].detach().float().cpu()
            Hs.append(h.unsqueeze(0))
    H = torch.cat(Hs, dim=0) if Hs else torch.zeros(8, get_output_embeddings(model).shape[1])
    X = H - H.mean(dim=0, keepdim=True)
    return compute_cov_precision(X, ridge)

def metric_WtW(model, invert=True, ridge=CIP_RIDGE):
    """Regularized (W^T W)^(-1) Gram form (or raw Gram if invert=False)."""
    W = get_output_embeddings(model)  # V×d
    G = (W.T @ W) / max(1, (W.shape[0]-1))
    if invert:
        lam = ridge * float(torch.mean(torch.diag(G)))
        G = G + lam * torch.eye(G.shape[0], dtype=G.dtype)
        L = torch.linalg.cholesky(G)
        I = torch.eye(G.shape[0], dtype=G.dtype)
        return torch.cholesky_solve(I, L)
    else:
        return G

def forward_last_prompt_state(model, input_ids, attention_mask):
    """Return last hidden state at the end of the prompt segment."""
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True, use_cache=False)
    return out.hidden_states[-1][0, input_ids.shape[1]-1, :].detach().float().cpu()

def recompute_full_hidden_states(model, seq_ids, attn_mask):
    """Return full last-layer hidden states for the whole sequence."""
    with torch.no_grad():
        out = model(input_ids=seq_ids, attention_mask=attn_mask,
                    output_hidden_states=True, use_cache=False)
    return out.hidden_states[-1][0].detach().float().cpu()

def cip_projection_series(M, concept_dir, H, prompt_len: int):
    """
    Project generated segment H[prompt_len:] on oriented concept_dir under metric M.
    Returns the per-step series, mean, and std.
    """
    num = concept_dir
    denom = torch.sqrt(torch.clamp(num @ (M @ num), min=1e-12))
    u = num / denom
    gen_H = H[prompt_len:]
    s = torch.mv(gen_H, M @ u)
    s_np = s.numpy()
    return s_np, float(s_np.mean()) if len(s_np)>0 else 0.0, float(s_np.std()) if len(s_np)>0 else 0.0

# ===== Behavior / extraction =====
def extract_yes_no(text: str) -> str:
    """
    Extract a final YES/NO decision from decoded text.
    Strategy:
      1) look for <final>yes|no</final> tag near the end
      2) scan last 10 non-empty lines for a trailing yes/no
      3) fallback: last yes/no anywhere
    """
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

def write_response_file(model_name, condition, seed, system_text, prompt_text, attempts, chosen_idx, outdir, pair_id):
    """Save prompts, attempts, and chosen decision for traceability."""
    path = outdir / "responses" / f"{model_name}_{condition}_p{pair_id}_seed{seed}.txt"
    ts = dt.datetime.now().isoformat(timespec="seconds")
    with path.open("w", encoding="utf-8") as f:
        f.write(f"[{ts}] model={model_name} condition={condition} pair={pair_id} seed={seed}\n")
        f.write("=== SYSTEM ===\n" + system_text + "\n")
        f.write("=== USER ===\n" + prompt_text + "\n")
        for i, (n_tok, text, decision) in enumerate(attempts, 1):
            mark = " <— CHOSEN" if (i-1) == chosen_idx else ""
            f.write(f"\n--- Attempt {i} (max_new_tokens={n_tok}){mark}\n")
            f.write(text.strip() + "\n")
            f.write(f"[DECISION={decision}]\n")
    return str(path)

def generate_with_retries(model, tok, input_ids, attention_mask, n0, retry_plan, seed):
    """
    Generate with a small set of max_new_tokens attempts to elicit a literal YES/NO.
    Returns chosen sequence/text/decision/used_n and the metadata of all attempts.
    """
    device = next(model.parameters()).device
    tries = [n0] + list(retry_plan)
    attempts_meta = []
    chosen = None; chosen_idx = 0

    def _one_try(n, gen_seed):
        set_all_seeds(gen_seed)
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=n,
                return_dict_in_generate=True,
                **GEN_KW_BASE,
            )
        seq = gen.sequences
        prompt_len = input_ids.shape[1]
        text = tok.decode(seq[0, prompt_len:], skip_special_tokens=True)
        decision = extract_yes_no(text)
        return seq, text, decision, n

    for i, n in enumerate(tries):
        seq, text, decision, used_n = _one_try(n, seed + i*100003)
        attempts_meta.append((used_n, text, decision))
        if decision != "UNKNOWN" and chosen is None:
            chosen = (seq, text, decision, used_n); chosen_idx = i; break
    if chosen is None:
        chosen = (seq, text, decision, used_n); chosen_idx = len(tries)-1
    return chosen[0], chosen[1], chosen[2], chosen[3], attempts_meta, chosen_idx

def ensure_local_models(hf_map, rev_map):
    """
    Ensure all role→HF model ids are locally present (snapshot_download).
    Raises a clear error if a model id appears to be an unedited placeholder.
    """
    from huggingface_hub import snapshot_download
    resolved = {}
    base_dir = Path("hf_models"); base_dir.mkdir(exist_ok=True)
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    for role, repo_id in hf_map.items():
        if isinstance(repo_id, str) and repo_id.strip().startswith("<HF_MODEL_ID"):
            raise ValueError(
                f"Model id for role '{role}' is a placeholder. "
                f"Please set HF_MODELS['{role}'] to a real HF repo id."
            )
        safe = repo_id.replace("/", "__")
        tgt = base_dir / safe
        if not (tgt.exists() and any(tgt.iterdir())):
            print(f"[download] {repo_id} → {tgt}")
            kw = dict(repo_id=repo_id, local_dir=str(tgt), local_dir_use_symlinks=False)
            if token: kw["token"] = token
            rev = rev_map.get(role); kw["revision"] = rev if rev else None
            snapshot_download(**{k: v for k, v in kw.items() if v is not None})
        resolved[role] = str(tgt)
    return resolved

def load_lines(path: str) -> List[str]:
    """Read non-empty lines from a UTF-8 text file; one prompt per line."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    lines = []
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                lines.append(ln)
    return lines

def condition_to_tag(c: str) -> str:
    return c.replace(" ", "_")

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default="cip", choices=["cip", "wth", "wtw", "all"],
                        help="Geometry metric: CIP (W covariance), whitening, or W^T W; 'all' runs all three.")
    parser.add_argument("--impersonal_txt",  type=str, default=IMPERSONAL_TXT,
                        help="Path to impersonal.txt (one prompt per line).")
    parser.add_argument("--personal_txt",    type=str, default=PERSONAL_TXT,
                        help="Path to personal.txt (one prompt per line).")
    parser.add_argument("--impersonal2_txt", type=str, default=IMPERSONAL2_TXT,
                        help="Path to impersonal2.txt used for the 'COT' role.")
    parser.add_argument("--personal2_txt",   type=str, default=PERSONAL2_TXT,
                        help="Path to personal2.txt used for the 'COT' role.")
    args = parser.parse_args()

    # Load prompt sets (two matched sets).
    imp_list_1 = load_lines(args.impersonal_txt)
    per_list_1 = load_lines(args.personal_txt)
    imp_list_2 = load_lines(args.impersonal2_txt)
    per_list_2 = load_lines(args.personal2_txt)

    if min(len(imp_list_1), len(per_list_1)) == 0:
        raise RuntimeError("Prompt lists (impersonal.txt/personal.txt) are empty.")
    if min(len(imp_list_2), len(per_list_2)) == 0:
        raise RuntimeError("Prompt lists (impersonal2.txt/personal2.txt) are empty.")

    # Map which lists each role uses:
    # reasoning → set1; baseline → set1; COT → set2
    PROMPTS: Dict[str, Tuple[List[str], List[str]]] = {
        "reasoning": (imp_list_1, per_list_1),
        "baseline":  (imp_list_1, per_list_1),
        "COT":       (imp_list_2, per_list_2),
    }
    PAIRS_COUNT = {name: min(len(PROMPTS[name][0]), len(PROMPTS[name][1])) for name in PROMPTS}
    print("[prompts] pairs per role:", {k: PAIRS_COUNT[k] for k in ["reasoning", "baseline", "COT"]})

    # Ensure local model snapshots (explicit, role-based).
    local_map = ensure_local_models(HF_MODELS, HF_REVISIONS)
    print("Local model paths:", json.dumps(local_map, indent=2))

    # Small generic corpus for whitening metric.
    WH_CORPUS = [
        "Summarize the following news article.",
        "Translate this sentence into French.",
        "Write a short email to decline a meeting politely.",
        "Explain the difference between precision and recall.",
        "Give me three creative slogans for a coffee brand."
    ]

    all_metric_delta = []  # for cross-metric sensitivity (includes pair_id)
    metrics = ["cip", "wth", "wtw"] if args.metric == "all" else [args.metric]

    for metric_name in metrics:
        results: Dict[Tuple[str, int, str, int], dict] = {}
        # key: (role_name, pair_id, condition, seed)

        # --------- GENERATION PHASE ----------
        for name, local_path in local_map.items():
            imp_list, per_list = PROMPTS[name]
            n_pairs = PAIRS_COUNT[name]
            for seed in range(SEED_START, SEED_START + N_SEEDS):
                set_all_seeds(seed)
                tok, model = setup_model(local_path)
                device = next(model.parameters()).device

                for pair_id in range(1, n_pairs + 1):
                    for cond in ["impersonal", "personal"]:
                        prompt = imp_list[pair_id - 1] if cond == "impersonal" else per_list[pair_id - 1]
                        print(f"\n=== {metric_name} · {name} · pair=p{pair_id} · {cond} · seed={seed} ===")

                        enc, _ = apply_chat(tok, SYSTEM_MSG, prompt)
                        input_ids = enc["input_ids"]; attention_mask = enc["attention_mask"]
                        prompt_len = input_ids.shape[1]

                        h_prompt = forward_last_prompt_state(model, input_ids.to(device), attention_mask.to(device))

                        n0 = GEN_TOKENS_INIT.get(name, 512)
                        seq, text, decision, used_n, attempts_meta, chosen_idx = generate_with_retries(
                            model, tok, input_ids, attention_mask, n0, RETRY_PLAN.get(name, []), seed
                        )

                        tok_path  = OUTDIR / "tokens" / f"{metric_name}_{name}_{condition_to_tag(cond)}_p{pair_id}_seed{seed}.pt"
                        meta_path = OUTDIR / "meta"   / f"{metric_name}_{name}_{condition_to_tag(cond)}_p{pair_id}_seed{seed}.json"
                        torch.save(seq.cpu(), tok_path)
                        with meta_path.open("w", encoding="utf-8") as f:
                            json.dump({
                                "metric": metric_name,
                                "model_name": name,
                                "model_path": local_path,
                                "pair_id": pair_id,
                                "condition": cond,
                                "seed": seed,
                                "prompt_len": int(prompt_len),
                                "gen_steps_used": int(used_n),
                                "prompt_text": prompt
                            }, f, ensure_ascii=False, indent=2)

                        resp_path = write_response_file(
                            model_name=name, condition=cond, seed=seed,
                            system_text=SYSTEM_MSG, prompt_text=prompt,
                            attempts=attempts_meta, chosen_idx=chosen_idx,
                            outdir=OUTDIR, pair_id=pair_id
                        )

                        attn_mask_full = torch.ones_like(seq, device=seq.device)
                        H = recompute_full_hidden_states(model, seq, attn_mask_full)

                        results[(name, pair_id, cond, seed)] = {
                            "model_name": name, "model_path": local_path,
                            "pair_id": pair_id, "condition": cond, "seed": seed,
                            "prompt_text": prompt, "prompt_len": int(prompt_len),
                            "gen_steps_used": int(used_n),
                            "generated_text": text.strip(), "decision": decision,
                            "h_prompt": h_prompt, "H_full": H,
                            "sequences_len": int(seq.shape[1]),
                            "tokens_file": str(tok_path), "meta_file": str(meta_path),
                            "response_file": str(resp_path),
                        }

                try: del model
                except Exception: pass
                free_mem()

        # --------- PROJECTION + SUMMARIES ----------
        summary_rows, delta_rows = [], []

        for name, local_path in local_map.items():
            imp_list, per_list = PROMPTS[name]
            n_pairs = PAIRS_COUNT[name]

            tok, model = setup_model(local_path)
            # Select metric
            if metric_name == "cip":
                M = metric_cip_W(model, ridge=CIP_RIDGE)
            elif metric_name == "wth":
                M = metric_whitening_from_hidden(model, tok, WH_CORPUS, ridge=CIP_RIDGE)
            elif metric_name == "wtw":
                M = metric_WtW(model, invert=True, ridge=CIP_RIDGE)
            else:
                raise ValueError("Invalid metric choice.")

            # For each pair, compute concept direction and projection series
            for pair_id in range(1, n_pairs + 1):
                seed0 = SEED_START
                h_imp = results[(name, pair_id, "impersonal", seed0)]["h_prompt"]
                h_per = results[(name, pair_id, "personal",  seed0)]["h_prompt"]
                concept_dir = (h_imp - h_per).contiguous()

                # Orient the sign so impersonal projections > personal on average
                imp_series, per_series = {}, {}
                imp_means, per_means = [], []
                for seed in range(SEED_START, SEED_START + N_SEEDS):
                    for cond in ["impersonal", "personal"]:
                        H = results[(name, pair_id, cond, seed)]["H_full"]
                        prompt_len = results[(name, pair_id, cond, seed)]["prompt_len"]
                        s_t, s_mean, _ = cip_projection_series(M, concept_dir, H, prompt_len)
                        if cond == "impersonal":
                            imp_means.append(s_mean); imp_series[seed] = s_t
                        else:
                            per_means.append(s_mean); per_series[seed] = s_t
                        # Save raw series
                        ts_path = OUTDIR / "timeseries" / f"{metric_name}_{name}_{cond}_p{pair_id}_seed{seed}.csv"
                        with ts_path.open("w", encoding="utf-8") as f:
                            f.write("step,proj\n")
                            for i, v in enumerate(s_t):
                                f.write(f"{i},{v:.6f}\n")

                mean_imp = float(np.mean(imp_means)) if imp_means else 0.0
                mean_per = float(np.mean(per_means)) if per_means else 0.0
                sign = 1.0
                if mean_imp < 0: sign *= -1.0; mean_imp, mean_per = -mean_imp, -mean_per
                if (mean_imp - mean_per) < 0: sign *= -1.0; mean_imp, mean_per = -mean_imp, -mean_per
                concept_dir = concept_dir * sign

                # Recompute final summaries per seed/condition
                for seed in range(SEED_START, SEED_START + N_SEEDS):
                    row_by_cond = {}
                    for cond in ["impersonal", "personal"]:
                        H = results[(name, pair_id, cond, seed)]["H_full"]
                        prompt_len = results[(name, pair_id, cond, seed)]["prompt_len"]
                        s_t, s_mean, s_std = cip_projection_series(M, concept_dir, H, prompt_len)
                        row = {
                            "metric": metric_name,
                            "model_name": name,
                            "pair_id": pair_id,
                            "condition": cond,
                            "seed": seed,
                            "prompt_len": prompt_len,
                            "gen_steps": results[(name, pair_id, cond, seed)]["gen_steps_used"],
                            "decision": results[(name, pair_id, cond, seed)]["decision"],
                            "proj_mean": s_mean, "proj_std": s_std,
                            "seq_len_total": results[(name, pair_id, cond, seed)]["sequences_len"],
                            "response_file": results[(name, pair_id, cond, seed)]["response_file"],
                            "tokens_file": results[(name, pair_id, cond, seed)]["tokens_file"],
                            "meta_file": results[(name, pair_id, cond, seed)]["meta_file"],
                            "prompt_text": results[(name, pair_id, cond, seed)]["prompt_text"],
                        }
                        summary_rows.append(row); row_by_cond[cond] = row
                    d = row_by_cond["impersonal"]["proj_mean"] - row_by_cond["personal"]["proj_mean"]
                    delta_rows.append({
                        "metric": metric_name, "model_name": name,
                        "pair_id": pair_id, "seed": seed, "delta_proj": d
                    })
                    all_metric_delta.append({
                        "metric": metric_name, "model_name": name,
                        "pair_id": pair_id, "seed": seed, "delta_proj": d
                    })

                # Overlay plots (mean ± std across seeds) per condition
                def overlay_plot(model_name, seed_series, title_cond, fname, pair_id):
                    max_len = max((len(v) for v in seed_series.values()), default=0)
                    arr = np.full((len(seed_series), max_len), np.nan, dtype=float)
                    for r, (_, series) in enumerate(sorted(seed_series.items())):
                        arr[r, :len(series)] = series
                    mean = np.nanmean(arr, axis=0); std = np.nanstd(arr, axis=0)
                    x = np.arange(len(mean))
                    plt.figure(figsize=(9, 3))
                    plt.plot(x, mean, label=title_cond)
                    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
                    plt.xlabel("Generated step"); plt.ylabel("Projection (oriented)")
                    plt.title(f"{model_name} · p{pair_id} · {title_cond} ({metric_name})")
                    plt.legend(); plt.tight_layout()
                    plt.savefig(OUTDIR / "plots" / f"overlay_{metric_name}_{model_name}_{fname}_p{pair_id}.png", dpi=160); plt.close()

                overlay_plot(name, imp_series, "Impersonal", "impersonal", pair_id)
                overlay_plot(name, per_series,  "Personal",   "personal",  pair_id)

            try: del model
            except Exception: pass
            free_mem()

        # ---- Save CSVs per metric ----
        df_sum = pd.DataFrame(summary_rows)
        df_delta = pd.DataFrame(delta_rows)
        df_sum.to_csv(OUTDIR / f"summary_per_seed_{metric_name}.csv", index=False)
        df_delta.to_csv(OUTDIR / f"delta_proj_per_seed_{metric_name}.csv", index=False)

        # ---- Aggregates (by role/condition; across all pairs and seeds) ----
        agg_rows = []
        for name in HF_MODELS.keys():
            sub = df_sum[df_sum.model_name == name]
            for cond in ["impersonal", "personal"]:
                s = sub[sub.condition == cond]
                agg_rows.append({
                    "metric": metric_name,
                    "model_name": name, "condition": cond,
                    "proj_mean_mean": float(s["proj_mean"].mean()) if not s.empty else float("nan"),
                    "proj_mean_std": float(s["proj_mean"].std(ddof=1)) if len(s) > 1 else 0.0,
                    "N_rows": int(len(s))
                })
        pd.DataFrame(agg_rows).to_csv(OUTDIR / f"summary_agg_{metric_name}.csv", index=False)

    # ===== Cross-metric sensitivity (Spearman on Δproj) =====
    def spearman(a, b):
        ra = pd.Series(a).rank(method="average"); rb = pd.Series(b).rank(method="average")
        return float(ra.corr(rb, method="pearson"))

    df_all = pd.DataFrame(all_metric_delta)
    rows = []
    for m1 in sorted(df_all["metric"].unique()):
        for m2 in sorted(df_all["metric"].unique()):
            if m1 >= m2: continue
            t1 = df_all[df_all.metric == m1].groupby(["model_name", "pair_id", "seed"])["delta_proj"].mean().reset_index()
            t2 = df_all[df_all.metric == m2].groupby(["model_name", "pair_id", "seed"])["delta_proj"].mean().reset_index()
            t  = pd.merge(t1, t2, on=["model_name", "pair_id", "seed"], suffixes=("_1", "_2"))
            rho = spearman(t["delta_proj_1"], t["delta_proj_2"]) if not t.empty else float("nan")
            rows.append({"metric_1": m1, "metric_2": m2, "spearman_delta_proj": rho})
    Path(OUTDIR / "sensitivity").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTDIR / "sensitivity" / "sensitivity_delta.csv", index=False)
    print("Sensitivity (Δproj) saved to:", (OUTDIR / "sensitivity" / "sensitivity_delta.csv").resolve())

if __name__ == "__main__":
    main()
