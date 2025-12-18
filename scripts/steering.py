# ------------------------------------------------------------
# 0. Imports & Model Load
# ------------------------------------------------------------

from huggingface_hub import login
login()

import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

### model_name should be selected among the following models ###
# LLaMA3: "meta-llama/Meta-Llama-3-8B-Instruct"
# Qwen2: "Qwen/Qwen2-1.5B"
# Gemma2: "google/gemma-2-2b"
# OLMo: "allenai/Olmo-3-1025-7B"

model_name = "model_name"
print("Loading:", model_name)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

model.eval()

print("Num layers:", model.config.num_hidden_layers)
print("Hidden dim:", model.config.hidden_size)



# ------------------------------------------------------------
# 1. Utility Functions (Embedding Extraction)
# ------------------------------------------------------------

def extract_hidden_single(text: str, layer: int):
    """
    Returns mean-pooled hidden embedding at a specific layer.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=200
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        h = outputs.hidden_states[layer]          # [1, seq, dim]
        mask = inputs["attention_mask"].unsqueeze(-1)

        masked = h * mask
        summed = masked.sum(dim=1)
        length = mask.sum(dim=1).clamp(min=1)
        emb = summed / length                     # [1, dim]

    return emb.squeeze(0)



# ------------------------------------------------------------
# 2. Cosine Similarity
# ------------------------------------------------------------

def cos_sim(a, b):
    a = a / (a.norm(p=2) + 1e-8)
    b = b / (b.norm(p=2) + 1e-8)
    return float((a * b).sum())



# ------------------------------------------------------------
# 3. Layer Detection (Internal Embedding Difference)
# ------------------------------------------------------------

def extract_last_token_states(df, model, tokenizer):
    cache = {}

    for _, row in df.iterrows():
        for key in ["logical", "pragmatic"]:
            sent = row[key]

            if sent not in cache:
                inputs = tokenizer(
                    sent,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128
                ).to(model.device)

                with torch.no_grad():
                    outputs = model(
                        **inputs,
                        output_hidden_states=True
                    )

                hidden_states = outputs.hidden_states

                layer_embs = []
                for layer in range(len(hidden_states)):
                    emb = hidden_states[layer][0, -1].detach().cpu()
                    layer_embs.append(emb)

                cache[sent] = layer_embs

    return cache


def cosine_score_for_layer_cached(df, cache, layer):
    scores = []

    for _, row in df.iterrows():
        l = cache[row["logical"]][layer].numpy().reshape(1, -1)
        p = cache[row["pragmatic"]][layer].numpy().reshape(1, -1)

        sim_lp = cosine_similarity(l, p)[0][0]
        scores.append(1 - sim_lp)  # separation score

    return np.mean(scores)


def detect_layer(df, model, tokenizer):
    print("\nDetecting layers ...")

    cache = extract_last_token_states(df, model, tokenizer)

    for L in range(model.config.num_hidden_layers):
        s = cosine_score_for_layer_cached(df, cache, L)
        print(f"Layer {L}: score={s:.4f}")



# ------------------------------------------------------------
# 4. Steering Vector
# ------------------------------------------------------------

def compute_steering_vector(df, layer):
    logical_vecs = []
    pragmatic_vecs = []

    for _, row in df.iterrows():
        logical_vecs.append(extract_hidden_single(row["logical"], layer))
        pragmatic_vecs.append(extract_hidden_single(row["pragmatic"], layer))

    logical_mean = torch.stack(logical_vecs).mean(dim=0)
    pragmatic_mean = torch.stack(pragmatic_vecs).mean(dim=0)

    v = pragmatic_mean - logical_mean
    v = v / v.norm()     # normalize
    return v



# ------------------------------------------------------------
# 5. Compute steering direction from multiple layers
# ------------------------------------------------------------

def compute_direction_multi_layers(df, layers):
    vecs = []

    print(f"\nComputing steering direction from {len(df)} samples...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing direction"):
        p = row["pragmatic"]
        l = row["logical"]

        vec_p = []
        vec_l = []

        for layer in layers:
            vec_p.append(extract_hidden_single(p, layer))
            vec_l.append(extract_hidden_single(l, layer))

        vec_p = torch.cat(vec_p)
        vec_l = torch.cat(vec_l)

        vecs.append(vec_p - vec_l)

    direction = torch.stack(vecs).mean(dim=0)
    direction = direction / direction.norm()

    print("Direction computed.")
    return direction



# ------------------------------------------------------------
# 6. Steering evaluation metric
# ------------------------------------------------------------

def evaluate_with_steering(df, out_path, layers, direction, fixed_alpha=None):
    rows = []
    print(f"\nEvaluating {len(df)} samples...")

    # --- helper: multi-layer embedding ---
    def get_multi_layer_emb(text, layers):
        embs = []
        for L in layers:
            embs.append(extract_hidden_single(text, L))
        return torch.cat(embs)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        row_dict = row.to_dict()

        anchor    = row["anchor"]
        logical   = row["logical"]
        pragmatic = row["pragmatic"]
        grade     = row["grade"]

        # --- baseline embeddings (multi-layer) ---
        a_emb = get_multi_layer_emb(anchor, layers)
        l_emb = get_multi_layer_emb(logical, layers)
        p_emb = get_multi_layer_emb(pragmatic, layers)

        # baseline similarities
        sim_int_log  = cos_sim(a_emb, l_emb)
        sim_int_prag = cos_sim(a_emb, p_emb)

        pref_int = 1 if sim_int_prag > sim_int_log else 0

        # alpha
        alpha = fixed_alpha if fixed_alpha is not None else alpha_from_grade(grade)

        # steering
        steered = (a_emb + alpha * direction)
        steered = steered / steered.norm()

        sim_st_log  = cos_sim(steered, l_emb)
        sim_st_prag = cos_sim(steered, p_emb)

        pref_st = 1 if sim_st_prag > sim_st_log else 0

        # merge results
        row_dict.update({
            "alpha": alpha,

            "sim_internal_logical": sim_int_log,
            "sim_internal_pragmatic": sim_int_prag,
            "pref_internal": pref_int,

            "sim_steered_logical": sim_st_log,
            "sim_steered_pragmatic": sim_st_prag,
            "pref_steered": pref_st,
        })

        rows.append(row_dict)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Saved:", out_path)



# ------------------------------------------------------------
# 7. α (alpha) setting
# ------------------------------------------------------------

grade_to_numeric = {
    "A": 1.0,
    "B": 0.75,
    "C": 0.5,
    "D": 0.25,
    "E": 0.0
}


### ALPHA Setting ###
# LLaMA3: 1.0 to 20.0
# Qwen2: 1.0 to 80.0
# Gemma2: 1.0 to 145.0
# OLMo: 1.0 to 15.0

ALPHA_MIN = 1.0
ALPHA_MAX = 15.0

def alpha_from_grade(g):
    g_num = grade_to_numeric[g]
    return ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * g_num



# ------------------------------------------------------------
# 8. Main pipeline
# ------------------------------------------------------------

# Load dataset
df = pd.read_csv("GraSD.csv")


# Detect Layers
df_probe = (
    df.groupby(["weak", "strong"], group_keys=False)
      .apply(lambda x: x.sample(n=1))
      .sample(n=121, random_state=0)
)

detect_layer(df_probe, model, tokenizer)


# Direction
df_dir = (
    df.groupby(["weak", "strong"], group_keys=False)
      .apply(lambda x: x.sample(n=3, random_state=42))
      .reset_index(drop=True)
)

### Detected layers ###
# LLaMa3: [11, 12, 13, 14]
# Qwen2: [14, 15, 16, 17]
# Gemma2: [12, 13, 14, 15, 16]
# OLMo: [15, 16, 17, 18]
layers = [15, 16, 17, 18]

direction = compute_direction_multi_layers(df_dir, layers)


# Evaluate (Uniform alpha)
evaluate_with_steering(
    df,
    "results_uniform.csv",
    layers,
    direction,
    fixed_alpha=15.0
)


# Evaluate (Graded alpha)
evaluate_with_steering(
    df,
    "results_grade.csv",
    layers,
    direction
)

