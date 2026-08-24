"""
Stage 1 — English accent modelling and error diagnostics.

Characterises the accent signal in frozen wav2vec 2.0 embeddings across American,
British and Indian English: audio preprocessing, a leakage/confound audit, accent
classification with statistical testing, a clip-length (truncation) fairness sweep,
embedding geometry and Whisper WER. Fits and freezes the classifier, scaler and
train/test split, which Stage 2 reuses.

Input : balanced English accent corpus (AccentDB) + metadata spreadsheet.
Output: figures, fitted artifacts (RandomForest, scaler, split) and metrics
        written to the Stage 1 output directory.


Author: Anjali Chakraborty (candidate 307998), MRes Advanced AI, University of Sussex.
Pretrained models and libraries used here are the work of others (reference in attribution
note in README.md), the pipeline design and analysis code are my own.
"""


import os
import re
import sys
import json
import argparse
import warnings
import platform
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
from scipy.stats import chi2_contingency

import joblib
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support,
)
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, \
    RemoveMultipleSpaces, Strip
import umap

warnings.filterwarnings("ignore")

################################# CONFIG #################################

ROOT_DIR         = "/Users/anjali98/Desktop/SUSSEX_LABS/data/AccentDataset"
METADATA_FILE    = "/Users/anjali98/Desktop/SUSSEX_LABS/data/metadata_with_whisper.xlsx"
OUTPUT_DIR       = "/Users/anjali98/Desktop/SUSSEX_LABS/outputs/stage1_v2"
WHISPER_MODEL    = "base"
WAV2VEC_MODEL    = "facebook/wav2vec2-base"

ACCENTS          = ["american", "british", "indian"]
CLIP_LENS        = [0.5, 1.0, 2.0, 3.0]
RF_N_ESTIMATORS  = 400
SEEDS            = [42, 43, 44]      # multi-seed; first seed used for plots
RANDOM_STATE     = SEEDS[0]
TEST_SIZE        = 0.20
TARGET_SR        = 16000
TARGET_RMS       = 0.1
TRIM_TOP_DB      = 30
N_BOOTSTRAP      = 1000
N_PERMUTATIONS   = 5000              # permutation tests
ALPHA            = 0.05

# Control flags
RUN_FEATURE_EXTRACTION = True     # False -> use cached embeddings
RUN_WHISPER_DECODE     = False    # True -> re-decode with Whisper
RUN_COMPUTE_WER        = True     # needs transcript_ref column

#############################################################

os.makedirs(OUTPUT_DIR, exist_ok=True)

CACHE_EMBEDDINGS = os.path.join(OUTPUT_DIR, "embeddings_cache.npz")
CACHE_SPLIT      = os.path.join(OUTPUT_DIR, "train_test_split.npz")
MODEL_PATH       = os.path.join(OUTPUT_DIR, "rf_model.joblib")    
SCALER_PATH      = os.path.join(OUTPUT_DIR, "scaler.joblib")       

COLORS = {"american": "#4C72B0", "british": "#DD8452", "indian": "#55A868"}

STEREO_WORDS = {
    "american": [
        "sidewalk", "elevator", "faucet", "apartment", "candy", "truck",
        "vacation", "fall", "downtown", "garbage", "cookie", "diaper",
        "soccer", "freeway", "eraser", "cell", "zee", "gotten", "math",
        "chips", "movies", "awesome", "sure", "bucks", "parking", "yard",
    ],
    "british": [
        "pavement", "lift", "tap", "flat", "sweets", "lorry", "holiday",
        "autumn", "city centre", "rubbish", "biscuit", "nappy", "football",
        "motorway", "rubber", "mobile", "zed", "got", "maths", "crisps",
        "cinema", "brilliant", "cheers", "quid", "car park", "garden",
        "whilst", "fortnight", "reckon", "proper", "lovely",
    ],
    "indian": [
        "prepone", "revert", "kindly", "do the needful", "out of station",
        "good name", "pass out", "updation", "cent percent", "itself",
        "only", "na", "yaar", "lakh", "crore", "godown", "batchmate",
        "timepass", "eve-teasing", "miscreant", "absconding",
    ],
}

STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "of", "to", "and", "that", "was",
    "for", "on", "are", "with", "as", "at", "be", "this", "by", "from",
    "or", "but", "not", "have", "had", "has", "he", "she", "they", "we",
    "you", "i", "my", "his", "her", "its", "our", "your", "their", "do",
    "did", "does", "will", "would", "can", "could", "should", "may",
    "might", "shall", "been", "being", "so", "if", "then", "there",
    "when", "what", "which", "who", "how", "all", "each", "any", "one",
    "two", "no", "up",
}



################################# UTILITIES #################################


def section(title):
    print(f"\n{'═' * 72}\n  {title}\n{'═' * 72}")


def save_fig(fig, name):
    p = os.path.join(OUTPUT_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Saved: {p}")


def wer_transform():
    return Compose([ToLowerCase(), RemovePunctuation(),
                    RemoveMultipleSpaces(), Strip()])


def load_metadata():
    ext = Path(METADATA_FILE).suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(METADATA_FILE, engine="openpyxl")
    else:
        try:
            df = pd.read_csv(METADATA_FILE, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(METADATA_FILE, encoding="latin-1")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def rows_with_ref(df):
    col = "transcript_ref"
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in metadata.")
    mask = (df[col].notna()
            & (df[col].astype(str).str.strip() != "")
            & (df[col].astype(str).str.lower() != "nan"))
    out = df[mask].copy()
    coverage = len(out) / len(df) * 100
    print(f"  transcript_ref coverage: {len(out)}/{len(df)} ({coverage:.1f}%)")
    if coverage < 50:
        print("  ⚠  Low coverage — WER results may not be representative.")
    return out


################################# Statistics helpers #################################

def bootstrap_metric(values_true, values_pred, metric_fn,
                     n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    # Generic paired bootstrap 95% CI for any metric_fn(y_true, y_pred)
    rng = np.random.RandomState(seed)
    n = len(values_true)
    vals = []
    vt = np.asarray(values_true, dtype=object)
    vp = np.asarray(values_pred, dtype=object)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        vals.append(metric_fn(vt[idx], vp[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)


def bootstrap_accuracy(y_true, y_pred, **kw):
    return bootstrap_metric(y_true, y_pred,
                            lambda t, p: accuracy_score(list(t), list(p)),
                            **kw)


def bootstrap_wer(refs, hyps, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    # Bootstrap 95% CI on corpus WER (resampling utterances)
    rng = np.random.RandomState(seed)
    n = len(refs)
    refs = list(refs)
    hyps = list(hyps)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            vals.append(wer([refs[i] for i in idx],
                            [hyps[i] for i in idx]) * 100)
        except Exception:
            pass
    if not vals:
        return 0.0, 0.0, 0.0
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)


def permutation_test_gap(group_a_correct, group_b_correct,
                         n_perm=N_PERMUTATIONS, seed=RANDOM_STATE):
    
    #Two-sided permutation test for the difference in mean correctness (accuracy) between two accent groups. Returns (observed_gap, p_value)

    rng = np.random.RandomState(seed)
    a = np.asarray(group_a_correct, dtype=float)
    b = np.asarray(group_b_correct, dtype=float)
    observed = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = pooled[:n_a].mean() - pooled[n_a:].mean()
        if abs(diff) >= abs(observed):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return float(observed), float(p)


def holm_bonferroni(pvals_dict, alpha=ALPHA):
    
    # Holm–Bonferroni step-down correction. pvals_dict: {label: p}. Returns {label: (p, reject_bool)}
    
    items = sorted(pvals_dict.items(), key=lambda kv: kv[1])
    m = len(items)
    results = {}
    rejected_so_far = True
    for rank, (label, p) in enumerate(items):
        threshold = alpha / (m - rank)
        reject = rejected_so_far and (p <= threshold)
        if not reject:
            rejected_so_far = False
        results[label] = (p, reject)
    return results


################################# Audio #################################

def preprocess_audio(path, target_sr=TARGET_SR, top_db=TRIM_TOP_DB):
    # Load → 16 kHz mono → trim silence → per-clip RMS normalise
    y, _ = librosa.load(str(path), sr=target_sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=top_db)
    rms = np.sqrt(np.mean(y ** 2))
    if rms > 0:
        y = y / rms * TARGET_RMS
    return y, target_sr


def crop_fixed_start(y, sr, duration):
    n = int(duration * sr)
    if len(y) >= n:
        return y[:n]
    return np.concatenate([y, np.zeros(n - len(y))])


################################# STEP 1 — DATASET VALIDATION & EDA #################################


def step1_eda(df):
    section("STEP 1 — Dataset Validation & EDA")

    print(f"  Total rows : {len(df)}")
    print(f"  Columns    : {list(df.columns)}")
    print(f"  Accents    :\n{df['accent'].value_counts().to_string()}")

    missing = df.isnull().sum()
    if missing.any():
        print(f"\n  ⚠  Missing values:\n{missing[missing > 0]}")
    else:
        print("  ✔  No missing values in core columns")

    missing_audio = [r["path"] for _, r in df.iterrows()
                     if not os.path.exists(str(r["path"]))]
    print(f"  Audio files missing: {len(missing_audio)}/{len(df)}")

    # Full duration audit
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Step 1 — Audio Duration Distribution per Accent",
                 fontsize=13, fontweight="bold")
    all_dur = {}
    for ax, acc in zip(axes, ACCENTS):
        sub = df[df["accent"] == acc]
        durs = []
        for _, row in sub.iterrows():
            try:
                y, sr = librosa.load(row["path"], sr=None, mono=True,
                                     duration=60)
                durs.append(len(y) / sr)
            except Exception:
                pass
        all_dur[acc] = durs
        if durs:
            ax.hist(durs, bins=20, color=COLORS[acc], edgecolor="black",
                    alpha=0.85)
            ax.axvline(np.mean(durs), color="red", linestyle="--",
                       label=f"Mean={np.mean(durs):.1f}s")
        ax.set_title(f"{acc.capitalize()} (n={len(durs)})")
        ax.set_xlabel("Duration (s)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    save_fig(fig, "s1_step1_duration_distribution.png")

    medians = {a: np.median(v) for a, v in all_dur.items() if v}
    print(f"\n  Median durations: { {a: f'{v:.2f}s' for a, v in medians.items()} }")
    if medians and max(medians.values()) - min(medians.values()) > 0.5:
        print("  ⚠  Duration imbalance > 0.5s — per-clip normalisation applied.")
    return all_dur



################################# STEP 2 — LEAKAGE AUDIT #################################


def step2_leakage_audit(df):
    
    section("STEP 2 — Leakage & Confound Audit")
    audit = {"has_speaker": False, "has_source": False,
             "source_confounded": None, "n_speakers": None}

    # Speaker column
    speaker_col = None
    for cand in ["speaker", "speaker_id", "spk", "spk_id", "client_id"]:
        if cand in df.columns:
            speaker_col = cand
            break

    if speaker_col:
        audit["has_speaker"] = True
        n_spk = df[speaker_col].nunique()
        audit["n_speakers"] = int(n_spk)
        per_acc = df.groupby("accent")[speaker_col].nunique()
        print(f"  Speaker column found: '{speaker_col}' — {n_spk} unique speakers")
        print(f"  Speakers per accent:\n{per_acc.to_string()}")
        ratio = len(df) / n_spk
        print(f"  Utterances per speaker (mean): {ratio:.1f}")
        if ratio > 5:
            print("  ⚠  Many utterances per speaker → a naive random split WILL")
            print("     leak speakers across train/test. Using speaker-disjoint")
            print("     (GroupShuffleSplit) as the primary split.")
        else:
            print("  ✔  Low utterance-per-speaker ratio; leakage risk modest,")
            print("     but speaker-disjoint split still used as primary.")
    else:
        print("  ⚠  NO speaker column found in metadata.")
        print("     → Cannot rule out speaker leakage. This MUST be stated as a")
        print("       limitation in the paper: the 3-way accuracy may partially")
        print("       reflect speaker identity rather than accent.")

    # Source column
    source_col = None
    for cand in ["source", "dataset", "origin", "corpus"]:
        if cand in df.columns:
            source_col = cand
            break

    if source_col:
        audit["has_source"] = True
        ct = pd.crosstab(df["accent"], df[source_col])
        print(f"\n  Source column found: '{source_col}'")
        print(f"  Accent × Source contingency table:\n{ct.to_string()}")
        chi2, p, dof, _ = chi2_contingency(ct)
        audit["source_confounded"] = bool(p < ALPHA)
        print(f"  Chi-squared test: χ²={chi2:.2f}, dof={dof}, p={p:.2e}")
        if p < ALPHA:
            print("  ⚠  Accent and source are STATISTICALLY DEPENDENT.")
            print("     The classifier may be learning recording-channel or")
            print("     corpus artefacts instead of accent. Report this and,")
            print("     if any accent maps 1:1 to a source, treat accuracy")
            print("     claims with caution / run a channel-control experiment.")
        else:
            print("  ✔  No significant accent×source dependence detected.")
    else:
        print("\n  ℹ  No source/dataset column found. If the corpus was merged")
        print("     from multiple Kaggle datasets, ADD a source column and")
        print("     re-run this audit — accent↔source confounding is the most")
        print("     likely explanation for anomalies like American@0.5s=100%.")

    audit["speaker_col"] = speaker_col
    audit["source_col"] = source_col
    return audit



################################# STEP 3 — wav2vec2 FEATURE EXTRACTION #################################


def step3_extract_embeddings(df, processor, model, device):
    section("STEP 3 — wav2vec2 Feature Extraction (full clips)")

    if os.path.exists(CACHE_EMBEDDINGS) and not RUN_FEATURE_EXTRACTION:
        print("  Loading cached embeddings …")
        data = np.load(CACHE_EMBEDDINGS, allow_pickle=True)
        return data["X"], data["y"], data["utt_ids"].tolist()

    le = LabelEncoder()
    le.fit(ACCENTS)
    embeddings, labels, utt_ids = [], [], []

    for idx, (_, row) in enumerate(df.iterrows()):
        try:
            y, sr = preprocess_audio(row["path"])
            inputs = processor(y, sampling_rate=sr, return_tensors="pt",
                               padding=True).input_values.to(device)
            with torch.no_grad():
                hidden = model(inputs).last_hidden_state
            emb = hidden.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(emb)
            labels.append(row["accent"])
            utt_ids.append(row["utt_id"])
        except Exception as e:
            print(f"  ⚠  Skipping {row['utt_id']}: {e}")
        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{len(df)} …")

    X = np.array(embeddings)
    y_enc = le.transform(labels)
    np.savez(CACHE_EMBEDDINGS, X=X, y=y_enc, utt_ids=np.array(utt_ids))
    print(f"  ✔  Embeddings shape: {X.shape}  — cached to {CACHE_EMBEDDINGS}")
    return X, y_enc, utt_ids



################################# STEP 4 — CLASSIFICATION: MULTI-SEED, MULTI-MODEL, SIGNIFICANCE #################################
          
def _make_split(df, y_enc, seed, audit):
    # Speaker-disjoint split if possible, else stratified random
    indices = np.arange(len(y_enc))
    if audit.get("has_speaker"):
        groups = df[audit["speaker_col"]].values
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                random_state=seed)
        train_idx, test_idx = next(gss.split(indices, y_enc, groups))
        return train_idx, test_idx, "speaker_disjoint"
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, stratify=y_enc, random_state=seed)
    return train_idx, test_idx, "stratified_random"


def _build_models(seed):
    # RF is the primary model, linear probe and MLP are comparison baselines
    return {
        "rf": RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, random_state=seed, n_jobs=-1),
        "linear_probe": LogisticRegression(
            max_iter=2000, random_state=seed),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256,), max_iter=500,
            random_state=seed, early_stopping=True),
    }


def step4_classify(df, X, y_enc, audit):
    
    # Multi-seed, multi-model classification with significance testing
    
    
    section("STEP 4 — Accent Classification "
            "(multi-seed, RF primary + baselines)")

    results = defaultdict(list)   # model → [acc per seed]
    per_acc_results = defaultdict(lambda: defaultdict(list))
    primary = {}                  # artefacts from seed[0] RF for plots/Step 5
    split_kind = None

    for seed in SEEDS:
        train_idx, test_idx, split_kind = _make_split(df, y_enc, seed, audit)
        scaler = StandardScaler().fit(X[train_idx])
        X_tr, X_te = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
        y_tr, y_te = y_enc[train_idx], y_enc[test_idx]

        for name, model in _build_models(seed).items():
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            acc = accuracy_score(y_te, y_pred)
            results[name].append(acc)
            for i, a in enumerate(ACCENTS):
                mask = y_te == i
                if mask.sum():
                    per_acc_results[name][a].append(
                        accuracy_score(y_te[mask], y_pred[mask]))

            if seed == SEEDS[0] and name == "rf":
                primary = dict(clf=model, scaler=scaler,
                               train_idx=train_idx, test_idx=test_idx,
                               y_te=y_te, y_pred=y_pred)

    print(f"\n  Split strategy: {split_kind}")
    if split_kind == "stratified_random" and not audit.get("has_speaker"):
        print("  ⚠  No speaker info → results may be inflated by speaker leakage.")

    #  mean ± std across seeds
    print(f"\n  Accuracy over {len(SEEDS)} seeds (mean ± std):")
    print(f"  {'Model':14s} | {'Accuracy':>18s}")
    print("  " + "-" * 36)
    for name in ["rf", "linear_probe", "mlp"]:
        accs = np.array(results[name]) * 100
        print(f"  {name:14s} | {accs.mean():6.2f}% ± {accs.std():.2f}")

    print(f"\n  Per-accent accuracy (RF, mean ± std over seeds):")
    for a in ACCENTS:
        vals = np.array(per_acc_results["rf"][a]) * 100
        print(f"    {a.capitalize():10s}: {vals.mean():6.2f}% ± {vals.std():.2f}")

    # Bootstrap CI on the primary (seed[0]) RF run
    y_te, y_pred = primary["y_te"], primary["y_pred"]
    acc_mean, acc_lo, acc_hi = bootstrap_accuracy(y_te, y_pred)
    print(f"\n  Primary RF (seed={SEEDS[0]}) accuracy: {acc_mean*100:.2f}% "
          f"[95% CI: {acc_lo*100:.1f}–{acc_hi*100:.1f}%]")
    print(classification_report(y_te, y_pred, target_names=ACCENTS))

    # Pairwise permutation tests + Holm correction
    print("  Pairwise accent accuracy gaps (permutation test, Holm-corrected):")
    correct = (y_te == y_pred).astype(int)
    pvals = {}
    gaps = {}
    for a, b in combinations(range(len(ACCENTS)), 2):
        ca = correct[y_te == a]
        cb = correct[y_te == b]
        gap, p = permutation_test_gap(ca, cb)
        label = f"{ACCENTS[a]} vs {ACCENTS[b]}"
        pvals[label] = p
        gaps[label] = gap * 100
    corrected = holm_bonferroni(pvals)
    for label, (p, reject) in corrected.items():
        star = "SIGNIFICANT" if reject else "n.s."
        print(f"    {label:22s}: gap={gaps[label]:+6.2f}pp  p={p:.4f}  [{star}]")

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[a.capitalize() for a in ACCENTS],
                yticklabels=[a.capitalize() for a in ACCENTS],
                linewidths=0.5, ax=ax)
    ax.set_title(f"Accent Confusion Matrix (seed={SEEDS[0]}, {split_kind})",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    save_fig(fig, "s1_step4_confusion_matrix.png")

    # Model comparison bar chart with seed error bars
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    names = ["rf", "linear_probe", "mlp"]
    means = [np.mean(results[n]) * 100 for n in names]
    stds = [np.std(results[n]) * 100 for n in names]
    bars = ax2.bar(["Random Forest", "Linear Probe", "MLP"], means,
                   yerr=stds, capsize=6,
                   color=["#4C72B0", "#DD8452", "#55A868"], edgecolor="black")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"Model Comparison (mean ± std over {len(SEEDS)} seeds)",
                  fontweight="bold")
    for bar, m, s in zip(bars, means, stds):
        ax2.text(bar.get_x() + bar.get_width() / 2, m + s + 0.4,
                 f"{m:.1f}±{s:.1f}", ha="center", fontweight="bold", fontsize=9)
    save_fig(fig2, "s1_step4_model_comparison.png")

    # Persist frozen artefacts
    np.savez(CACHE_SPLIT, train_idx=primary["train_idx"],
             test_idx=primary["test_idx"])
    joblib.dump(primary["clf"], MODEL_PATH)
    joblib.dump(primary["scaler"], SCALER_PATH)
    print(f"  ✔  Frozen RF → {MODEL_PATH}")
    print(f"  ✔  Scaler   → {SCALER_PATH}")
    print(f"  ✔  Split    → {CACHE_SPLIT}")

    stats_out = {
        "split_strategy": split_kind,
        "seeds": SEEDS,
        "accuracy_mean_std": {n: [float(np.mean(results[n])),
                                  float(np.std(results[n]))] for n in names},
        "primary_rf_ci": [acc_mean, acc_lo, acc_hi],
        "pairwise_tests": {k: {"gap_pp": gaps[k], "p": v[0],
                               "significant_holm": v[1]}
                           for k, v in corrected.items()},
    }
    return (primary["clf"], primary["scaler"], primary["train_idx"],
            primary["test_idx"], primary["y_pred"], stats_out)



################################# STEP 5 — LENGTH-SENSITIVITY  #################################


def step5_length_sensitivity(df, clf, scaler, test_idx, y_enc,
                             processor, model, device, audit):

    section("STEP 5 — Length-Sensitivity ( frozen RF, cropped clips)")
    print("  ℹ  RF and scaler from Step 4 are FROZEN.")

    le = LabelEncoder()
    le.fit(ACCENTS)

    df_test = df.iloc[test_idx].reset_index(drop=True)
    y_test = y_enc[test_idx]

    data = np.load(CACHE_EMBEDDINGS, allow_pickle=True)
    X_full_test = scaler.transform(data["X"][test_idx])
    y_pred_full = clf.predict(X_full_test)

    full_acc_per = {}
    for i, acc in enumerate(ACCENTS):
        mask = y_test == i
        full_acc_per[acc] = (accuracy_score(y_test[mask], y_pred_full[mask]) * 100
                             if mask.sum() > 0 else 0.0)

    results = {acc: [] for acc in ACCENTS}
    overall = []
    per_source = defaultdict(lambda: defaultdict(list))  # [FIX-7]
    source_col = audit.get("source_col")

    for clip_len in CLIP_LENS:
        print(f"  Clip = {clip_len}s … ", end="")
        crop_embs, crop_labels, crop_sources = [], [], []

        for _, row in df_test.iterrows():
            try:
                y, sr = preprocess_audio(row["path"])
                y_c = crop_fixed_start(y, sr, clip_len)
                inputs = processor(y_c, sampling_rate=sr, return_tensors="pt",
                                   padding=True).input_values.to(device)
                with torch.no_grad():
                    hidden = model(inputs).last_hidden_state
                emb = hidden.mean(dim=1).squeeze().cpu().numpy()
                crop_embs.append(emb)
                crop_labels.append(row["accent"])
                crop_sources.append(row[source_col] if source_col else "all")
            except Exception:
                pass

        if not crop_embs:
            for acc in ACCENTS:
                results[acc].append(0.0)
            overall.append(0.0)
            continue

        X_crop = scaler.transform(np.array(crop_embs))
        y_crop = le.transform(crop_labels)
        y_hat = clf.predict(X_crop)

        ov = accuracy_score(y_crop, y_hat) * 100
        overall.append(ov)
        print(f"Overall={ov:.1f}%", end="  | ")

        for i, acc in enumerate(ACCENTS):
            mask = y_crop == i
            acc_acc = (accuracy_score(y_crop[mask], y_hat[mask]) * 100
                       if mask.sum() > 0 else 0.0)
            results[acc].append(acc_acc)
            print(f"{acc}={acc_acc:.1f}%", end="  ")

            # [FIX-7] per-source breakdown at this clip length
            if source_col:
                srcs = np.array(crop_sources)
                for s in np.unique(srcs):
                    smask = mask & (srcs == s)
                    if smask.sum() > 0:
                        per_source[clip_len][(acc, s)].append(
                            accuracy_score(y_crop[smask], y_hat[smask]) * 100)
        print()

    # Anomaly diagnostics
    if source_col and per_source:
        print("\n  [ANOMALY CHECK] Per-source accuracy at each clip length:")
        for clip_len in CLIP_LENS:
            for (acc, s), vals in sorted(per_source[clip_len].items()):
                print(f"    {clip_len}s  {acc:10s} src={s}: {np.mean(vals):.1f}%")
        print("  → If one source dominates an accent's short-clip accuracy,")
        print("    the anomaly is a corpus artefact, not an accent property.")

    # Table + fairness gaps
    print(f"\n  TABLE I ")
    print(f"  {'Accent':12s}  {'Full':>6s}" +
          "".join(f"  {l}s" for l in CLIP_LENS))
    print("  " + "─" * 56)
    for acc in ACCENTS:
        print(f"  {acc.capitalize():12s}  {full_acc_per[acc]:>5.1f}%" +
              "".join(f"  {v:5.1f}%" for v in results[acc]))

    gaps = []
    for li, cl in enumerate(CLIP_LENS):
        accs_at = {acc: results[acc][li] for acc in ACCENTS}
        gap = max(accs_at.values()) - min(accs_at.values())
        gaps.append(gap)
        flag = "⚠ >5pp" if gap > 5 else "✔"
        print(f"  Gap @ {cl}s: {gap:.1f}pp {flag}")
    full_gap = max(full_acc_per.values()) - min(full_acc_per.values())
    print(f"  Gap @ full: {full_gap:.1f}pp (baseline)")

    # Plots 
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("Accuracy vs Speech Duration",
                 fontsize=12, fontweight="bold")
    for acc in ACCENTS:
        ax.plot(CLIP_LENS, results[acc], marker="o", label=acc.capitalize(),
                color=COLORS[acc], linewidth=2.2, markersize=8)
        ax.scatter([CLIP_LENS[-1] + 0.4], [full_acc_per[acc]],
                   marker="*", color=COLORS[acc], s=180, zorder=5)
    star = mlines.Line2D([], [], marker="*", color="grey", linestyle="None",
                         markersize=10, label="★ full-clip baseline")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [star], labels=labels + ["★ full-clip baseline"],
              fontsize=9, loc="lower right")
    ax.axhline(100/3, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Clip Length (s)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(-5, 110)
    ax.set_xticks(CLIP_LENS)
    ax.grid(alpha=0.3)
    save_fig(fig, "s1_step5_accuracy_curves.png")

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle("Accent Fairness under Truncation",
                  fontsize=13, fontweight="bold")
    ax_l = axes2[0]
    bar_colors = ["#C0392B" if g > 5 else "#27AE60" for g in gaps]
    bars = ax_l.bar([f"{l}s" for l in CLIP_LENS], gaps,
                    color=bar_colors, edgecolor="black", linewidth=0.8)
    ax_l.axhline(5, color="red", linestyle="--", linewidth=1.5,
                 label="5pp threshold")
    ax_l.axhline(full_gap, color="grey", linestyle=":", linewidth=1.5,
                 label=f"Full-clip gap ({full_gap:.1f}pp)")
    for bar, g in zip(bars, gaps):
        ax_l.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                  f"{g:.1f}pp", ha="center", fontsize=10, fontweight="bold")
    ax_l.set_ylabel("Accuracy Gap (pp)")
    ax_l.set_title("Fairness Gap per Duration")
    ax_l.legend(fontsize=9)

    ax_r = axes2[1]
    x = np.arange(len(CLIP_LENS))
    w = 0.25
    for acc, offset in zip(ACCENTS, [-w, 0, w]):
        drops = [full_acc_per[acc] - results[acc][li]
                 for li in range(len(CLIP_LENS))]
        ax_r.bar(x + offset, drops, w, label=acc.capitalize(),
                 color=COLORS[acc], edgecolor="black", linewidth=0.6)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels([f"{l}s" for l in CLIP_LENS])
    ax_r.set_ylabel("Accuracy Drop from Baseline (pp)")
    ax_r.set_title("Per-Accent Degradation")
    ax_r.legend(fontsize=9)
    ax_r.axhline(0, color="black", linewidth=0.8)
    ax_r.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig2, "s1_step5_fairness_gap.png")

    mat = np.array([[results[acc][li] for li in range(len(CLIP_LENS))]
                    for acc in ACCENTS])
    mat_df = pd.DataFrame(mat, index=[a.capitalize() for a in ACCENTS],
                          columns=[f"{l}s" for l in CLIP_LENS])
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.heatmap(mat_df, annot=True, fmt=".1f", cmap="RdYlGn",
                vmin=0, vmax=100, linewidths=0.5, ax=ax3,
                cbar_kws={"label": "Accuracy (%)"})
    ax3.set_title("Accuracy Heatmap: Accent × Duration",
                  fontsize=11, fontweight="bold")
    save_fig(fig3, "s1_step5_heatmap.png")

    return results, gaps, full_acc_per



################################# STEP 6 — UMAP #################################


def step6_umap(X, y_enc):
    section("STEP 6 — UMAP Visualisation")
    print("  ℹ  UMAP is ILLUSTRATIVE only — quantitative claims should use")
    print("     the 768-d space (e.g. cosine similarities), not 2-D distances.")

    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE,
                        n_neighbors=15)
    X_2d = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, acc in enumerate(ACCENTS):
        mask = y_enc == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=COLORS[acc],
                   label=acc.capitalize(), alpha=0.6, s=18, edgecolors="none")
    ax.set_title("UMAP — wav2vec2 Accent Embeddings (illustrative)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=2)
    save_fig(fig, "s1_step6_umap.png")

    np.save(os.path.join(OUTPUT_DIR, "umap_2d_stage1.npy"), X_2d)
    np.save(os.path.join(OUTPUT_DIR, "umap_labels_stage1.npy"), y_enc)
    return X_2d


################################# STEP 7 — WHISPER ASR + WER (with CIs and permutation tests) #################################


def step7_whisper_decode(df):
    if not RUN_WHISPER_DECODE:
        section("STEP 7a — Whisper Decode [SKIPPED — using existing transcripts]")
        return df

    section("STEP 7a — Whisper ASR Decode")
    import whisper
    wmodel = whisper.load_model(WHISPER_MODEL)
    transcripts = []
    for _, row in df.iterrows():
        try:
            res = wmodel.transcribe(row["path"], language="en")
            transcripts.append(res["text"].strip())
        except Exception as e:
            transcripts.append("")
            print(f"  ⚠  {row['utt_id']}: {e}")
    df = df.copy()
    df["whisper_transcript"] = transcripts
    out = METADATA_FILE.replace(".xlsx", "_redecoded.xlsx")
    df.to_excel(out, index=False)
    print(f"  ✔  Saved to {out}")
    return df


def step7_compute_wer(df):
    section("STEP 7b — WER per Accent (bootstrap CIs + permutation tests)")

    if not RUN_COMPUTE_WER:
        print("  [SKIPPED]")
        return {}, {}

    df_ref = rows_with_ref(df)
    tf = wer_transform()
    wer_overall = {}
    wer_cis = {}
    utt_wers = defaultdict(list)
    per_accent_pairs = {}

    for acc in ACCENTS:
        sub = df_ref[df_ref["accent"] == acc]
        if sub.empty:
            print(f"  [{acc}] No reference transcripts.")
            continue
        refs = [tf(str(t)) for t in sub["transcript_ref"]]
        hyps = [tf(str(t)) for t in sub["whisper_transcript"]]
        per_accent_pairs[acc] = (refs, hyps)

        w_mean, w_lo, w_hi = bootstrap_wer(refs, hyps)          # [FIX-3]
        wer_overall[acc] = round(wer(refs, hyps) * 100, 2)
        wer_cis[acc] = (w_lo, w_hi)
        print(f"  {acc.capitalize():10s}: WER = {wer_overall[acc]:.2f}% "
              f"[95% CI: {w_lo:.2f}–{w_hi:.2f}]  (n={len(sub)})")

        for r, h in zip(refs, hyps):
            try:
                utt_wers[acc].append(wer([r], [h]) * 100)
            except Exception:
                utt_wers[acc].append(0.0)

    # Pairwise permutation tests on per-utterance WER
    if len(utt_wers) >= 2:
        print("\n  Pairwise WER gaps (permutation test, Holm-corrected):")
        pvals, gaps = {}, {}
        for a, b in combinations([x for x in ACCENTS if utt_wers[x]], 2):
            gap, p = permutation_test_gap(utt_wers[a], utt_wers[b])
            label = f"{a} vs {b}"
            pvals[label] = p
            gaps[label] = gap
        corrected = holm_bonferroni(pvals)
        for label, (p, reject) in corrected.items():
            star = "SIGNIFICANT" if reject else "n.s."
            print(f"    {label:22s}: gap={gaps[label]:+6.2f}pp  "
                  f"p={p:.4f}  [{star}]")

    if wer_overall:
        best = min(wer_overall, key=wer_overall.get)
        worst = max(wer_overall, key=wer_overall.get)
        gap = wer_overall[worst] - wer_overall[best]
        print(f"\n  Best  : {best.capitalize()} ({wer_overall[best]:.2f}%)")
        print(f"  Worst : {worst.capitalize()} ({wer_overall[worst]:.2f}%)")
        print(f"  Gap   : {gap:.2f}pp "
              f"{'⚠ >5pp' if gap > 5 else '✔ <5pp'}")

    # Top-5 worst utterances per accent (unchanged, useful for error analysis)
    print("\n  Top-5 highest-WER utterances per accent:")
    for acc in ACCENTS:
        sub = df_ref[df_ref["accent"] == acc].reset_index(drop=True)
        pairs = []
        for _, row in sub.iterrows():
            r = tf(str(row["transcript_ref"]))
            h = tf(str(row["whisper_transcript"]))
            try:
                w_v = wer([r], [h]) * 100
            except Exception:
                w_v = 0.0
            pairs.append((w_v, row["utt_id"], r, h))
        pairs.sort(reverse=True)
        print(f"\n  [{acc.capitalize()}]")
        for w_v, uid, ref, hyp in pairs[:5]:
            print(f"    {uid}  WER={w_v:.0f}%")
            print(f"      REF: {ref}")
            print(f"      HYP: {hyp}")

    # Plot with CI error bars
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Step 7 — Whisper WER Analysis", fontsize=14,
                 fontweight="bold")

    ax = axes[0]
    accs_present = [a for a in ACCENTS if a in wer_overall]
    vals = [wer_overall[a] for a in accs_present]
    errs = np.array([[wer_overall[a] - wer_cis[a][0] for a in accs_present],
                     [wer_cis[a][1] - wer_overall[a] for a in accs_present]])
    bars = ax.bar([a.capitalize() for a in accs_present], vals,
                  yerr=errs, capsize=6,
                  color=[COLORS[a] for a in accs_present], edgecolor="black")
    ax.set_ylabel("WER (%)")
    ax.set_title("Overall WER by Accent (95% bootstrap CI)")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax2 = axes[1]
    box_data = [utt_wers[a] for a in ACCENTS if utt_wers[a]]
    box_labs = [a.capitalize() for a in ACCENTS if utt_wers[a]]
    bp = ax2.boxplot(box_data, patch_artist=True, labels=box_labs)
    for patch, acc in zip(bp["boxes"], ACCENTS):
        patch.set_facecolor(COLORS[acc])
        patch.set_alpha(0.7)
    ax2.set_ylabel("Per-Utterance WER (%)")
    ax2.set_title("WER Distribution")
    plt.tight_layout()
    save_fig(fig, "s1_step7_wer.png")

    return wer_overall, utt_wers



################################# STEP 8 — SHORT-CLIP ANOMALY INVESTIGATION #################################


def step8_anomaly(df, audit):
    section("STEP 8 — Short-Clip Anomaly Investigation")

    source_col = audit.get("source_col")
    stats = defaultdict(lambda: defaultdict(list))

    for acc in ACCENTS:
        sub = df[df["accent"] == acc].sample(
            min(50, len(df[df["accent"] == acc])), random_state=RANDOM_STATE)
        for _, row in sub.iterrows():
            if not os.path.exists(str(row["path"])):
                continue
            try:
                y, sr = librosa.load(str(row["path"]), sr=16000, mono=True)
            except Exception:
                continue
            stats[acc]["duration"].append(len(y) / sr)
            rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            stats[acc]["silence_ratio"].append(np.mean(rms_db < -40))
            half = int(0.5 * sr)
            stats[acc]["energy_first_half"].append(
                np.mean(y[:half] ** 2) if len(y) >= half else 0)
            stats[acc]["energy_second_half"].append(
                np.mean(y[half:2 * half] ** 2) if len(y) >= 2 * half else 0)
            if source_col:
                stats[acc]["source"].append(row[source_col])

    print(f"\n  {'Accent':10s} | {'Avg Dur':>8s} | {'Silence%':>9s} | "
          f"{'E[0–0.5]':>10s} | {'E[0.5–1]':>10s}")
    print("  " + "-" * 55)
    for acc in ACCENTS:
        d = stats[acc]
        if not d["duration"]:
            continue
        print(f"  {acc.capitalize():10s} | "
              f"{np.mean(d['duration']):>7.2f}s | "
              f"{np.mean(d['silence_ratio']) * 100:>8.1f}% | "
              f"{np.mean(d['energy_first_half']) * 1e4:>9.4f} | "
              f"{np.mean(d['energy_second_half']) * 1e4:>9.4f}")

    if source_col:
        print("\n  Source composition of anomaly sample:")
        for acc in ACCENTS:
            if stats[acc].get("source"):
                print(f"    {acc.capitalize()}: "
                      f"{dict(Counter(stats[acc]['source']))}")
        print("  → If one accent is dominated by a single source with distinct")
        print("    recording characteristics, treat short-clip results for that")
        print("    accent as potentially confounded.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Step 8 — Short-Clip Anomaly: Audio Characteristics",
                 fontsize=13, fontweight="bold")
    panels = [
        ("duration", "Duration (s)", "Clip Duration"),
        ("silence_ratio", "Silence Ratio", "Silence Ratio"),
        ("energy_first_half", "Energy (x1e-4)", "Energy: First 0.5s"),
    ]
    for ax, (key, ylabel, title) in zip(axes, panels):
        data = [stats[a][key] for a in ACCENTS if stats[a][key]]
        labels = [a.capitalize() for a in ACCENTS if stats[a][key]]
        bp = ax.boxplot(data, patch_artist=True, labels=labels)
        for patch, acc in zip(bp["boxes"], ACCENTS):
            patch.set_facecolor(COLORS[acc])
            patch.set_alpha(0.75)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
    save_fig(fig, "s1_step8_anomaly.png")
    return stats



################################# STEP 9 — VOCABULARY LEAKAGE #################################


def step9_vocab_leakage(df):
    section("STEP 9 — Vocabulary Leakage Check")


    have_ref = ("transcript_ref" in df.columns
                and df["transcript_ref"].notna().sum() > 0.5 * len(df))
    if have_ref:
        text_col = "transcript_ref"
        print("  Using GROUND-TRUTH transcripts (transcript_ref).")
    else:
        text_col = "whisper_transcript"
        print("  ⚠  transcript_ref unavailable/sparse — falling back to Whisper")
        print("     transcripts. CAVEAT for the paper: Whisper output already")
        print("     reflects ASR accent bias, so lexical 'leakage' measured this")
        print("     way conflates corpus vocabulary with recognition artefacts.")

    def hit_rate(texts, wordlist):
        tokens = re.findall(r"\b\w+\b", " ".join(str(t) for t in texts).lower())
        wl = {w.lower() for w in wordlist}
        hits = sum(1 for t in tokens if t in wl)
        return hits / len(tokens) if tokens else 0

    matrix = {}
    for probe in ACCENTS:
        matrix[probe] = {}
        for corpus_acc in ACCENTS:
            sub = df[df["accent"] == corpus_acc]
            txts = list(sub[text_col].dropna().astype(str))
            matrix[probe][corpus_acc] = hit_rate(txts, STEREO_WORDS[probe])

    print(f"\n  {'Probe →':20s}" +
          "".join(f"{a.capitalize():>12s}" for a in ACCENTS))
    print("  " + "-" * (20 + 12 * len(ACCENTS)))
    for probe in ACCENTS:
        row_str = f"  {probe.capitalize() + ' words':20s}"
        for ca in ACCENTS:
            marker = " ◄" if probe == ca else "  "
            row_str += f"{matrix[probe][ca] * 100:>10.2f}%{marker}"
        print(row_str)

    for probe in ACCENTS:
        in_grp = matrix[probe][probe]
        out_max = max(matrix[probe][a] for a in ACCENTS if a != probe)
        if in_grp > out_max:
            print(f"  ⚠  {probe.capitalize()}: in-group lexical signal present "
                  f"({in_grp*100:.2f}% > {out_max*100:.2f}%) — accent labels")
            print(f"     partially recoverable from WORDS alone; note as confound.")
        else:
            print(f"  ✔  {probe.capitalize()}: no in-group lexical advantage "
                  f"({in_grp*100:.2f}% ≤ {out_max*100:.2f}%)")

    mat_df = pd.DataFrame(matrix).T[ACCENTS]
    mat_df.index = [a.capitalize() for a in ACCENTS]
    mat_df.columns = [a.capitalize() for a in ACCENTS]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(mat_df * 100, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Hit rate (%)"})
    ax.set_title(f"Stereotypical Vocabulary Leakage ({text_col})",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Corpus Accent")
    ax.set_ylabel("Probe Word-List")
    save_fig(fig, "s1_step9_vocab_leakage.png")

    return matrix



################################# STEP 10 — SUMMARY DASHBOARD #################################


def step10_dashboard(wer_results, utt_wers, da_results, da_gaps, da_full,
                     stats_out):
    section("STEP 10 — Summary Dashboard")
    if not wer_results:
        print("  [Skipped — no WER results]")
        return

    fig = plt.figure(figsize=(20, 9))
    fig.suptitle("Accent Fairness Diagnostic — Summary Dashboard (v2)",
                 fontsize=15, fontweight="bold", y=0.99)

    ax1 = fig.add_subplot(2, 4, 1)
    bars = ax1.bar([a.capitalize() for a in wer_results],
                   list(wer_results.values()),
                   color=[COLORS[a] for a in wer_results], edgecolor="black")
    ax1.set_title("WER by Accent (%)")
    for bar, val in zip(bars, wer_results.values()):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")

    ax2 = fig.add_subplot(2, 4, 2)
    vdata = [utt_wers[a] for a in ACCENTS if utt_wers[a]]
    vlabs = [a.capitalize() for a in ACCENTS if utt_wers[a]]
    parts = ax2.violinplot(vdata, showmedians=True)
    for pc, acc in zip(parts["bodies"], ACCENTS):
        pc.set_facecolor(COLORS[acc])
        pc.set_alpha(0.7)
    ax2.set_xticks(range(1, len(vlabs) + 1))
    ax2.set_xticklabels(vlabs)
    ax2.set_title("Per-Utterance WER")

    ax3 = fig.add_subplot(2, 4, 3)
    for acc in ACCENTS:
        ax3.plot(CLIP_LENS, da_results[acc], marker="o",
                 label=acc.capitalize(), color=COLORS[acc], linewidth=2)
        ax3.scatter([CLIP_LENS[-1] + 0.35], [da_full[acc]],
                    marker="*", color=COLORS[acc], s=120, zorder=5)
    ax3.set_title("Accuracy Curves")
    ax3.set_ylim(-5, 110)
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(2, 4, 4)
    bar_cols = ["#C0392B" if g > 5 else "#27AE60" for g in da_gaps]
    ax4.bar([f"{l}s" for l in CLIP_LENS], da_gaps,
            color=bar_cols, edgecolor="black")
    ax4.axhline(5, color="red", linestyle="--", linewidth=1.2)
    ax4.set_title("Fairness Gap per Duration")
    ax4.set_ylabel("Gap (pp)")

    ax5 = fig.add_subplot(2, 1, 2)
    ax5.axis("off")
    best = min(wer_results, key=wer_results.get)
    worst = max(wer_results, key=wer_results.get)
    wer_gap = wer_results[worst] - wer_results[best]

    sig_lines = []
    for label, d in stats_out.get("pairwise_tests", {}).items():
        sig = "SIGNIFICANT" if d["significant_holm"] else "n.s."
        sig_lines.append(f"  {label}: gap={d['gap_pp']:+.2f}pp "
                         f"p={d['p']:.4f} [{sig}]")

    text = (
        "FAIRNESS SUMMARY (v2 — with statistical testing)\n\n"
        f"Split strategy: {stats_out.get('split_strategy')}\n"
        f"Seeds: {stats_out.get('seeds')}\n\n"
        f"WER: Best={best.capitalize()} {wer_results[best]:.2f}% | "
        f"Worst={worst.capitalize()} {wer_results[worst]:.2f}% | "
        f"Gap={wer_gap:.2f}pp\n\n"
        "Classification pairwise tests (Holm-corrected):\n"
        + "\n".join(sig_lines)
    )
    ax5.text(0.02, 0.95, text, transform=ax5.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.85))
    plt.tight_layout()
    save_fig(fig, "s1_step10_dashboard.png")



################################# LIVE RECORDING + INFERENCE (--live) #################################


def live_record_and_classify(processor, model, device, duration=4.0):
    # Loads the FROZEN joblib artefacts — no refitting
    section("LIVE DEMO — Record & Classify (frozen model)")
    try:
        import sounddevice as sd
    except ImportError:
        print("  ✘  Install sounddevice:  pip install sounddevice")
        return

    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        print("  ✘  Frozen model not found. Run the full pipeline first.")
        return
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"  ✔  Loaded frozen RF + scaler from {OUTPUT_DIR}")

    print(f"  Recording {duration}s … speak now!")
    audio = sd.rec(int(duration * TARGET_SR), samplerate=TARGET_SR,
                   channels=1, dtype="float32")
    sd.wait()
    y = audio.flatten()
    print(f"  ✔  Recorded {len(y) / TARGET_SR:.1f}s")

    y, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    rms = np.sqrt(np.mean(y ** 2))
    if rms > 0:
        y = y / rms * TARGET_RMS

    inputs = processor(y, sampling_rate=TARGET_SR, return_tensors="pt",
                       padding=True).input_values.to(device)
    with torch.no_grad():
        hidden = model(inputs).last_hidden_state
    emb = hidden.mean(dim=1).squeeze().cpu().numpy().reshape(1, -1)
    emb_sc = scaler.transform(emb)

    pred = clf.predict(emb_sc)[0]
    proba = clf.predict_proba(emb_sc)[0]
    le = LabelEncoder()
    le.fit(ACCENTS)
    pred_label = le.inverse_transform([pred])[0]

    print(f"\n  Predicted accent: {pred_label.capitalize()}")
    for i, acc in enumerate(ACCENTS):
        print(f"    {acc.capitalize():12s}: {proba[i] * 100:.1f}%")

    try:
        import whisper
        import soundfile as sf
        wmodel = whisper.load_model(WHISPER_MODEL)
        tmp = os.path.join(OUTPUT_DIR, "_live_temp.wav")
        sf.write(tmp, y, TARGET_SR)
        result = wmodel.transcribe(tmp, language="en")
        print(f"\n  Whisper transcript: {result['text'].strip()}")
        os.remove(tmp)
    except ImportError:
        print("  ℹ  Install openai-whisper for live transcription.")

    return pred_label, proba



################################# MAIN #################################


def library_versions():
    # Log environment for reproducibility
    import sklearn
    import transformers
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sklearn": sklearn.__version__,
        "librosa": librosa.__version__,
        "umap": umap.__version__,
    }
    return versions


def main():
    parser = argparse.ArgumentParser(description="Stage 1 v2 — Accent Modelling")
    parser.add_argument("--live", action="store_true",
                        help="Record from microphone and classify accent")
    args = parser.parse_args()

    print("\n" + "★" * 72)
    print("  STAGE 1 (v2) — ENGLISH ACCENT MODELLING & ERROR DIAGNOSTICS")
    print("  Multi-seed | Leakage-audited | Significance-tested")
    print("★" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL)
    model_w2v = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL)
    model_w2v.eval()
    model_w2v.to(device)
    print(f"\n  wav2vec2 on: {device}")

    if args.live:
        live_record_and_classify(processor, model_w2v, device)
        return

    df = load_metadata()

    step1_eda(df)
    audit = step2_leakage_audit(df)                                

    X, y_enc, utt_ids = step3_extract_embeddings(df, processor,
                                                 model_w2v, device)

    clf, scaler, train_idx, test_idx, y_pred, stats_out = \
        step4_classify(df, X, y_enc, audit)                        

    da_results, da_gaps, da_full = step5_length_sensitivity(
        df, clf, scaler, test_idx, y_enc, processor, model_w2v, device,
        audit)                                                     

    step6_umap(X, y_enc)

    df = step7_whisper_decode(df)
    wer_results, utt_wers = step7_compute_wer(df)                  

    step8_anomaly(df, audit)                                       
    step9_vocab_leakage(df)                                        

    if wer_results:
        step10_dashboard(wer_results, utt_wers, da_results, da_gaps,
                         da_full, stats_out)

    y_te = y_enc[test_idx]
    summary = {
        "version": "stage1_v2",
        "methodology": "Train full, test crops; "
                       "multi-seed; leakage-audited",
        "environment": library_versions(),                         
        "leakage_audit": {k: v for k, v in audit.items()},
        "classification": stats_out,
        "primary_seed_accuracy": float(accuracy_score(y_te, y_pred)),
        "wer_per_accent": wer_results,
        "designA_per_accent": da_results,
        "designA_gaps_pp": {str(CLIP_LENS[i]): round(da_gaps[i], 2)
                            for i in range(len(CLIP_LENS))},
        "designA_full_acc": da_full,
        "outputs": sorted(os.listdir(OUTPUT_DIR)),
    }
    with open(os.path.join(OUTPUT_DIR, "stage1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "★" * 72)
    print(f"  STAGE 1 (v2) COMPLETE — outputs in: {OUTPUT_DIR}")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    {fname}")
    print("★" * 72 + "\n")


if __name__ == "__main__":
    main()
