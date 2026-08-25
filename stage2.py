"""
Stage 2 — Hindi ASR and cross-lingual misinterpretation.

Reuses Stage 1's frozen artifacts to audit an English–Hindi pipeline end to end,
the language-identification routing gate, cross-lingual decoding by a Hindi ASR,
entropy-based confusion mining, a native-calibrated and null-baselined cross-lingual
risk score (CLRS), joint embedding geometry and case studies. Locates the accent
disparity at the router rather than the recogniser.

Input : Stage 1 outputs (embeddings, fitted classifier, scaler) + native Hindi
        speech (OpenSLR-103 / MUCS) for plausibility calibration.
Output: figures, cross-lingual results, CLRS, and summary written to the Stage 2
        output directory.

Run   : python stage2.py          (full pipeline)
        python stage2.py --live    (record from mic and run end to end)

Author: Anjali Chakraborty (candidate 307998), MRes Advanced AI, University of Sussex.
Pretrained models and libraries used here are the work of others (see the attribution
note in README.md), the pipeline idea, design , steps of code and analysis are my own.
"""

import os
import re
import sys
import json
import time
import argparse
import warnings
import platform
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import chi2_contingency

import joblib
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2Processor, Wav2Vec2Model,
    Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
)
from sklearn.preprocessing import LabelEncoder
from jiwer import wer as compute_wer, cer as compute_cer
import umap

warnings.filterwarnings("ignore")



################################# CONFIG #################################


STAGE1_DIR       = "/Users/anjali98/Desktop/SUSSEX_LABS/outputs/stage1_v2"
METADATA_FILE    = "/Users/anjali98/Desktop/SUSSEX_LABS/data/metadata_with_whisper.xlsx"

# Hindi data
HINDI_CV_DIR     = "/Users/anjali98/Desktop/SUSSEX_LABS/data/hindi_common_voice"
HINDI_CV_HF      = "mozilla-foundation/common_voice_11_0"
HINDI_CV_LANG    = "hi"

# Output
OUTPUT_DIR       = "/Users/anjali98/Desktop/SUSSEX_LABS/outputs/stage2_v2"

# Models
HINDI_ASR_MODEL  = "ai4bharat/indicwav2vec-hindi"
USE_PRETRAINED_ASR = True
WAV2VEC_MODEL    = "facebook/wav2vec2-base"       # embeddings only (match Stage 1)
LANG_ID_MODEL    = "speechbrain/lang-id-voxlingua107-ecapa"
NLLB_MODEL       = "facebook/nllb-200-distilled-600M"   


# Build:  lmplz -o 4 < hindi_train_text.txt > hindi_4gram.arpa
KENLM_PATH       = None   
KENLM_ALPHA      = 0.5
KENLM_BETA       = 1.5

# Fine-tuning hyper-params (only if USE_PRETRAINED_ASR=False)
HINDI_EPOCHS     = 10
HINDI_LR         = 3e-5
HINDI_BATCH      = 4
WARMUP_STEPS     = 200
EARLY_STOP_PAT   = 3
HINDI_MAX_TRAIN  = 3000
HINDI_MAX_VAL    = 500
HINDI_MAX_TEST   = 500
ENGLISH_MAX_PER_ACCENT = None

# Analysis
ACCENTS          = ["american", "british", "indian"]
ENTROPY_THRESHOLDS = [0.5, 1.0, 1.5, 2.0]   # sensitivity sweep
PRIMARY_ENTROPY  = 1.0                       # headline threshold (justified by sweep)
MIN_BIGRAM_COUNT = 3
PLAUS_PERCENTILE = 25     # plausible = conf ≥ 10th pct of native Hindi
N_NULL_NOISE     = 100    # white-noise null clips
N_PERMUTATIONS   = 5000   # permutation tests
N_BOOTSTRAP      = 1000
ALPHA            = 0.05
RANDOM_STATE     = 42
TARGET_SR        = 16000
TARGET_RMS       = 0.1
TRIM_TOP_DB      = 30
MIN_LANGID_SEC   = 1.0    # lang-ID needs ≥1s to be meaningful

COLORS = {"american": "#4C72B0", "british": "#DD8452", "indian": "#55A868"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

HINDI_ASR_SAVE   = os.path.join(OUTPUT_DIR, "hindi_asr_model")
NATIVE_CONF_PATH = os.path.join(OUTPUT_DIR, "native_hindi_confidences.json")
STAGE1_MODEL     = os.path.join(STAGE1_DIR, "rf_model.joblib")
STAGE1_SCALER    = os.path.join(STAGE1_DIR, "scaler.joblib")



################################# UTILITIES #################################


def section(title):
    print(f"\n{'═' * 72}\n  {title}\n{'═' * 72}")


def save_fig(fig, name):
    p = os.path.join(OUTPUT_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {p}")


def preprocess_audio(path, target_sr=TARGET_SR, top_db=TRIM_TOP_DB):
    # Same pre-processing as Stage 1 for consistency
    y, _ = librosa.load(str(path), sr=target_sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=top_db)
    rms = np.sqrt(np.mean(y ** 2))
    if rms > 0:
        y = y / rms * TARGET_RMS
    return y, target_sr


def load_stage1_metadata():
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


def load_stage1_embeddings():
    cache = os.path.join(STAGE1_DIR, "embeddings_cache.npz")
    if not os.path.exists(cache):
        raise FileNotFoundError(
            f"Stage 1 embeddings not found at {cache}. Run Stage 1 v2 first.")
    data = np.load(cache, allow_pickle=True)
    return data["X"], data["y"], data["utt_ids"].tolist()


def entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * np.log2(p) for p in probs)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def library_versions():                                            
    import sklearn
    import transformers
    v = {
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "sklearn": sklearn.__version__, "librosa": librosa.__version__,
        "umap": umap.__version__,
        "hindi_asr_model": HINDI_ASR_MODEL,
        "use_pretrained_asr": USE_PRETRAINED_ASR,
        "kenlm": KENLM_PATH is not None,
    }
    try:
        import speechbrain
        v["speechbrain"] = speechbrain.__version__
    except ImportError:
        pass
    return v


################################# Statistics helpers  #################################

def bootstrap_proportion(binary_outcomes, n_boot=N_BOOTSTRAP,
                         seed=RANDOM_STATE):
    # Bootstrap 95% CI on a proportion
    rng = np.random.RandomState(seed)
    x = np.asarray(binary_outcomes, dtype=float)
    n = len(x)
    if n == 0:
        return 0.0, 0.0, 0.0
    vals = [x[rng.randint(0, n, size=n)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)


def bootstrap_wer_pairs(refs, hyps, metric=compute_wer,
                        n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    # Bootstrap 95% CI on corpus WER/CER by resampling utterances
    rng = np.random.RandomState(seed)
    n = len(refs)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            vals.append(metric([refs[i] for i in idx],
                               [hyps[i] for i in idx]) * 100)
        except Exception:
            pass
    if not vals:
        return 0.0, 0.0, 0.0
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)


def permutation_test_gap(a, b, n_perm=N_PERMUTATIONS, seed=RANDOM_STATE):
    # Two-sided permutation test for difference in means of two groups
    rng = np.random.RandomState(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    observed = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(pooled[:n_a].mean() - pooled[n_a:].mean()) >= abs(observed):
            count += 1
    return float(observed), float((count + 1) / (n_perm + 1))


def holm_bonferroni(pvals_dict, alpha=ALPHA):
    items = sorted(pvals_dict.items(), key=lambda kv: kv[1])
    m = len(items)
    results = {}
    rejected_so_far = True
    for rank, (label, p) in enumerate(items):
        reject = rejected_so_far and (p <= alpha / (m - rank))
        if not reject:
            rejected_so_far = False
        results[label] = (p, reject)
    return results


################################# Local NLLB translation #################################

_translator = None

def translate_hi_to_en(text):
    #Local NLLB-200 Hindi -> English. Returns gloss or explanatory placeholder

    global _translator
    text = str(text).strip()
    if not text or text.lower() == "nan":
        return ""
    if _translator is None:
        try:
            from transformers import pipeline
            _translator = pipeline(
                "translation", model=NLLB_MODEL,
                src_lang="hin_Deva", tgt_lang="eng_Latn",
                max_length=128)
        except Exception as e:
            _translator = False
            print(f"     NLLB unavailable ({e}). Glosses disabled.")
    if _translator is False:
        return "(translation unavailable — install sentencepiece & NLLB)"
    try:
        return _translator(text)[0]["translation_text"]
    except Exception as e:
        return f"(translation failed: {e})"



################################# HINDI DATA LOADING  #################################


def _split_data(paths, texts):
    n = len(paths)
    idx = np.random.RandomState(RANDOM_STATE).permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    tr, va, te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
    out = tuple(([paths[i] for i in ix], [texts[i] for i in ix])
                for ix in (tr, va, te))
    print(f"  Split: Train={len(out[0][0])}  Val={len(out[1][0])}  "
          f"Test={len(out[2][0])}")
    return out


def _resolve_audio_paths(df, path_col, clips_dir):
    paths, texts = [], []
    audio_exts = [".mp3", ".wav", ".flac", ".ogg", ""]
    for _, row in df.iterrows():
        raw_path = str(row[path_col])
        sentence = str(row.get("sentence", ""))
        if not sentence.strip() or sentence.lower() == "nan":
            continue
        found = False
        candidates = [raw_path]
        if clips_dir:
            candidates += [os.path.join(clips_dir, raw_path),
                           os.path.join(clips_dir, os.path.basename(raw_path))]
        candidates.append(os.path.join(HINDI_CV_DIR, raw_path))
        for cand in candidates:
            for ext in audio_exts:
                full = cand if ext == "" else os.path.splitext(cand)[0] + ext
                if os.path.exists(full):
                    paths.append(full)
                    texts.append(sentence)
                    found = True
                    break
            if found:
                break
    return paths, texts


def _try_load_openslr(base_dir):
    audio_exts = {".wav", ".mp3", ".flac"}

    def find_audio_and_text(search_dir):
        audio_files, transcript_files = [], []
        for root, dirs, files in os.walk(search_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                full = os.path.join(root, f)
                if ext in audio_exts:
                    audio_files.append(full)
                elif f.lower() in ("transcription.txt", "transcript.txt",
                                   "text", "labels.txt", "utt2text"):
                    transcript_files.append(full)
                elif ext == ".txt" and "transcri" in f.lower():
                    transcript_files.append(full)
        return sorted(audio_files), transcript_files

    def parse_transcripts(tfs):
        utt2text = {}
        for tf in tfs:
            with open(tf, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t", 1)
                    if len(parts) != 2:
                        parts = line.split(" ", 1)
                    if len(parts) == 2:
                        utt2text[parts[0].strip()] = parts[1].strip()
        return utt2text

    def match(audio_files, utt2text):
        paths, texts = [], []
        for af in audio_files:
            stem = os.path.splitext(os.path.basename(af))[0]
            if stem in utt2text:
                paths.append(af)
                texts.append(utt2text[stem])
        return paths, texts

    layouts = [
        (os.path.join(base_dir, "Hindi", "train"),
         os.path.join(base_dir, "Hindi", "test")),
        (os.path.join(base_dir, "train"), os.path.join(base_dir, "test")),
        (os.path.join(base_dir, "Hindi_train"),
         os.path.join(base_dir, "Hindi_test")),
    ]
    for train_dir, test_dir in layouts:
        if os.path.isdir(train_dir) and os.path.isdir(test_dir):
            print(f"  Detected OpenSLR/MUCS format: {train_dir}")
            ta, ttf = find_audio_and_text(train_dir)
            tu = parse_transcripts(ttf)
            train_paths, train_texts = (match(ta, tu) if tu
                                        else (ta, [""] * len(ta)))
            ea, etf = find_audio_and_text(test_dir)
            eu = parse_transcripts(etf)
            test_paths, test_texts = (match(ea, eu) if eu
                                      else (ea, [""] * len(ea)))
            n = len(train_paths)
            idx = np.random.RandomState(RANDOM_STATE).permutation(n)
            n_val = max(1, int(0.1 * n))
            val = ([train_paths[i] for i in idx[:n_val]],
                   [train_texts[i] for i in idx[:n_val]])
            train = ([train_paths[i] for i in idx[n_val:]],
                     [train_texts[i] for i in idx[n_val:]])
            print(f"  Train: {len(train[0])}  Val: {len(val[0])}  "
                  f"Test: {len(test_paths)}")
            return train, val, (test_paths, test_texts)

    audio_files, transcript_files = find_audio_and_text(base_dir)
    if len(audio_files) > 50:
        utt2text = parse_transcripts(transcript_files)
        paths, texts = (match(audio_files, utt2text) if utt2text
                        else (audio_files, [""] * len(audio_files)))
        if paths:
            print(f"  Found {len(paths)} audio files in flat layout.")
            return _split_data(paths, texts)
    return None


def load_hindi_common_voice():
    section("Loading Hindi Speech Data")
    if os.path.exists(HINDI_CV_DIR):
        print(f"  Found local directory: {HINDI_CV_DIR}")
        openslr = _try_load_openslr(HINDI_CV_DIR)
        if openslr is not None:
            return openslr

        clips_dir = None
        for candidate in ["clips", "audio", "wavs"]:
            p = os.path.join(HINDI_CV_DIR, candidate)
            if os.path.isdir(p):
                clips_dir = p
                break

        train_tsv = os.path.join(HINDI_CV_DIR, "train.tsv")
        dev_tsv = os.path.join(HINDI_CV_DIR, "dev.tsv")
        test_tsv = os.path.join(HINDI_CV_DIR, "test.tsv")
        if os.path.exists(train_tsv) and os.path.exists(test_tsv):
            print("  Using pre-split TSVs")
            splits = {}
            for name, tsv in [("train", train_tsv), ("val", dev_tsv),
                              ("test", test_tsv)]:
                if not os.path.exists(tsv):
                    splits[name] = ([], [])
                    continue
                df = pd.read_csv(tsv, sep="\t")
                path_col = "path" if "path" in df.columns else "filename"
                if path_col not in df.columns:
                    splits[name] = ([], [])
                    continue
                splits[name] = _resolve_audio_paths(df, path_col, clips_dir)
                print(f"  {name}: {len(splits[name][0])} utterances")
            return splits["train"], splits["val"], splits["test"]

        for tsv_name in ["validated.tsv", "validated.csv", "other.tsv"]:
            tsv_path = os.path.join(HINDI_CV_DIR, tsv_name)
            if os.path.exists(tsv_path):
                print(f"  Using {tsv_name} (auto-split 80/10/10)")
                sep = "\t" if tsv_name.endswith(".tsv") else ","
                df = pd.read_csv(tsv_path, sep=sep)
                path_col = "path" if "path" in df.columns else "filename"
                if path_col not in df.columns:
                    continue
                paths, texts = _resolve_audio_paths(df, path_col, clips_dir)
                if paths:
                    return _split_data(paths, texts)

    raise FileNotFoundError(
        f"Hindi data not found in {HINDI_CV_DIR}. Download OpenSLR-103 "
        "(openslr.org/103) or Common Voice Hindi and extract there.")



################################# STEP 1 — HINDI ASR #################################


class HindiASRDataset(Dataset):
    def __init__(self, audio_paths, transcripts, processor, max_len_sec=10):
        self.audio_paths = audio_paths
        self.transcripts = transcripts
        self.processor = processor
        self.max_len = int(max_len_sec * TARGET_SR)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        y, sr = librosa.load(self.audio_paths[idx], sr=TARGET_SR, mono=True)
        if len(y) > self.max_len:
            y = y[:self.max_len]
        inputs = self.processor(y, sampling_rate=TARGET_SR,
                                return_tensors="pt", padding=False)
        labels = self.processor.tokenizer(self.transcripts[idx],
                                          return_tensors="pt")
        return {"input_values": inputs.input_values.squeeze(0),
                "labels": labels.input_ids.squeeze(0)}


def collate_fn(batch):
    input_values = [b["input_values"] for b in batch]
    labels = [b["labels"] for b in batch]
    max_in = max(iv.size(0) for iv in input_values)
    padded = torch.zeros(len(batch), max_in)
    attn = torch.zeros(len(batch), max_in, dtype=torch.long)
    for i, iv in enumerate(input_values):
        padded[i, :iv.size(0)] = iv
        attn[i, :iv.size(0)] = 1
    max_lab = max(l.size(0) for l in labels)
    plabels = torch.full((len(batch), max_lab), -100, dtype=torch.long)
    for i, l in enumerate(labels):
        plabels[i, :l.size(0)] = l
    return {"input_values": padded, "attention_mask": attn, "labels": plabels}



_ctc_decoder = None

def get_ctc_decoder(processor):
    global _ctc_decoder
    if _ctc_decoder is not None:
        return _ctc_decoder if _ctc_decoder is not False else None
    if KENLM_PATH is None or not os.path.exists(str(KENLM_PATH)):
        _ctc_decoder = False
        return None
    try:
        from pyctcdecode import build_ctcdecoder
        vocab = list(dict(sorted(
            processor.tokenizer.get_vocab().items(),
            key=lambda kv: kv[1])).keys())
        _ctc_decoder = build_ctcdecoder(
            labels=vocab, kenlm_model_path=KENLM_PATH,
            alpha=KENLM_ALPHA, beta=KENLM_BETA)
        print(f"      KenLM shallow-fusion decoder loaded: {KENLM_PATH}")
        return _ctc_decoder
    except Exception as e:
        print(f"     pyctcdecode/KenLM unavailable ({e}) — greedy decode only.")
        _ctc_decoder = False
        return None


def decode_hindi(model, processor, device, audio_array):
  
    # Returns (transcript, blank_masked_confidence)
    
    model.eval()
    inputs = processor(audio_array, sampling_rate=TARGET_SR,
                       return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    pred_ids = torch.argmax(logits, dim=-1)

    decoder = get_ctc_decoder(processor)
    if decoder is not None:
        lp = log_probs.cpu().numpy()[0]
        transcript = decoder.decode(lp)
    else:
        transcript = processor.batch_decode(pred_ids)[0]

    # blank-masked confidence
    pad_id = processor.tokenizer.pad_token_id
    best_lp = log_probs.gather(2, pred_ids.unsqueeze(-1)).squeeze(-1)
    mask = pred_ids != pad_id
    if mask.sum() > 0:
        confidence = best_lp[mask].mean().item()
    else:
        confidence = float("-inf")   # pure-blank output = no Hindi content

    return transcript.strip(), confidence


def evaluate_hindi_asr(model, processor, device, paths, refs, max_n=500):
    # WER + CER with bootstrap CIs. Also returns per-utterance confidences (used for CLRS calibration in Step 5)
    model.eval()
    all_refs, all_hyps, all_confs = [], [], []
    for path, ref in zip(paths[:max_n], refs[:max_n]):
        if not str(ref).strip():
            continue
        try:
            y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
            hyp, conf = decode_hindi(model, processor, device, y)
            all_refs.append(str(ref).strip())
            all_hyps.append(hyp)
            all_confs.append(conf)
        except Exception:
            pass

    if not all_refs:
        return {"wer": 100.0, "cer": 100.0}, []

    w_mean, w_lo, w_hi = bootstrap_wer_pairs(all_refs, all_hyps, compute_wer)
    c_mean, c_lo, c_hi = bootstrap_wer_pairs(all_refs, all_hyps, compute_cer)
    print(f"  Hindi WER: {w_mean:.2f}% [95% CI: {w_lo:.2f}–{w_hi:.2f}]  "
          f"(n={len(all_refs)})")
    print(f"  Hindi CER: {c_mean:.2f}% [95% CI: {c_lo:.2f}–{c_hi:.2f}]")

    # Save native confidences for CLRS calibration
    finite = [c for c in all_confs if np.isfinite(c)]
    with open(NATIVE_CONF_PATH, "w") as f:
        json.dump({"confidences": finite}, f)
    print(f"      Native-Hindi confidences saved for CLRS calibration "
          f"(n={len(finite)})")

    # Print 3 sample decodes 
    print("\n  Sample decodes (REF vs HYP):")
    for r, h in list(zip(all_refs, all_hyps))[:3]:
        print(f"    REF: {r}")
        print(f"    HYP: {h}\n")

    return {"wer": {"mean": w_mean, "ci": [w_lo, w_hi]},
            "cer": {"mean": c_mean, "ci": [c_lo, c_hi]},
            "n": len(all_refs)}, all_confs


def step1_hindi_asr():
    # Load Hindi-pretrained ASR 
    section("STEP 1 — Hindi ASR "
            f"({'pretrained' if USE_PRETRAINED_ASR else 'fine-tune'}: "
            f"{HINDI_ASR_MODEL})")

    infer_device = get_device()
    print(f"  Inference device: {infer_device}")

    (train_p, train_t), (val_p, val_t), (test_p, test_t) = \
        load_hindi_common_voice()

    if USE_PRETRAINED_ASR:
        model = Wav2Vec2ForCTC.from_pretrained(HINDI_ASR_MODEL).to(infer_device)
        processor = Wav2Vec2Processor.from_pretrained(HINDI_ASR_MODEL)
        print(f"      Loaded pretrained Hindi ASR: {HINDI_ASR_MODEL}")
    else:
        model, processor = _finetune_hindi(
            train_p, train_t, val_p, val_t, infer_device)

    print("\n  Evaluating on held-out Hindi test set …")
    metrics, native_confs = evaluate_hindi_asr(
        model, processor, infer_device, test_p, test_t)

    return model, processor, infer_device, metrics, (test_p, test_t)


def _finetune_hindi(train_p, train_t, val_p, val_t, infer_device):
    
    train_device = torch.device("cpu")
    print(f"  Fine-tuning on {train_device} (CTC/MPS instability workaround)")

    processor = Wav2Vec2Processor.from_pretrained(HINDI_ASR_MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(HINDI_ASR_MODEL)
    model.freeze_feature_encoder()
    model.to(train_device)

    if len(train_p) > HINDI_MAX_TRAIN:
        train_p, train_t = train_p[:HINDI_MAX_TRAIN], train_t[:HINDI_MAX_TRAIN]
    if len(val_p) > HINDI_MAX_VAL:
        val_p, val_t = val_p[:HINDI_MAX_VAL], val_t[:HINDI_MAX_VAL]

    train_loader = DataLoader(
        HindiASRDataset(train_p, train_t, processor), batch_size=HINDI_BATCH,
        shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(
        HindiASRDataset(val_p, val_t, processor), batch_size=HINDI_BATCH,
        shuffle=False, collate_fn=collate_fn, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=HINDI_LR)
    total_steps = len(train_loader) * HINDI_EPOCHS
    warm = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=WARMUP_STEPS)
    main = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - WARMUP_STEPS))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warm, main], milestones=[WARMUP_STEPS])

    best_val, patience, tr_hist, va_hist = float("inf"), 0, [], []
    for epoch in range(HINDI_EPOCHS):
        model.train()
        e_loss, nb = 0, 0
        for batch in train_loader:
            out = model(input_values=batch["input_values"].to(train_device),
                        attention_mask=batch["attention_mask"].to(train_device),
                        labels=batch["labels"].to(train_device))
            optimizer.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            e_loss += out.loss.item()
            nb += 1
        tr_hist.append(e_loss / max(nb, 1))

        model.eval()
        v_loss, vb = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(
                    input_values=batch["input_values"].to(train_device),
                    attention_mask=batch["attention_mask"].to(train_device),
                    labels=batch["labels"].to(train_device))
                v_loss += out.loss.item()
                vb += 1
        va = v_loss / max(vb, 1)
        va_hist.append(va)
        print(f"  Epoch {epoch+1:>2d}/{HINDI_EPOCHS}  "
              f"train={tr_hist[-1]:.4f}  val={va:.4f}")
        if va < best_val:
            best_val, patience = va, 0
            model.save_pretrained(HINDI_ASR_SAVE)
            processor.save_pretrained(HINDI_ASR_SAVE)
        else:
            patience += 1
            if patience >= EARLY_STOP_PAT:
                print(f"  ⏹  Early stopping at epoch {epoch+1}")
                break

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(tr_hist) + 1), tr_hist, marker="o", label="Train")
    ax.plot(range(1, len(va_hist) + 1), va_hist, marker="s", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CTC Loss")
    ax.set_title("Hindi ASR Fine-tuning Curve", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, "s2_step1_training_curve.png")

    model = Wav2Vec2ForCTC.from_pretrained(HINDI_ASR_SAVE).to(infer_device)
    processor = Wav2Vec2Processor.from_pretrained(HINDI_ASR_SAVE)
    return model, processor



################################# STEP 2 — LANGUAGE IDENTIFICATION GATE #################################


def _load_lang_id():
    from speechbrain.inference.classifiers import EncoderClassifier
    clf = EncoderClassifier.from_hparams(
        source=LANG_ID_MODEL,
        savedir=os.path.join(OUTPUT_DIR, "lang_id_cache"))
    
    try:
        clf.hparams.label_encoder.expect_len(107)
    except Exception:
        pass
    return clf


def _langid_probs(lang_clf, y):
    # Run lang-ID on a 16 kHz waveform, return (pred_label, p_hindi, p_english) 
    
    if len(y) < MIN_LANGID_SEC * TARGET_SR:
        return None, None, None
    signal = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    prediction = lang_clf.classify_batch(signal)
    posteriors = prediction[0].squeeze()
    pred_label = prediction[3][0]

    # log-space  -> probability space 
    post = posteriors.detach().cpu()
    if post.max() <= 0:
        post = torch.exp(post)
    post = post / post.sum()

    lab2ind = lang_clf.hparams.label_encoder.lab2ind
    def prob_of(*names):
        for lang, idx in lab2ind.items():
            key = lang.lower().split(":")[0].strip()
            if key in names:
                return float(post[idx])
        return 0.0

    return pred_label, prob_of("hi", "hindi"), prob_of("en", "english")


def _plot_lang_id(res_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Step 2 — Language Identification Gate",
                 fontsize=14, fontweight="bold")
    ax.axis("off")
    col_labels = ["Accent", "Clips", " ->Hindi", " ->English",
                  "Other", "Mean P(Hindi)"]
    table_data = []
    for acc in ACCENTS:
        sub = res_df[res_df["accent"] == acc]
        if sub.empty:
            continue
        as_hi = int(sub["misclassified_as_hindi"].sum())
        as_en = int(sub["classified_as_english"].sum())
        other = len(sub) - as_hi - as_en
        table_data.append([acc.capitalize(), str(len(sub)), str(as_hi),
                           str(as_en), str(other),
                           f"{sub['p_hindi'].mean():.6f}"])
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellLoc="center", loc="upper center",
                     colColours=["#D5E8D4"] * 6)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)
    for i, acc in enumerate(ACCENTS[:len(table_data)]):
        table[i + 1, 0].set_facecolor(COLORS[acc])
        table[i + 1, 0].set_text_props(color="white", fontweight="bold")
    ax.text(0.5, 0.15,
            "If 'Other' is large, treat non-English predictions as gate noise:\n"
            "a deployment gate would route on P(English) vs P(Hindi), not on\n"
            "the argmax over 107 languages.",
            transform=ax.transAxes, ha="center", fontsize=9, fontstyle="italic",
            bbox=dict(boxstyle="round", facecolor="#FFF3CD", alpha=0.9))
    plt.tight_layout()
    save_fig(fig, "s2_step2_lang_id_gate.png")


def step2_lang_id_gate(df):
    section("STEP 2 — Language Identification Gate Analysis")

    cache_path = os.path.join(OUTPUT_DIR, "lang_id_results.csv")
    if os.path.exists(cache_path):
        print(f"      Loading cached results from {cache_path}")
        res_df = pd.read_csv(cache_path)
        _plot_lang_id(res_df)
        return res_df

    try:
        lang_clf = _load_lang_id()
    except ImportError:
        print("     speechbrain not installed — skipping lang-ID analysis.")
        return None

    start = time.time()
    results = []
    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        try:
            # [FIX-4] plain 16 kHz load + trim; NO RMS renormalisation here
            y, _ = librosa.load(str(row["path"]), sr=TARGET_SR, mono=True)
            y, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
            pred_label, p_hi, p_en = _langid_probs(lang_clf, y)
            if pred_label is None:
                continue
            plabel = str(pred_label).lower()
            results.append({
                "utt_id": row["utt_id"],
                "accent": row["accent"],
                "pred_lang": pred_label,
                "p_hindi": p_hi,
                "p_english": p_en,
                "misclassified_as_hindi": plabel.startswith(("hi", "hindi")),
                "classified_as_english": plabel.startswith(("en", "english")),
            })
        except Exception as e:
            if (idx + 1) % 200 == 0:
                print(f"     Error at {idx}: {e}")
        if (idx + 1) % 100 == 0:
            rate = (idx + 1) / (time.time() - start)
            eta = (total - idx - 1) / rate / 60
            print(f"  Processed {idx+1}/{total} ({(idx+1)/total*100:.0f}%)  "
                  f"ETA {eta:.1f} min")

    if not results:
        print("  ✘  No results obtained.")
        return None

    res_df = pd.DataFrame(results)
    print(f"\n  {'Accent':12s} | {' ->Hindi%':>8s} | {' ->English%':>10s} | "
          f"{'Mean P(HI)':>11s} | {'n':>5s}")
    print("  " + "-" * 56)
    for acc in ACCENTS:
        sub = res_df[res_df["accent"] == acc]
        if sub.empty:
            continue
        print(f"  {acc.capitalize():12s} | "
              f"{sub['misclassified_as_hindi'].mean()*100:>7.2f}% | "
              f"{sub['classified_as_english'].mean()*100:>9.2f}% | "
              f"{sub['p_hindi'].mean():>11.6f} | {len(sub):>5d}")

    res_df.to_csv(cache_path, index=False)
    _plot_lang_id(res_df)
    return res_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — CROSS-LINGUAL DECODING
# ─────────────────────────────────────────────────────────────────────────────

def step3_cross_lingual_decode(df, hindi_model, hindi_processor, device,
                               lang_id_df=None):
    section("STEP 3 — Cross-Lingual Decoding (English  -> Hindi ASR)")

    cache_path = os.path.join(OUTPUT_DIR, "cross_lingual_results.csv")
    if os.path.exists(cache_path):
        print(f"      Loading cached results from {cache_path}")
        return pd.read_csv(cache_path)

    start = time.time()
    records = []
    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        try:
            y, _ = preprocess_audio(row["path"])
            hi_text, conf = decode_hindi(hindi_model, hindi_processor,
                                         device, y)
            record = {
                "utt_id": row["utt_id"],
                "accent": row["accent"],
                "transcript_en": str(row.get("whisper_transcript", "")),
                "transcript_hi": hi_text,
                "confidence_hi": conf,
            }
            if lang_id_df is not None:
                lid = lang_id_df[lang_id_df["utt_id"] == row["utt_id"]]
                if not lid.empty:
                    record["p_hindi_langid"] = lid.iloc[0]["p_hindi"]
            records.append(record)
        except Exception as e:
            if (idx + 1) % 200 == 0:
                print(f"     Error at {idx}: {e}")
        if (idx + 1) % 100 == 0:
            rate = (idx + 1) / (time.time() - start)
            eta = (total - idx - 1) / rate / 60
            print(f"  Decoded {idx+1}/{total} ({(idx+1)/total*100:.0f}%)  "
                  f"ETA {eta:.1f} min")
        if (idx + 1) % 500 == 0:
            pd.DataFrame(records).to_csv(cache_path, index=False)
            print(f"    ↳ checkpoint saved ({len(records)} rows)")

    xling_df = pd.DataFrame(records)
    xling_df.to_csv(cache_path, index=False)
    print(f"      Cross-lingual results: {len(xling_df)} utterances")

    print(f"\n  {'Accent':12s} | {'Mean conf':>10s} | {'Non-empty HI%':>14s}")
    print("  " + "-" * 44)
    for acc in ACCENTS:
        sub = xling_df[xling_df["accent"] == acc]
        if sub.empty:
            continue
        finite = sub["confidence_hi"].replace([-np.inf], np.nan).dropna()
        non_empty = (sub["transcript_hi"].fillna("").astype(str)
                     .str.strip() != "").mean() * 100
        print(f"  {acc.capitalize():12s} | {finite.mean():>10.4f} | "
              f"{non_empty:>13.1f}%")
    return xling_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — CONFUSION MINING + THRESHOLD SENSITIVITY [FIX-6]
# ─────────────────────────────────────────────────────────────────────────────

def step4_confusion_mining(xling_df):
    section("STEP 4 — Confusion Pattern Mining "
            f"(threshold sweep: {ENTROPY_THRESHOLDS})")

    def get_bigrams(text):
        words = re.findall(r"\b[a-z]+\b", str(text).lower())
        return [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]

    bigram_map = defaultdict(lambda: defaultdict(list))
    for _, row in xling_df.iterrows():
        for bg in get_bigrams(row["transcript_en"]):
            bigram_map[bg][row["accent"]].append(str(row["transcript_hi"]))

    frequent = {bg: m for bg, m in bigram_map.items()
                if any(len(v) >= MIN_BIGRAM_COUNT for v in m.values())}
    print(f"  Total bigrams: {len(bigram_map)}  |  "
          f"Frequent (≥{MIN_BIGRAM_COUNT}): {len(frequent)}")

    rows = []
    for bg, acc_map in frequent.items():
        for acc in ACCENTS:
            outputs = acc_map.get(acc, [])
            if len(outputs) < MIN_BIGRAM_COUNT:
                continue
            counter = Counter(outputs)
            h = entropy(list(counter.values()))
            top_hi, top_n = counter.most_common(1)[0]
            rows.append({
                "english_bigram": bg, "accent": acc,
                "n_occurrences": len(outputs),
                "n_unique_hindi": len(counter),
                "entropy_bits": round(h, 4),
                "most_common_hindi": top_hi,
                "consistency": top_n / len(outputs),
            })
    cp_df = pd.DataFrame(rows)
    if cp_df.empty:
        print("     No confusion pairs found.")
        return cp_df, {}

    # [FIX-6] Sensitivity sweep
    sweep = {}
    print(f"\n  Stable-pair counts by threshold "
          f"(pair is 'stable' if H < threshold):")
    header = f"  {'Threshold':>9s} |" + "".join(
        f" {a.capitalize():>9s} |" for a in ACCENTS)
    print(header)
    print("  " + "-" * (12 + 12 * len(ACCENTS)))
    for th in ENTROPY_THRESHOLDS:
        counts = {a: int(((cp_df["accent"] == a)
                          & (cp_df["entropy_bits"] < th)).sum())
                  for a in ACCENTS}
        sweep[th] = counts
        print(f"  {th:>8.1f}  |" + "".join(
            f" {counts[a]:>9d} |" for a in ACCENTS))

    # Chi-squared test of stable/unstable × accent at the primary threshold
    cp_df["is_stable"] = cp_df["entropy_bits"] < PRIMARY_ENTROPY
    ct = pd.crosstab(cp_df["accent"], cp_df["is_stable"])
    chi2_res = None
    if ct.shape == (len(ACCENTS), 2):
        chi2, p, dof, _ = chi2_contingency(ct)
        chi2_res = {"chi2": float(chi2), "p": float(p), "dof": int(dof)}
        print(f"\n  Chi-squared (stable × accent @ H<{PRIMARY_ENTROPY}): "
              f"χ²={chi2:.2f}, dof={dof}, p={p:.4f} "
              f"{'[SIGNIFICANT]' if p < ALPHA else '[n.s.]'}")

    # Top-10 consistent pairs per accent (with NLLB gloss)  [FIX-5]
    print("\n  Top consistent confusion pairs (primary threshold):")
    for acc in ACCENTS:
        sub = cp_df[(cp_df["accent"] == acc) & cp_df["is_stable"]] \
            .nlargest(10, "consistency")
        if sub.empty:
            continue
        print(f"\n  [{acc.capitalize()}]")
        for _, row in sub.head(5).iterrows():
            gloss = translate_hi_to_en(row["most_common_hindi"])
            print(f"    \"{row['english_bigram']}\"  -> "
                  f"\"{row['most_common_hindi']}\" (≈ \"{gloss}\")  "
                  f"consistency={row['consistency']:.0%}, "
                  f"H={row['entropy_bits']:.2f} bits")

    cp_df.to_csv(os.path.join(OUTPUT_DIR, "confusion_pairs.csv"), index=False)

    # Plots: sweep + entropy distributions
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Step 4 — Confusion Mining & Threshold Sensitivity",
                 fontsize=13, fontweight="bold")
    ax = axes[0]
    for acc in ACCENTS:
        ax.plot(ENTROPY_THRESHOLDS, [sweep[t][acc] for t in ENTROPY_THRESHOLDS],
                marker="o", label=acc.capitalize(), color=COLORS[acc],
                linewidth=2)
    ax.axvline(PRIMARY_ENTROPY, color="red", linestyle="--",
               label=f"Primary ({PRIMARY_ENTROPY} bit)")
    ax.set_xlabel("Entropy Threshold (bits)")
    ax.set_ylabel("Stable Pair Count")
    ax.set_title("Sensitivity of Stable-Pair Counts to Threshold")
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    for acc in ACCENTS:
        sub = cp_df[cp_df["accent"] == acc]
        if not sub.empty:
            ax2.hist(sub["entropy_bits"], bins=20, alpha=0.5,
                     label=acc.capitalize(), color=COLORS[acc],
                     edgecolor="black")
    ax2.axvline(PRIMARY_ENTROPY, color="red", linestyle="--")
    ax2.set_xlabel("Entropy (bits)")
    ax2.set_ylabel("Count")
    ax2.set_title("Entropy Distribution of English ->Hindi Mappings")
    ax2.legend()
    plt.tight_layout()
    save_fig(fig, "s2_step4_confusion_mining.png")

    return cp_df, {"sweep": sweep, "chi2": chi2_res}



############################### STEP 5 — CLRS + CALIBRATION + NULL BASELINES ###############################


def _plausibility_threshold():
    # conf threshold = PLAUS_PERCENTILE-th pct of NATIVE Hindi confidences
    
    if not os.path.exists(NATIVE_CONF_PATH):
        print("     Native confidences missing — run Step 1 evaluation first.")
        return None
    with open(NATIVE_CONF_PATH) as f:
        confs = json.load(f)["confidences"]
    if not confs:
        return None
    th = float(np.percentile(confs, PLAUS_PERCENTILE))
    print(f"  Plausibility threshold (P{PLAUS_PERCENTILE} of native Hindi "
          f"confidence): {th:.4f}")
    return th


def _noise_null_baseline(hindi_model, hindi_processor, device, conf_th):
    # Decode white-noise clips  -> plausibility floor of the pipeline
  
    rng = np.random.RandomState(RANDOM_STATE)
    plaus = []
    for _ in range(N_NULL_NOISE):
        dur = rng.uniform(3.0, 5.0)
        y = rng.randn(int(dur * TARGET_SR)).astype(np.float32)
        y = y / (np.sqrt(np.mean(y ** 2)) + 1e-9) * TARGET_RMS
        try:
            hi, conf = decode_hindi(hindi_model, hindi_processor, device, y)
            ok = (hi.strip() != "" and np.isfinite(conf)
                  and (conf_th is None or conf > conf_th))
            plaus.append(int(ok))
        except Exception:
            pass
    if not plaus:
        return None
    m, lo, hi_ = bootstrap_proportion(plaus)
    print(f"  NULL (white noise) plausibility rate: {m*100:.1f}% "
          f"[95% CI: {lo*100:.1f}–{hi_*100:.1f}]  (n={len(plaus)})")
    return {"mean": m, "ci": [lo, hi_], "n": len(plaus)}


def step5_risk_scores(xling_df, cp_df, hindi_model, hindi_processor, device):
    section("STEP 5 — Cross-Lingual Risk Scores "
            "(calibrated, null-baselined, significance-tested)")

    if cp_df.empty:
        print("     No confusion pairs — skipping risk scoring.")
        return {}

    conf_th = _plausibility_threshold()                            # [FIX-7]

    stable_bigrams = defaultdict(set)
    for _, row in cp_df[cp_df["is_stable"]].iterrows():
        stable_bigrams[row["accent"]].add(row["english_bigram"])

    def has_stable_mapping(en_text, accent):
        words = re.findall(r"\b[a-z]+\b", str(en_text).lower())
        bgs = {f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)}
        return bool(bgs & stable_bigrams.get(accent, set()))

    def is_plausible(row):
        hi = str(row["transcript_hi"]).strip()
        c = row["confidence_hi"]
        return (hi != "" and hi.lower() != "nan" and np.isfinite(c)
                and (conf_th is None or c > conf_th))

    # Per-utterance binary CLRS outcomes
    per_acc_outcomes = {}
    risk_scores = {}
    print(f"\n  {'Accent':12s} | {'CLRS':>6s} | {'95% CI':>16s} | "
          f"{'Stable%':>8s} | {'Plausible%':>11s}")
    print("  " + "-" * 64)
    for acc in ACCENTS:
        sub = xling_df[xling_df["accent"] == acc]
        if sub.empty:
            risk_scores[acc] = 0.0
            per_acc_outcomes[acc] = []
            continue
        outcomes, n_stab, n_plaus = [], 0, 0
        for _, row in sub.iterrows():
            st = has_stable_mapping(row["transcript_en"], acc)
            pl = is_plausible(row)
            n_stab += int(st)
            n_plaus += int(pl)
            outcomes.append(int(st and pl))
        per_acc_outcomes[acc] = outcomes
        m, lo, hi = bootstrap_proportion(outcomes)                 
        risk_scores[acc] = m * 100
        print(f"  {acc.capitalize():12s} | {m*100:>5.1f}% | "
              f"[{lo*100:>5.1f}–{hi*100:>5.1f}%] | "
              f"{n_stab/len(sub)*100:>7.1f}% | {n_plaus/len(sub)*100:>10.1f}%")

    # noise null
    print("\n  Null baseline (a): white-noise plausibility floor")
    noise_null = _noise_null_baseline(hindi_model, hindi_processor,
                                      device, conf_th)

    # pairwise permutation tests, Holm-corrected
    print("\n  Pairwise CLRS gaps (permutation test, Holm-corrected):")
    pvals, gaps = {}, {}
    for a, b in combinations([x for x in ACCENTS if per_acc_outcomes[x]], 2):
        gap, p = permutation_test_gap(per_acc_outcomes[a],
                                      per_acc_outcomes[b])
        label = f"{a} vs {b}"
        pvals[label] = p
        gaps[label] = gap * 100
    corrected = holm_bonferroni(pvals)
    for label, (p, reject) in corrected.items():
        star = "SIGNIFICANT" if reject else "n.s."
        print(f"    {label:22s}: gap={gaps[label]:+6.2f}pp  "
              f"p={p:.4f}  [{star}]")

  
    #  Plot CLRS with CIs and null floor 
    fig, ax = plt.subplots(figsize=(8, 5))
    means, los, his = [], [], []
    for acc in ACCENTS:
        m, lo, hi = bootstrap_proportion(per_acc_outcomes[acc])
        means.append(m * 100)
        los.append((m - lo) * 100)
        his.append((hi - m) * 100)

    x = [a.capitalize() for a in ACCENTS]
    bars = ax.bar(x, means, yerr=[los, his], capsize=6,
                  color=[COLORS[a] for a in ACCENTS],
                  edgecolor="black", zorder=3)

    # White-noise null floor
    if noise_null:
        ax.axhline(noise_null["mean"] * 100, color="grey", linestyle="--",
                   linewidth=1.5, zorder=2,
                   label=f"White-noise null ({noise_null['mean']*100:.1f}%)")

 
    data_max = max(means + [noise_null["mean"] * 100 if noise_null else 0])
    top = max(1.0, data_max * 1.4)          
    ax.set_ylim(0, top)

    # Value labels sit just above each bar, inside the axis
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + top * 0.03,
                f"{m:.1f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=11)

    ax.set_ylabel("Cross-Lingual Risk Score (%)")
    ax.set_title("Cross-Lingual Risk Score by Accent\n"
                 "(95% bootstrap CI; dashed line = white-noise floor)",
                 fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
    ax.legend(loc="upper right", frameon=True)

    
    if data_max < 1e-6:
        ax.text(0.5, 0.55,
                "No accent exceeds the white-noise floor\n"
                "(CLRS = 0.0% for all varieties)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, style="italic", color="#444444",
                bbox=dict(boxstyle="round,pad=0.5",
                          facecolor="#f2f2f2", edgecolor="#cccccc"))

    fig.tight_layout()
    save_fig(fig, "s2_step5_clrs.png")
    
    return {
        "clrs_pct": risk_scores,
        "plausibility_threshold": conf_th,
        "noise_null": noise_null,
        "pairwise_tests": {k: {"gap_pp": gaps[k], "p": v[0],
                               "significant_holm": v[1]}
                           for k, v in corrected.items()},
    }



############################### STEP 6 — JOINT UMAP + 768-d COSINE SIMILARITY ###############################


def step6_joint_umap(hindi_test_paths, device):
    section("STEP 6 — Joint Embedding Analysis (768-d cosine + UMAP)")
    print("  ℹ  Quantitative claims from the 768-d space; UMAP is illustrative.")

    X_en, y_en, _ = load_stage1_embeddings()
    print(f"  English embeddings: {X_en.shape}")

    base_processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL)
    base_model = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL)
    base_model.eval()
    base_model.to(device)

    max_hindi = min(200, len(hindi_test_paths))
    print(f"  Extracting Hindi embeddings ({max_hindi} clips) …")
    hindi_embs = []
    for path in hindi_test_paths[:max_hindi]:
        try:
            y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
            inputs = base_processor(y, sampling_rate=TARGET_SR,
                                    return_tensors="pt",
                                    padding=True).input_values.to(device)
            with torch.no_grad():
                hidden = base_model(inputs).last_hidden_state
            hindi_embs.append(hidden.mean(dim=1).squeeze().cpu().numpy())
        except Exception:
            pass
    if not hindi_embs:
        print("     No Hindi embeddings extracted.")
        return None
    X_hi = np.array(hindi_embs)

    # 768-d centroid similarities with bootstrap CIs over clips
    hindi_centroid = X_hi.mean(axis=0)
    rng = np.random.RandomState(RANDOM_STATE)
    centroid_sims = {}
    print("\n  Cosine similarity to Hindi centroid (768-d, bootstrap CI):")
    for i, acc in enumerate(ACCENTS):
        Xa = X_en[y_en == i]
        point = 1 - cosine_dist(Xa.mean(axis=0), hindi_centroid)
        boots = []
        for _ in range(200):
            ia = rng.randint(0, len(Xa), size=len(Xa))
            ih = rng.randint(0, len(X_hi), size=len(X_hi))
            boots.append(1 - cosine_dist(Xa[ia].mean(axis=0),
                                         X_hi[ih].mean(axis=0)))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        centroid_sims[acc] = {"sim": float(point),
                              "ci": [float(lo), float(hi)]}
        print(f"    {acc.capitalize():10s}: {point:.4f} "
              f"[95% CI: {lo:.4f}–{hi:.4f}]")

    closest = max(centroid_sims, key=lambda a: centroid_sims[a]["sim"])
    print(f"   -> {closest.capitalize()} English is closest to Hindi.")

    # UMAP (illustrative)
    X_joint = np.vstack([X_en, X_hi])
    labels_joint = np.array(list(y_en) + [-1] * len(X_hi))
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE,
                        n_neighbors=15)
    X_2d = reducer.fit_transform(X_joint)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, acc in enumerate(ACCENTS):
        mask = labels_joint == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=COLORS[acc],
                   label=f"{acc.capitalize()} English", alpha=0.4, s=12,
                   edgecolors="none")
    hm = labels_joint == -1
    ax.scatter(X_2d[hm, 0], X_2d[hm, 1], c="#9B59B6",
               label="Hindi (native)", alpha=0.7, s=25,
               edgecolors="black", linewidth=0.3, marker="^")
    ax.set_title("Joint UMAP — English Accents + Native Hindi (illustrative)\n"
                 f"(Closest in 768-d: {closest.capitalize()}, "
                 f"cos={centroid_sims[closest]['sim']:.4f})",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=2, fontsize=9)
    save_fig(fig, "s2_step6_joint_umap.png")

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    means = [centroid_sims[a]["sim"] for a in ACCENTS]
    errs = [[centroid_sims[a]["sim"] - centroid_sims[a]["ci"][0]
             for a in ACCENTS],
            [centroid_sims[a]["ci"][1] - centroid_sims[a]["sim"]
             for a in ACCENTS]]
    bars = ax2.bar([a.capitalize() for a in ACCENTS], means, yerr=errs,
                   capsize=6, color=[COLORS[a] for a in ACCENTS],
                   edgecolor="black")
    ax2.set_ylabel("Cosine Similarity to Hindi Centroid")
    ax2.set_title("Embedding Proximity to Hindi (768-d, 95% CI)",
                  fontweight="bold")
    for bar, a in zip(bars, ACCENTS):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 centroid_sims[a]["ci"][1] + 0.002,
                 f"{centroid_sims[a]['sim']:.4f}", ha="center", fontsize=9)
    save_fig(fig2, "s2_step6_cosine_similarity.png")

    np.save(os.path.join(OUTPUT_DIR, "joint_umap_2d.npy"), X_2d)
    np.save(os.path.join(OUTPUT_DIR, "joint_umap_labels.npy"), labels_joint)
    return centroid_sims



############################### STEP 7 — CASE STUDIES ###############################


def step7_case_studies(xling_df, lang_id_df=None):
    section("STEP 7 — Case-Study Demonstrations (with local NLLB glosses)")

    valid = xling_df.copy()
    valid["transcript_en"] = valid["transcript_en"].fillna("").astype(str)
    valid["transcript_hi"] = valid["transcript_hi"].fillna("").astype(str)
    valid = valid[
        np.isfinite(valid["confidence_hi"])
        & (valid["transcript_en"].str.strip() != "")
        & (valid["transcript_en"].str.strip().str.lower() != "nan")
        & (valid["transcript_hi"].str.strip() != "")
        & (valid["transcript_hi"].str.strip().str.lower() != "nan")
    ]
    top_cases = valid.nlargest(15, "confidence_hi")

    selected = []
    for acc in ACCENTS:
        acc_cases = top_cases[top_cases["accent"] == acc]
        if not acc_cases.empty:
            selected.append(acc_cases.iloc[0])
    for _, row in top_cases.iterrows():
        if len(selected) >= 5:
            break
        if row["utt_id"] not in [s["utt_id"] for s in selected]:
            selected.append(row)

    case_data = []
    print(f"\n  Selected {len(selected)} case studies:\n")
    for i, row in enumerate(selected):
        gloss = translate_hi_to_en(row["transcript_hi"])           
        p_hindi = None
        if lang_id_df is not None:
            lid = lang_id_df[lang_id_df["utt_id"] == row["utt_id"]]
            if not lid.empty:
                p_hindi = float(lid.iloc[0]["p_hindi"])
        print(f"  Case {i+1} [{row['accent'].capitalize()}]")
        print(f"    EN said : {row['transcript_en']}")
        print(f"    HI heard: {row['transcript_hi']}")
        print(f"    Gloss   : {gloss}")
        print(f"    Conf    : {row['confidence_hi']:.4f}"
              + (f"   P(Hindi)={p_hindi:.4f}" if p_hindi is not None else "")
              + "\n")
        case_data.append({
            "case_number": i + 1,
            "utt_id": row["utt_id"],
            "accent": row["accent"],
            "english_transcript": str(row["transcript_en"]),
            "hindi_output": str(row["transcript_hi"]),
            "hindi_gloss_nllb": gloss,
            "hindi_confidence": float(row["confidence_hi"]),
            "p_hindi_langid": p_hindi,
        })

    with open(os.path.join(OUTPUT_DIR, "case_studies.json"), "w",
              encoding="utf-8") as f:
        json.dump(case_data, f, indent=2, ensure_ascii=False)
    print("      Case studies saved to case_studies.json")
    print("  ℹ  PAPER NOTE: present the raw Devanagari alongside the NLLB")
    print("     gloss. If the gloss is broken English, that is evidence the")
    print("     ASR output is not genuine Hindi — report it honestly rather")
    print("     than substituting a fluent online-MT paraphrase.")



###################################### STEP 8 — LIVE DEMO ######################################


def step8_live_demo(hindi_model, hindi_processor, device):
    section("STEP 8 — Live Cross-Lingual Demo (frozen Stage-1 model)")

    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        print("  ✘  Install: pip install sounddevice soundfile")
        return

    # load frozen Stage-1 artefacts
    if not (os.path.exists(STAGE1_MODEL) and os.path.exists(STAGE1_SCALER)):
        print(f"  ✘  Frozen Stage-1 model not found in {STAGE1_DIR}. "
              "Run Stage 1 v2 first.")
        return
    clf = joblib.load(STAGE1_MODEL)
    scaler = joblib.load(STAGE1_SCALER)

    base_processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL)
    base_model = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL)
    base_model.eval()
    base_model.to(device)
    le = LabelEncoder()
    le.fit(ACCENTS)

    # Microphone diagnostics
    try:
        in_idx = sd.default.device[0]
        in_dev = sd.query_devices(in_idx)
        print(f"  Input device: [{in_idx}] {in_dev['name']} "
              f"({in_dev['max_input_channels']} input ch)")
        if in_dev["max_input_channels"] < 1:
            print("  ✘  Default device has no input channels. Available:")
            print(sd.query_devices())
            print("  Set one with: sd.default.device = (<input_idx>, None)")
            return
    except Exception as e:
        print(f"     Could not query audio devices: {e}")

    duration = 4.0
    print(f"\n  Recording {duration}s — speak now!")
    audio = sd.rec(int(duration * TARGET_SR), samplerate=TARGET_SR,
                   channels=1, dtype="float32")
    sd.wait()
    y_full = audio.flatten()

    raw_rms = float(np.sqrt(np.mean(y_full ** 2))) if len(y_full) else 0.0
    peak = float(np.max(np.abs(y_full))) if len(y_full) else 0.0
    print(f"  Captured: RMS={raw_rms:.6f}  peak={peak:.4f}")
    if raw_rms < 1e-5:
        print("  ✘  Recorded audio is SILENT (all zeros). Likely causes:")
        print("     1) macOS mic permission: System Settings  -> Privacy &")
        print("        Security  -> Microphone  -> enable for Terminal/iTerm/your")
        print("        IDE, then RESTART that app. (macOS silently records")
        print("        zeros when permission is denied — no error is raised.)")
        print("     2) Wrong input device selected — list devices with:")
        print("        python -c \"import sounddevice as sd; "
              "print(sd.query_devices())\"")
        return

    y_raw, _ = librosa.effects.trim(y_full, top_db=TRIM_TOP_DB)
    if len(y_raw) < 0.5 * TARGET_SR:
        print("     <0.5s of speech left after silence trimming — speak "
              "louder/closer to the mic and retry.")
        return
    print(f"      Recorded {len(y_raw)/TARGET_SR:.1f}s of speech")

    # RMS-normalised copy for embedding/ASR, raw copy for lang-ID 
    rms = np.sqrt(np.mean(y_raw ** 2))
    y = y_raw / rms * TARGET_RMS if rms > 0 else y_raw

    # 1. Accent classification (frozen model)
    inputs = base_processor(y, sampling_rate=TARGET_SR, return_tensors="pt",
                            padding=True).input_values.to(device)
    with torch.no_grad():
        hidden = base_model(inputs).last_hidden_state
    emb = hidden.mean(dim=1).squeeze().cpu().numpy().reshape(1, -1)
    proba = clf.predict_proba(scaler.transform(emb))[0]
    accent = le.inverse_transform([int(np.argmax(proba))])[0]
    print(f"\n  Predicted accent: {accent.capitalize()}")
    for i, acc in enumerate(ACCENTS):
        print(f"    {acc.capitalize():12s}: {proba[i]*100:.1f}%")

    # 2. Whisper English ASR
    en_text = "(whisper not installed)"
    try:
        import whisper
        tmp = os.path.join(OUTPUT_DIR, "_live_temp.wav")
        sf.write(tmp, y, TARGET_SR)
        wmodel = whisper.load_model("base")
        en_text = wmodel.transcribe(tmp, language="en")["text"].strip()
        os.remove(tmp)
    except ImportError:
        pass
    print(f"\n  English transcript: {en_text}")

    # 3. Hindi ASR (blank-masked confidence, optional KenLM)
    hi_text, hi_conf = decode_hindi(hindi_model, hindi_processor, device, y)
    gloss = translate_hi_to_en(hi_text)                            # [FIX-5]
    print(f"  Hindi ASR output : {hi_text}")
    print(f"  NLLB gloss       : {gloss}")
    print(f"  Confidence       : {hi_conf:.4f}")

    # 4. Lang-ID (fixed path) 
    pred_lang, p_hi, p_en = None, None, None
    try:
        lang_clf = _load_lang_id()
        pred_lang, p_hi, p_en = _langid_probs(lang_clf, y_raw)
        print(f"  Lang-ID: {pred_lang}  P(Hindi)={p_hi:.4f}  "
              f"P(English)={p_en:.4f}")
        if p_en is not None and p_hi is not None:
            gate = "ENGLISH route" if p_en > p_hi else "HINDI route"
            print(f"  Deployment gate (P(en) vs P(hi)):  -> {gate}")
    except Exception as e:
        print(f"  Lang-ID unavailable: {e}")

    # 5. Boxed summary panel
    _print_live_summary(accent, proba, en_text, hi_text, gloss, hi_conf,
                        pred_lang, p_hi, p_en)


############################### Live-demo summary ###############################

_BOX_W = 74   # inner width of the box


def _wrap(text, width):
    
    import textwrap
    text = str(text) if text is not None else ""
    lines = textwrap.wrap(text, width=width) or [""]
    return lines


def _row(text=""):
    # One box row, padded to the inner width
    print(f"  ║ {str(text):<{_BOX_W}} ║")


def _rows(text, indent=0):
    # Multiple wrapped box rows
    pad = " " * indent
    for line in _wrap(text, _BOX_W - indent):
        _row(pad + line)


def _sep():
    print("  ╠" + "═" * (_BOX_W + 2) + "╣")


def _print_live_summary(accent, proba, en_text, hi_text, gloss, hi_conf,
                        pred_lang, p_hi, p_en):
    # Detailed boxed pipeline summary for the live demo
    has_hindi = bool(str(hi_text).strip()
                     and str(hi_text).strip().lower() != "nan")

    # Confidence-based risk band ,calibrated against native Hindi decoding:

    if not has_hindi:
        risk = "NONE"
    elif hi_conf > -0.1:
        risk = "HIGH"
    elif hi_conf > -0.3:
        risk = "MODERATE"
    else:
        risk = "LOW"

    # Check : Does the lang-ID gate actually catch this?
    if p_hi is not None and p_en is not None:
        gate_catches = p_en > p_hi
        gate_verdict = ("BLOCKED — routed to the English ASR"
                        if gate_catches
                        else "NOT BLOCKED — routed to the Hindi ASR")
    else:
        gate_catches = None
        gate_verdict = "unavailable (lang-ID did not run)"

    print("\n  ╔" + "═" * (_BOX_W + 2) + "╗")
    _row(f"{'LIVE DEMO — FULL PIPELINE SUMMARY':^{_BOX_W}}")
    _sep()

    # 1. Accent
    _row("1. SPEAKER ACCENT DETECTION  (frozen Stage-1 RF)")
    _row(f"     Predicted accent : {accent.capitalize()}")
    for i, acc in enumerate(ACCENTS):
        bar = "█" * int(round(proba[i] * 30))
        _row(f"     {acc.capitalize():<10s}: {proba[i]*100:5.1f}%  {bar}")
    _sep()

    # 2. English ASR
    _row("2. ENGLISH ASR  (Whisper)")
    _rows(f"What you said    : {en_text}", indent=5)
    _sep()

    # 3. Hindi ASR
    _row(f"3. HINDI ASR  (cross-lingual decode: {HINDI_ASR_MODEL})")
    _rows(f"Hindi output     : {hi_text}", indent=5)
    _rows(f"NLLB gloss       : {gloss}", indent=5)
    _row(f"     Confidence       : {hi_conf:.4f}  "
         f"(blank-masked mean log-prob)")
    _sep()

    # 4. Lang-ID gate
    _row("4. LANGUAGE IDENTIFICATION GATE  (VoxLingua107)")
    _row(f"     Argmax over 107  : {pred_lang}")
    if p_hi is not None:
        _row(f"     P(English)       : {p_en:.4f}")
        _row(f"     P(Hindi)         : {p_hi:.4f}")
        _row(f"     Routing rule     : P(en) > P(hi)  -> English ASR")
        _row(f"     Gate decision    : {gate_verdict}")
    else:
        _row(f"     Gate decision    : {gate_verdict}")
    _sep()

    # 5. Cross-lingual risk
    _row("5. CROSS-LINGUAL RISK ASSESSMENT")
    _row(f"     Acoustic risk    : {risk}")
    _row("")
    if has_hindi:
        _rows(f"The {accent.capitalize()}-accented English phrase:")
        _rows(f'"{en_text}"', indent=5)
        _rows("was decoded by the Hindi ASR as:")
        _rows(f'"{hi_text}"', indent=5)
        if gloss and not gloss.startswith("("):
            _rows("which glosses to English as:")
            _rows(f'"{gloss}"', indent=5)
        _row("")
        _rows("Interpretation: accented English is acoustically mapped onto "
              "Devanagari output by a Hindi ASR that never sees a language "
              "check. This is the failure mode the gate exists to prevent.")
        _row("")
        if gate_catches is True:
            _rows("However, the lang-ID gate DID catch this utterance and "
                  "would route it to the English ASR. The end-to-end risk is "
                  "therefore mitigated here — the danger is confined to "
                  "pipelines with no gate, or to speakers the gate misroutes.")
        elif gate_catches is False:
            _rows("CRITICAL: the lang-ID gate did NOT catch this utterance — "
                  "it would be routed to the Hindi ASR, so the "
                  "misinterpretation above would reach the user end-to-end.")
    else:
        _rows("No Hindi output was produced — the ASR emitted only CTC "
              "blanks. No cross-lingual misinterpretation for this utterance.")
    print("  ╚" + "═" * (_BOX_W + 2) + "╝")

    # Corpus-level context (from the batch run), so a single live utterance is never over-generalised
    print("\n  Corpus context (Stage 2 batch run, n=2400):")
    print("    Clips misrouted to the Hindi ASR under the P(hi)>P(en) rule:")
    print("      American 0.64%   British 1.01%   Indian 18.63%")
    print("     -> A single live utterance is illustrative, not evidence. The")
    print("      accent-conditioned disparity above is the reportable result.")


###################################### STEP 9 - SUMMARY ######################################



def step9_summary(hindi_metrics, risk_out, centroid_sims, mining_out,
                  lang_id_df, xling_df):
    section("STEP 9 — Summary & Mitigation")

    # Mitigation 1: lang-ID gating (using P(en) vs P(hi) routing)
    if lang_id_df is not None and not xling_df.empty:
        print("\n  MITIGATION 1 — Lang-ID gating (route to Hindi ASR only if")
        print("  P(hi) > P(en)):")
        merged = xling_df.merge(
            lang_id_df[["utt_id", "p_hindi", "p_english"]],
            on="utt_id", how="left")
        for acc in ACCENTS:
            sub = merged[merged["accent"] == acc].dropna(
                subset=["p_hindi", "p_english"])
            if sub.empty:
                continue
            routed_hi = (sub["p_hindi"] > sub["p_english"]).mean() * 100
            print(f"    {acc.capitalize():10s}: {routed_hi:.2f}% of clips "
                  f"would still reach the Hindi ASR")

    # Mitigation 2: confidence thresholding
    if not xling_df.empty and risk_out.get("plausibility_threshold"):
        th = risk_out["plausibility_threshold"]
        finite = xling_df[np.isfinite(xling_df["confidence_hi"])]
        rejected = (finite["confidence_hi"] < th).mean() * 100
        print(f"\n  MITIGATION 2 — Native-calibrated confidence threshold "
              f"({th:.4f})")
        print(f"    rejects {rejected:.1f}% of cross-lingual Hindi outputs")

    def to_native(obj):
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_native(v) for v in obj]
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    summary = to_native({
        "version": "stage2_v2",
        "environment": library_versions(),                         
        "hindi_asr": {"model": HINDI_ASR_MODEL,
                      "pretrained_asis": USE_PRETRAINED_ASR,
                      "kenlm": KENLM_PATH,
                      "metrics": hindi_metrics},
        "entropy_thresholds_swept": ENTROPY_THRESHOLDS,
        "primary_entropy_threshold": PRIMARY_ENTROPY,
        "mining": mining_out,
        "risk": risk_out,
        "centroid_cosine_similarities": centroid_sims or {},
        "n_english_utterances_decoded": int(len(xling_df))
        if xling_df is not None else 0,
    })
    with open(os.path.join(OUTPUT_DIR, "stage2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n      Summary saved to stage2_summary.json")



###################################### MAIN ######################################


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 v2 — Cross-Lingual Analysis")
    parser.add_argument("--live", action="store_true",
                        help="Record from microphone and run full pipeline")
    args = parser.parse_args()

    print("\n" + "  *" * 72)
    print("  STAGE 2 (v2) — HINDI ASR & CROSS-LINGUAL MISINTERPRETATION")
    print("  Pretrained Indic ASR | Calibrated CLRS | Null-baselined | Tested")
    print("  *" * 72)

    device = get_device()
    print(f"\n  Device: {device}")

    if not os.path.exists(os.path.join(STAGE1_DIR, "embeddings_cache.npz")):
        print("  ✘  Stage 1 v2 outputs not found. Run stage1_pipeline_v2.py "
              "first.")
        sys.exit(1)

    if args.live:
        # load the ASR only — no Hindi data loading, no test-set evaluation
        src = (HINDI_ASR_SAVE if (not USE_PRETRAINED_ASR
                                  and os.path.exists(HINDI_ASR_SAVE))
               else HINDI_ASR_MODEL)
        print(f"\n  Loading Hindi ASR for live demo: {src}")
        hindi_model = Wav2Vec2ForCTC.from_pretrained(src).to(device)
        hindi_processor = Wav2Vec2Processor.from_pretrained(src)
        step8_live_demo(hindi_model, hindi_processor, device)
        return

    # Step 1
    hindi_model, hindi_processor, device, hindi_metrics, \
        (test_paths_hi, test_texts_hi) = step1_hindi_asr()

    # Step 2
    df = load_stage1_metadata()
    if ENGLISH_MAX_PER_ACCENT:
        df = df.groupby("accent").head(ENGLISH_MAX_PER_ACCENT) \
            .reset_index(drop=True)
    lang_id_df = step2_lang_id_gate(df)

    # Step 3
    xling_df = step3_cross_lingual_decode(
        df, hindi_model, hindi_processor, device, lang_id_df)

    # Step 4
    cp_df, mining_out = step4_confusion_mining(xling_df)

    # Step 5
    risk_out = step5_risk_scores(xling_df, cp_df, hindi_model,
                                 hindi_processor, device)



    # Step 6
    centroid_sims = None
    if test_paths_hi:
        centroid_sims = step6_joint_umap(test_paths_hi, device)
    else:
        print("\n     Skipping joint UMAP — no Hindi test data.")

    # Step 7
    step7_case_studies(xling_df, lang_id_df)

    # Step 9
    step9_summary(hindi_metrics, risk_out, centroid_sims, mining_out,
                  lang_id_df, xling_df)

    print("\n" + "  *" * 72)
    print(f"  STAGE 2 (v2) COMPLETE — outputs in: {OUTPUT_DIR}")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    {fname}")
    print("  *" * 72 + "\n")


if __name__ == "__main__":
    main()
