# Linguistic and Accent Fairness in Spoken Language Pipelines

An end-to-end audit of an English–Hindi speech pipeline across American, British,
and Indian English. The work runs in two stages: Stage 1 characterises the accent
signal on its own, and Stage 2 follows the same frozen artifacts through the
routing gate and the Hindi recogniser.

## Stage 1 — `stage1.py`

Characterises the English accent signal and diagnoses recognition errors.

- **Preprocessing & data** — `load_metadata`, `preprocess_audio`, `crop_fixed_start`, `rows_with_ref`
- **Statistics helpers** — `bootstrap_metric`, `bootstrap_accuracy`, `bootstrap_wer`, `permutation_test_gap`, `holm_bonferroni`
- **Step 1 – EDA** — `step1_eda` (per-accent duration distributions)
- **Step 2 – Leakage/confound audit** — `step2_leakage_audit`
- **Step 3 – Embeddings** — `step3_extract_embeddings` (frozen wav2vec 2.0)
- **Step 4 – Accent classification** — `step4_classify`, `_make_split`, `_build_models`
- **Step 5 – Length sensitivity** — `step5_length_sensitivity` (truncation sweep)
- **Step 6 – Embedding geometry** — `step6_umap`
- **Step 7 – Whisper WER** — `step7_whisper_decode`, `step7_compute_wer`
- **Step 8 – Anomaly check** — `step8_anomaly`
- **Step 9 – Vocab leakage** — `step9_vocab_leakage`
- **Step 10 – Dashboard** — `step10_dashboard`
- **Live demo** — `live_record_and_classify`
- **Entry point** — `main`

## Stage 2 — `stage2.py`

Reuses Stage 1's frozen artifacts and audits the routing gate and Hindi recogniser.

- **Loaders & helpers** — `load_stage1_metadata`, `load_stage1_embeddings`, `load_hindi_common_voice`, `translate_hi_to_en`, `entropy`, `get_device`
- **Statistics helpers** — `bootstrap_proportion`, `bootstrap_wer_pairs`, `permutation_test_gap`, `holm_bonferroni`
- **Step 1 – Hindi ASR** — `step1_hindi_asr`, `evaluate_hindi_asr`, `decode_hindi`
- **Step 2 – LID routing gate** — `step2_lang_id_gate`
- **Step 3 – Cross-lingual decode** — `step3_cross_lingual_decode`
- **Step 4 – Confusion mining** — `step4_confusion_mining`
- **Step 5 – Risk scoring (CLRS)** — `step5_risk_scores`, `_plausibility_threshold`, `_noise_null_baseline`
- **Step 6 – Joint embedding geometry** — `step6_joint_umap`
- **Step 7 – Case studies** — `step7_case_studies`
- **Step 8 – Live demo** — `step8_live_demo`
- **Step 9 – Summary & mitigation** — `step9_summary`
- **Entry point** — `main`

## `main_run.py`

The single entry point that runs the whole pipeline. It imports both stages and
executes them in order (Stage 1, then Stage 2), so Stage 2 picks up the frozen
artifacts Stage 1 writes to disk. Run this to reproduce all results end to end.

## `stage2_transliteration_breakdown.py`

A standalone post-processing script for the transliteration-vs-drift analysis. It
reads the Hindi outputs Stage 2 already saved, glosses each back to English with
Stage 2's NLLB translator, and classifies every clip as transliteration, semantic
drift, or empty. It reports the breakdown corpus-wide and for the misrouted-to-Hindi
clips, at swept similarity thresholds. Stage 2 does not need to be re-run.

## Running

Both files must sit in the same folder so imports resolve.

```bash
# Full pipeline — generates all results
python main_run.py

# Live demo — records from the mic and runs it through the pipeline
python main_run.py --live
```

The stage scripts can also be run on their own (`python stage1.py`,
`python stage2.py`), each accepting the same `--live` flag. Run the breakdown
analysis after Stage 2 has produced its outputs:

```bash
python stage2_transliteration_breakdown.py
```


## Code attribution

The pipeline design, experimental methodology and all analysis code in this repository are my own work. This includes the two-stage pipeline, the
leakage/confound audit, the accent-classification and truncation-fairness analyses, the blank-masked CTC confidence, the entropy-based confusion mining, the native-calibrated and null-baselined cross-lingual risk score (CLRS), the routing-rule comparison and the transliteration-vs-drift breakdown.

No algorithmic code was copied from another person's codebase. The pretrained models and third-party libraries used as components — wav2vec 2.0, Whisper, the VoxLingua107 language-ID model, AI4Bharat indicwav2vec-hindi and NLLB, together with PyTorch,
Hugging Face Transformers, scikit-learn, librosa, jiwer, umap-learn, SciPy, NumPy, pandas, Matplotlib, seaborn and joblib — are the work of their respective authors and are imported . Full details and licences are listed in the dissertation appendices (Software Used, Pretrained Models).

Standard formulas (WER, cosine similarity, Shannon entropy, RMS energy) are established methods, their application to router-level accent fairness is the contribution of this work. Claude code was used as a development assistant mainly for optimisation and debugging.

