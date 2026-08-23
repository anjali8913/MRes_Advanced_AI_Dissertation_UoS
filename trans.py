import os
import re
import json
import argparse
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

import stage2

OUTPUT_DIR = stage2.OUTPUT_DIR
ACCENTS = stage2.ACCENTS
XLING_PATH = os.path.join(OUTPUT_DIR, "cross_lingual_results.csv")
LANGID_PATH = os.path.join(OUTPUT_DIR, "lang_id_results.csv")
GLOSS_CACHE = os.path.join(OUTPUT_DIR, "gloss_cache.json")
OUT_CSV = os.path.join(OUTPUT_DIR, "transliteration_breakdown_rows.csv")
OUT_JSON = os.path.join(OUTPUT_DIR, "transliteration_breakdown_summary.json")

SIM_THRESHOLDS = [0.40, 0.50, 0.60]
PRIMARY_SIM = 0.50


def normalise(text):
    return " ".join(re.findall(r"[a-z]+", str(text).lower()))


def similarity(a, b):
    a, b = normalise(a), normalise(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def load_gloss_cache():
    if os.path.exists(GLOSS_CACHE):
        with open(GLOSS_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_gloss_cache(cache):
    with open(GLOSS_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def gloss_all(hindi_texts):
    cache = load_gloss_cache()
    uniques = sorted({str(t) for t in hindi_texts if str(t).strip()})
    todo = [t for t in uniques if t not in cache]
    print(f"  Unique non-empty Hindi outputs: {len(uniques)}  "
          f"(cached: {len(uniques) - len(todo)}, to gloss: {len(todo)})")
    for i, t in enumerate(todo):
        cache[t] = stage2.translate_hi_to_en(t)
        if (i + 1) % 25 == 0:
            print(f"    glossed {i + 1}/{len(todo)}")
            save_gloss_cache(cache)
    save_gloss_cache(cache)
    return cache


def classify(hindi_text, english_text, gloss, sim_threshold):
    if not str(hindi_text).strip():
        return "empty"
    if str(gloss).startswith("("):
        return "unglossed"
    if similarity(gloss, english_text) >= sim_threshold:
        return "transliteration"
    return "semantic_drift"


def breakdown(df, sim_threshold):
    out = {}
    for acc in ACCENTS + ["all"]:
        sub = df if acc == "all" else df[df["accent"] == acc]
        n = len(sub)
        if n == 0:
            continue
        labels = [classify(r["transcript_hi"], r["transcript_en"],
                           r["gloss"], sim_threshold)
                  for _, r in sub.iterrows()]
        counts = {k: labels.count(k) for k in
                  ["transliteration", "semantic_drift", "empty", "unglossed"]}
        non_empty = n - counts["empty"]
        share_translit_nonempty = (counts["transliteration"] / non_empty * 100
                                   if non_empty > 0 else 0.0)
        out[acc] = {
            "n": n,
            "counts": counts,
            "pct_of_all": {k: round(v / n * 100, 2)
                           for k, v in counts.items()},
            "transliteration_share_of_nonempty_pct":
                round(share_translit_nonempty, 2),
        }
    return out


def misrouted_ids(langid_df, rule):
    if rule == "argmax":
        m = langid_df["misclassified_as_hindi"].astype(bool)
    else:
        m = langid_df["p_hindi"] > langid_df["p_english"]
    return set(langid_df[m]["utt_id"].tolist())


def print_block(title, bd):
    print(f"\n  {title}")
    print(f"  {'Accent':10s} | {'n':>5s} | {'Translit':>9s} | "
          f"{'Drift':>6s} | {'Empty':>6s} | {'Translit% of non-empty':>22s}")
    print("  " + "-" * 74)
    for acc in ACCENTS + ["all"]:
        if acc not in bd:
            continue
        d = bd[acc]
        c = d["counts"]
        print(f"  {acc.capitalize():10s} | {d['n']:>5d} | "
              f"{c['transliteration']:>9d} | {c['semantic_drift']:>6d} | "
              f"{c['empty']:>6d} | "
              f"{d['transliteration_share_of_nonempty_pct']:>21.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Transliteration vs semantic-drift breakdown of Stage 2")
    parser.add_argument("--sim", type=float, default=PRIMARY_SIM)
    args = parser.parse_args()

    print("\n" + "=" * 72)
    print("  TRANSLITERATION vs SEMANTIC DRIFT — Stage 2 post-hoc breakdown")
    print("=" * 72)

    if not os.path.exists(XLING_PATH):
        raise FileNotFoundError(
            f"{XLING_PATH} not found. Run Stage 2 first.")
    xling = pd.read_csv(XLING_PATH)
    xling["transcript_hi"] = xling["transcript_hi"].fillna("")
    xling["transcript_en"] = xling["transcript_en"].fillna("")
    print(f"\n  Loaded {len(xling)} decoded clips from cross_lingual_results.csv")

    cache = gloss_all(xling["transcript_hi"].tolist())
    xling["gloss"] = [cache.get(str(t), "") if str(t).strip() else ""
                      for t in xling["transcript_hi"]]

    xling["label_primary"] = [
        classify(r["transcript_hi"], r["transcript_en"], r["gloss"], args.sim)
        for _, r in xling.iterrows()]
    xling.to_csv(OUT_CSV, index=False)

    summary = {"primary_sim_threshold": args.sim,
               "sim_thresholds_swept": SIM_THRESHOLDS,
               "corpus": {}, "misrouted": {}}

    print("\n" + "-" * 72)
    print("  CORPUS-WIDE (all decoded English clips)")
    print("-" * 72)
    for th in SIM_THRESHOLDS:
        bd = breakdown(xling, th)
        summary["corpus"][str(th)] = bd
        tag = "  [PRIMARY]" if abs(th - args.sim) < 1e-9 else ""
        print_block(f"similarity threshold = {th}{tag}", bd)

    if os.path.exists(LANGID_PATH):
        langid = pd.read_csv(LANGID_PATH)
        for rule in ["argmax", "comparative"]:
            ids = misrouted_ids(langid, rule)
            sub = xling[xling["utt_id"].isin(ids)].copy()
            print("\n" + "-" * 72)
            print(f"  MISROUTED-TO-HINDI ONLY  ({rule} rule, "
                  f"n={len(sub)} clips)")
            print("-" * 72)
            rule_out = {}
            for th in SIM_THRESHOLDS:
                bd = breakdown(sub, th)
                rule_out[str(th)] = bd
                tag = "  [PRIMARY]" if abs(th - args.sim) < 1e-9 else ""
                print_block(f"similarity threshold = {th}{tag}", bd)
            summary["misrouted"][rule] = rule_out
    else:
        print(f"\n  ⚠  {LANGID_PATH} not found — skipping misrouted breakdown.")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 72)
    print(f"  Saved per-clip labels : {OUT_CSV}")
    print(f"  Saved summary         : {OUT_JSON}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()