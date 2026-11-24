"""Reproducible benchmark: evaluate every checkpoint, write JSON scorecard."""
from __future__ import annotations
import json
from .data import human_ratings, simulate_hypothesis
from .wer import wer as wer_fn
from .scorers import train_scorers, features
from .calibration import calibration_report
from .adversarial import generate_adversarial

def run_benchmark(utterances, checkpoints, seed=42, adversarial_n=60):
    trained = train_scorers(utterances, checkpoints, seed=seed)
    u_scorer, w_scorer = trained["utterance"], trained["wer"]
    per_ckpt, calib_rows = {}, []
    for c in checkpoints:
        wers, quals = [], []
        for u in utterances:
            hyp = simulate_hypothesis(u, c, seed=seed)
            w = wer_fn(u.text, hyp); wers.append(w)
            ratings = human_ratings(u, c, seed=seed); human = sum(ratings)/len(ratings)
            auto = u_scorer.predict(features(u, c)); quals.append(auto)
            calib_rows.append({"human": human, "auto": auto, "dialect": u.dialect,
                               "noise": u.noise, "checkpoint": c.id, "wer": w,
                               "wer_pred": w_scorer.predict(features(u, c))})
        per_ckpt[c.id] = {"kind": c.kind, "mean_wer": round(sum(wers)/len(wers), 4),
                          "mean_auto_quality": round(sum(quals)/len(quals), 4),
                          "latency_ms": c.latency_ms, "n": len(wers)}
    # adversarial regression: WER delta on hardest checkpoint vs best
    adv = generate_adversarial(utterances[:adversarial_n])
    best = min(checkpoints, key=lambda c: c.base_wer)
    clean_w = sum(wer_fn(u.text, simulate_hypothesis(u, best, seed)) for u in utterances[:adversarial_n]) / max(1, adversarial_n)
    adv_w = sum(wer_fn(u.text, simulate_hypothesis(u, best, seed)) for u in adv) / max(1, len(adv))
    calib = calibration_report(calib_rows)
    return {"seed": seed, "n_utterances": len(utterances), "n_checkpoints": len(checkpoints),
            "raters_per_utterance": 12, "checkpoints": per_ckpt, "calibration": calib,
            "adversarial": {"n_variants": len(adv), "clean_mean_wer": round(clean_w, 4),
                            "adversarial_mean_wer": round(adv_w, 4),
                            "wer_delta": round(adv_w - clean_w, 4)},
            "scorers": {"n_train": trained["n_train"],
                        "utterance_weights": [round(x, 4) for x in u_scorer.weights],
                        "wer_weights": [round(x, 4) for x in w_scorer.weights]}}

def write_scorecard(scorecard, path):
    with open(path, "w", encoding="utf-8") as f: json.dump(scorecard, f, indent=2)
    return path
