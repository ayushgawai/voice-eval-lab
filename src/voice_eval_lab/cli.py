"""CLI: ingest / generate / train / calibrate / adversarial / benchmark / eval."""
from __future__ import annotations
import argparse, json, sys
from .data import generate_utterances, default_checkpoints, human_ratings, simulate_hypothesis
from .ingest import load_checkpoints, load_utterances, ingest_summary
from .scorers import train_scorers, features
from .calibration import calibration_report
from .adversarial import generate_adversarial
from .benchmark import run_benchmark, write_scorecard
from .wer import wer as wer_fn

def main(argv=None):
    p = argparse.ArgumentParser(prog="voice-eval", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("ingest", "generate", "train", "calibrate", "adversarial", "benchmark", "eval"):
        sp = sub.add_parser(name)
        sp.add_argument("--n", type=int, default=2400)
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--out", default=None)
        sp.add_argument("--checkpoints", default=None)
        sp.add_argument("--utterances", default=None)
    a = p.parse_args(argv)
    ckpts = load_checkpoints(a.checkpoints)
    utts = load_utterances(a.utterances, n=a.n, seed=a.seed) if a.cmd != "generate" else generate_utterances(a.n, a.seed)

    if a.cmd == "generate":
        payload = [u.to_dict() for u in utts]
    elif a.cmd == "ingest":
        payload = ingest_summary(ckpts, utts)
    elif a.cmd == "train":
        t = train_scorers(utts, ckpts, seed=a.seed)
        payload = {"n_train": t["n_train"], "raters_per_utterance": 12,
                   "utterance_weights": [round(x, 4) for x in t["utterance"].weights],
                   "wer_weights": [round(x, 4) for x in t["wer"].weights]}
    elif a.cmd == "calibrate":
        t = train_scorers(utts, ckpts, seed=a.seed); rows = []
        for i, u in enumerate(utts):
            c = ckpts[i % len(ckpts)]; r = human_ratings(u, c, seed=a.seed)
            rows.append({"human": sum(r)/len(r), "auto": t["utterance"].predict(features(u, c)),
                         "dialect": u.dialect, "noise": u.noise, "checkpoint": c.id})
        payload = calibration_report(rows)
    elif a.cmd == "adversarial":
        adv = generate_adversarial(utts[:50]); best = min(ckpts, key=lambda c: c.base_wer)
        payload = {"base_n": 50, "variants": len(adv),
                   "adversarial_mean_wer_best_ckpt": round(sum(wer_fn(u.text, simulate_hypothesis(u, best, a.seed)) for u in adv)/len(adv), 4),
                   "examples": [u.to_dict() for u in adv[:3]]}
    else:  # benchmark / eval (eval = end-to-end benchmark)
        payload = run_benchmark(utts, ckpts, seed=a.seed)
        if a.cmd == "eval" and not a.out: a.out = "scorecard.json"

    text = json.dumps(payload, indent=2)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f: f.write(text + "\n")
        print(f"wrote {a.out}"); print(text[:1200])
    else:
        print(text)
    return 0

if __name__ == "__main__":
    sys.exit(main())
