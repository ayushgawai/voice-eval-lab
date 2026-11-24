#!/usr/bin/env python3
"""Reproducible benchmark script: writes JSON scorecard (default scaled-down).

Usage:
  python scripts/benchmark.py [--n 2400] [--seed 42] [--out scorecard.json]
Full spec is 2,400 utterances x 8 checkpoints; default here is 240 for speed.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voice_eval_lab.data import generate_utterances, default_checkpoints
from voice_eval_lab.benchmark import run_benchmark, write_scorecard

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", default="scorecard.json")
    a = ap.parse_args()
    card = run_benchmark(generate_utterances(a.n, a.seed), default_checkpoints(), seed=a.seed)
    write_scorecard(card, a.out)
    print(f"wrote {a.out}: n={card['n_utterances']} ckpts={card['n_checkpoints']} "
          f"pearson_r={card['calibration']['pearson_r']} adv_delta={card['adversarial']['wer_delta']}")
