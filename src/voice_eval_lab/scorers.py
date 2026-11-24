"""Detector module: ridge-regression scorers trained on 12-rater labels.

Two scorers share the feature extractor:
 - utterance scorer: predicts mean human quality (1-5)
 - WER scorer:       predicts word error rate (0-1)
Stdlib only: normal equations solved by Gaussian elimination.
"""
from __future__ import annotations
from .data import Utterance, Checkpoint, DIALECT_DIFFICULTY, NOISE_PENALTY, human_ratings, simulate_hypothesis
from .wer import wer as wer_fn

def features(utt: Utterance, ckpt: Checkpoint) -> list[float]:
    return [1.0, ckpt.quality, ckpt.base_wer * 5, utt.snr_db / 30.0,
            DIALECT_DIFFICULTY[utt.dialect] * 5, NOISE_PENALTY[utt.noise] * 5,
            len(utt.text.split()) / 20.0, utt.duration_s / 10.0]

def _solve(A, b):
    n = len(b); M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c])); M[c], M[p] = M[p], M[c]
        piv = M[c][c] or 1e-12
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c] / piv
                for k in range(c, n + 1): M[r][k] -= f * M[c][k]
    return [M[i][n] / (M[i][i] or 1e-12) for i in range(n)]

class RidgeScorer:
    def __init__(self, weights=None, lam=1.0): self.weights = weights or []; self.lam = lam
    def fit(self, X, y):
        d = len(X[0]); A = [[0.0]*d for _ in range(d)]; b = [0.0]*d
        for xi, yi in zip(X, y):
            for i in range(d):
                b[i] += xi[i]*yi
                for j in range(d): A[i][j] += xi[i]*xi[j]
        for i in range(d): A[i][i] += self.lam
        self.weights = _solve(A, b); return self
    def predict(self, x): return sum(w*v for w, v in zip(self.weights, x))
    def to_dict(self): return {"weights": self.weights, "lambda": self.lam}
    @classmethod
    def from_dict(cls, d): s = cls(d["weights"], d.get("lambda", 1.0)); return s

def train_scorers(utterances, checkpoints, seed=42):
    """Train on first checkpoint pairing per utterance (deterministic)."""
    X, y_q, y_w = [], [], []
    for i, u in enumerate(utterances):
        c = checkpoints[i % len(checkpoints)]
        ratings = human_ratings(u, c, seed=seed)
        X.append(features(u, c)); y_q.append(sum(ratings)/len(ratings))
        y_w.append(wer_fn(u.text, simulate_hypothesis(u, c, seed=seed)))
    return {"utterance": RidgeScorer().fit(X, y_q), "wer": RidgeScorer().fit(X, y_w),
            "n_train": len(X), "raters_per_utterance": 12}
