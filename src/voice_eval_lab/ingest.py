"""Ingest manifests of checkpoints / utterances (JSON) with validation."""
from __future__ import annotations
import json
from .data import Checkpoint, Utterance, default_checkpoints, generate_utterances

def load_checkpoints(path=None):
    if not path: return default_checkpoints()
    data = json.load(open(path, encoding="utf-8"))
    ckpts = [Checkpoint(**d) for d in data]
    assert ckpts, "no checkpoints in manifest"
    return ckpts

def load_utterances(path=None, n=2400, seed=42):
    if not path: return generate_utterances(n, seed)
    data = json.load(open(path, encoding="utf-8"))
    return [Utterance(**d) for d in data]

def ingest_summary(checkpoints, utterances):
    from collections import Counter
    return {"checkpoints": len(checkpoints), "utterances": len(utterances),
            "kinds": dict(Counter(c.kind for c in checkpoints)),
            "dialects": dict(Counter(u.dialect for u in utterances)),
            "sources": dict(Counter(u.source for u in utterances)),
            "noises": dict(Counter(u.noise for u in utterances))}
