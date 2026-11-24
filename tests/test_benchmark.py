import json, subprocess, sys
from voice_eval_lab.data import generate_utterances, default_checkpoints
from voice_eval_lab.benchmark import run_benchmark

def test_benchmark_scorecard():
    card = run_benchmark(generate_utterances(48), default_checkpoints())
    assert card["n_checkpoints"] == 8 and card["n_utterances"] == 48
    assert -1 <= card["calibration"]["pearson_r"] <= 1
    assert card["adversarial"]["wer_delta"] >= -0.05  # adversarial should not much help

def test_end_to_end_cli(tmp_path):
    out = tmp_path / "scorecard.json"
    r = subprocess.run([sys.executable, "-m", "voice_eval_lab", "eval", "--n", "48", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    card = json.loads(out.read_text()); assert card["n_checkpoints"] == 8
