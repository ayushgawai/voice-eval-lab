from voice_eval_lab.data import generate_utterances, default_checkpoints
from voice_eval_lab.scorers import train_scorers, features
def test_train_and_predict():
    t = train_scorers(generate_utterances(120), default_checkpoints())
    assert t["n_train"] == 120 and t["raters_per_utterance"] == 12
    u = generate_utterances(1)[0]; c = default_checkpoints()[-1]
    assert 0 <= t["wer"].predict(features(u, c)) < 1.5
    assert 0 < t["utterance"].predict(features(u, c)) < 6
