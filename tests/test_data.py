from voice_eval_lab.data import generate_utterances, default_checkpoints, human_ratings
def test_counts():
    assert len(default_checkpoints()) == 8
    assert len(generate_utterances(2400)) == 2400
def test_deterministic():
    a = generate_utterances(10, 42); b = generate_utterances(10, 42)
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
def test_raters_12():
    u = generate_utterances(1)[0]; c = default_checkpoints()[0]
    r = human_ratings(u, c); assert len(r) == 12 and all(1 <= x <= 5 for x in r)
