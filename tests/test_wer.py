from voice_eval_lab.wer import wer, corpus_wer
def test_wer_identical(): assert wer("hello world", "hello world") == 0.0
def test_wer_one_sub(): assert wer("hello world", "hello there") == 0.5
def test_wer_empty_ref(): assert wer("", "") == 0.0 and wer("", "x") == 1.0
def test_corpus(): assert corpus_wer([("a b", "a b"), ("a b", "a")]) == 0.25
