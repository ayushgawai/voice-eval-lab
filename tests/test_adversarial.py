from voice_eval_lab.data import generate_utterances
from voice_eval_lab.adversarial import generate_adversarial, dialect_variant
def test_adversarial_expands():
    adv = generate_adversarial(generate_utterances(10))
    assert len(adv) == 40  # 1 dialect + 3 noise per base
def test_dialect_swap():
    u = generate_utterances(200)[24]  # template with football/apartment cycle check not required
    v = dialect_variant(u); assert v.dialect == "en-GB" and v.id.endswith("-dialect")
