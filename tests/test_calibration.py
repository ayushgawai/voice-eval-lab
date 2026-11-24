from voice_eval_lab.calibration import pearson, calibration_report
def test_pearson():
    assert abs(pearson([1,2,3],[1,2,3]) - 1.0) < 1e-9
    assert abs(pearson([1,2,3],[3,2,1]) + 1.0) < 1e-9
def test_report():
    rows = [{"human": 4.0, "auto": 3.9, "dialect": "en-US", "noise": "clean", "checkpoint": "a"}]*5
    r = calibration_report(rows); assert r["n"] == 5 and r["agreement_within_0.5"] == 1.0
    assert "dialect" in r["slices"]
