import numpy as np
from ml.engine.stats import agreement_table, mcnemar_test, bootstrap_confidence_intervals, group_by_crop_size

def test_agreement_table():
    y_true = np.array([0, 1, 2, 3])
    y_p1 = np.array([0, 1, 0, 0]) # correct: 0, 1. wrong: 2, 3
    y_p2 = np.array([0, 0, 2, 0]) # correct: 0, 2. wrong: 1, 3
    
    # both: 0. only 1: 1. only 2: 2. both wrong: 3.
    bc, o1, o2, bw = agreement_table(y_true, y_p1, y_p2)
    assert bc == 1
    assert o1 == 1
    assert o2 == 1
    assert bw == 1
    assert (bc + o1 + o2 + bw) == len(y_true)
    
def test_mcnemar_test():
    p, msg = mcnemar_test(10, 2)
    assert p < 0.05
    assert "statistically significant" in msg
    
    p2, msg2 = mcnemar_test(5, 5)
    assert p2 == 1.0
    assert "not statistically significant" in msg2

def test_bootstrap_intervals():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_p1 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0]) # 90% acc
    y_p2 = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 0]) # 70% acc
    
    res = bootstrap_confidence_intervals(y_true, y_p1, y_p2, n_resamples=100, seed=42)
    
    # The point estimates should ideally be within the intervals
    acc1_point = 0.9
    acc2_point = 0.7
    diff_point = 0.2
    
    assert res['model1_acc'][0] <= acc1_point <= res['model1_acc'][1]
    assert res['model2_acc'][0] <= acc2_point <= res['model2_acc'][1]
    assert res['acc_diff'][0] <= diff_point <= res['acc_diff'][1]
    
def test_group_by_crop_size():
    y_true = np.array([0, 1, 2, 3, 4, 5])
    y_p1 = np.array([0, 1, 2, 3, 4, 5]) # 100%
    y_p2 = np.array([0, 0, 2, 0, 4, 0]) # 50%
    sizes = np.array([10, 20, 30, 40, 50, 60])
    
    res = group_by_crop_size(y_true, y_p1, y_p2, sizes)
    assert 'small' in res
    assert 'medium' in res
    assert 'large' in res
    
    assert res['small']['count'] == 2
    assert res['medium']['count'] == 2
    assert res['large']['count'] == 2
    
    assert res['small']['acc1'] == 1.0
