import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

def agreement_table(y_true, y_pred1, y_pred2):
    """
    Returns counts: (both_correct, only_1_correct, only_2_correct, both_wrong)
    """
    c1 = (y_true == y_pred1)
    c2 = (y_true == y_pred2)
    
    both_correct = np.sum(c1 & c2)
    only_1 = np.sum(c1 & ~c2)
    only_2 = np.sum(~c1 & c2)
    both_wrong = np.sum(~c1 & ~c2)
    
    return int(both_correct), int(only_1), int(only_2), int(both_wrong)

def mcnemar_test(only_1_correct, only_2_correct):
    """
    Computes exact binomial McNemar test when discordant count is small.
    Returns p-value and a plain-language interpretation.
    """
    b = only_1_correct
    c = only_2_correct
    n = b + c
    if n == 0:
        p_val = 1.0
    else:
        # Exact binomial test for p=0.5
        res = binomtest(min(b, c), n, 0.5, alternative='two-sided')
        p_val = res.pvalue
        
    if p_val < 0.05:
        interp = f"The difference in accuracy is statistically significant (p = {p_val:.4f})."
    else:
        interp = f"The difference in accuracy is not statistically significant (p = {p_val:.4f})."
        
    return p_val, interp

def bootstrap_confidence_intervals(y_true, y_pred1, y_pred2, n_resamples=1000, seed=42):
    """
    Computes 95% bootstrap confidence intervals for accuracy and macro-F1,
    and for the difference in accuracy.
    """
    np.random.seed(seed)
    n = len(y_true)
    
    acc1_samples = []
    f1_1_samples = []
    acc2_samples = []
    f1_2_samples = []
    diff_samples = []
    
    for _ in range(n_resamples):
        indices = np.random.randint(0, n, n)
        y_t = y_true[indices]
        y_p1 = y_pred1[indices]
        y_p2 = y_pred2[indices]
        
        acc1 = accuracy_score(y_t, y_p1)
        f1_1 = f1_score(y_t, y_p1, average='macro', zero_division=0)
        acc2 = accuracy_score(y_t, y_p2)
        f1_2 = f1_score(y_t, y_p2, average='macro', zero_division=0)
        
        acc1_samples.append(acc1)
        f1_1_samples.append(f1_1)
        acc2_samples.append(acc2)
        f1_2_samples.append(f1_2)
        diff_samples.append(acc1 - acc2)
        
    return {
        'model1_acc': (np.percentile(acc1_samples, 2.5), np.percentile(acc1_samples, 97.5)),
        'model1_f1': (np.percentile(f1_1_samples, 2.5), np.percentile(f1_1_samples, 97.5)),
        'model2_acc': (np.percentile(acc2_samples, 2.5), np.percentile(acc2_samples, 97.5)),
        'model2_f1': (np.percentile(f1_2_samples, 2.5), np.percentile(f1_2_samples, 97.5)),
        'acc_diff': (np.percentile(diff_samples, 2.5), np.percentile(diff_samples, 97.5))
    }

def group_by_crop_size(y_true, y_pred1, y_pred2, shorter_sides):
    """
    Splits into small, medium, large by terciles of shorter side.
    Returns accuracies per group for both models.
    """
    p33 = np.percentile(shorter_sides, 33.33)
    p66 = np.percentile(shorter_sides, 66.67)
    
    groups = {
        'small': shorter_sides <= p33,
        'medium': (shorter_sides > p33) & (shorter_sides <= p66),
        'large': shorter_sides > p66
    }
    
    results = {}
    for name, mask in groups.items():
        if np.sum(mask) == 0:
            results[name] = {'acc1': 0.0, 'acc2': 0.0, 'count': 0}
            continue
        acc1 = accuracy_score(y_true[mask], y_pred1[mask])
        acc2 = accuracy_score(y_true[mask], y_pred2[mask])
        results[name] = {'acc1': acc1, 'acc2': acc2, 'count': int(np.sum(mask))}
        
    return results
