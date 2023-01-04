import numpy as np
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

def dice(pred, true, k = 1):
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice

def get_dice(pred_npy_loc, gt_npy_loc, dw_mri=False):
    pred = np.load(pred_npy_loc)
    gt = np.load(gt_npy_loc)
    assert gt.shape == pred.shape
    if dw_mri:
        pred = np.moveaxis(pred, 0, -1)
    return dice(pred, gt)

def calculate_roc_auc(df, col='org_mean'):
    labels = [1 if x=='High quality' else 0 for x in df['Organoid quality'].tolist()]
    values = df[col]
    roc_auc = roc_auc_score(labels, values)
    roc_auc = max(1 - roc_auc, roc_auc)

    return roc_auc

def calculate_p_val_test(df, col='org_mean'):
    values_hq = df[df['Organoid quality']=='High quality'][col]
    values_lq = df[df['Organoid quality']=='Low quality'][col]
    return ttest_ind(values_hq, values_lq)[1]

def calculate_pr_auc(df, col='org_mean'):
    labels = [1 if x=='High quality' else 0 for x in df['Organoid quality'].tolist()]
    values = df[col]
    precision, recall, thresholds = precision_recall_curve(labels, values)
    # Use AUC function to calculate the area under the curve of precision recall curve
    pr_auc = auc(recall, precision)
    pr_auc = max(1 - pr_auc, pr_auc)

    return pr_auc