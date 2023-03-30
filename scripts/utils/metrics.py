import numpy as np
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve


def dice(pred, true, k=1):
    """Computes Dice score.

    Args:
        pred (ndarray): prediction
        true (ndarray): ground truth
        k (int, optional): true class. Defaults to 1.

    Returns:
        _type_: _description_
    """
    intersection = np.sum(pred[true == k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice


def get_dice(pred_npy_loc, gt_npy_loc, dw_mri=False):
    """Computes Dice score using numpy file locations.

    Args:
        pred_npy_loc (str): path to prediction numpy file
        gt_npy_loc (str): path to ground truth numpy file
        dw_mri (bool, optional): For consistency between mask and ground truth. Defaults to False.

    Returns:
        float: Dice score
    """
    pred = np.load(pred_npy_loc)
    gt = np.load(gt_npy_loc)
    assert gt.shape == pred.shape
    if dw_mri:
        pred = np.moveaxis(pred, 0, -1)
    return dice(pred, gt)


def calculate_roc_auc(df, col='org_mean'):
    """Calculates ROC AUC for separation of high- and low-quality organoids.

    Args:
        df (pd.DataFrame): dataframe with columns 'Organoid quality' and "col"
        col (str, optional): column to calculate ROC AUC. Defaults to 'org_mean'.

    Returns:
        float: ROC AUC
    """
    labels = [1
              if x == 'High quality' else 0
              for x in df['Organoid quality'].tolist()]
    values = df[col]
    roc_auc = roc_auc_score(labels, values)
    roc_auc = max(1 - roc_auc, roc_auc)

    return roc_auc


def calculate_p_val_test(df, col='org_mean'):
    """Calculates ROC AUC for separation of high- and low-quality organoids.

    Args:
        df (pd.DataFrame): dataframe with columns 'Organoid quality' and "col"
        col (str, optional): column to calculate p-value. Defaults to 'org_mean'.

    Returns:
        float: t-test p-value
    """
    values_hq = df[df['Organoid quality'] == 'High quality'][col]
    values_lq = df[df['Organoid quality'] == 'Low quality'][col]
    return ttest_ind(values_hq, values_lq)[1]


def calculate_pr_auc(df, col='org_mean'):
    """Calculates PR AUC for separation of high- and low-quality organoids.

    Args:
        df (pd.DataFrame): dataframe with columns 'Organoid quality' and "col"
        col (str, optional): column to calculate PR AUC. Defaults to 'org_mean'.

    Returns:
        float: PR AUC
    """
    labels = [1
              if x == 'High quality' else 0
              for x in df['Organoid quality'].tolist()]
    values = df[col]
    precision, recall, _ = precision_recall_curve(labels, values)
    # Use AUC function to calculate the area under the curve of precision recall curve
    pr_auc = auc(recall, precision)
    pr_auc = max(1 - pr_auc, pr_auc)

    return pr_auc
