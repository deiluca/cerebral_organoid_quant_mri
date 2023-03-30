""" Diffusion weighted MRI: calculate metrics for global cyst seg"""

import pandas as pd
from statsmodels.stats.multitest import multipletests
from scripts.utils.io_utils import get_orig_imgs, get_masks
from scripts.utils.constants import DWMRI_SEQUENCES, CSV_GLOBAL_CYST_ANNOT
from scripts.utils.metrics import calculate_roc_auc, calculate_p_val_test
from scripts.utils.global_cyst_classification import get_org_mean_and_compactness


def get_metrics_global_cyst_seg_dw_mri():
    """Compute ROC AUC and adjusted p-values for the separation of high- and low-quality organoids based on mean organoid intensity.

    Returns:
        pd.DataFrame: one DTI sequence per row; in each row: ROC AUC and p-value
        list: list of pandas dataframes. One dataframe per DTI sequence; each dataframe contains the mean organoid intensities for each organoid
    """
    dw_mri_gt = get_masks(kind='gt_org_loc_dwmri')
    roc_aucs, p_vals = [], []
    dfs = {}
    for i in range(22):
        imgs_seq = get_orig_imgs(kind='dwmri', seq=i)
        df = get_org_mean_and_compactness(imgs_seq,
                                          org_loc=dw_mri_gt,
                                          otsu=None,
                                          only_org_mean=True,
                                          mult_255=False)
        quality_annot = pd.read_csv(CSV_GLOBAL_CYST_ANNOT)
        df = df.merge(quality_annot)
        dfs[i] = df
        roc_auc_moi = calculate_roc_auc(df, col='mean_organoid_intensity')
        roc_aucs.append(roc_auc_moi)
        p_val_moi = calculate_p_val_test(df, col='mean_organoid_intensity')
        p_vals.append(p_val_moi)
    roc_auc_col = 'ROC AUC'
    p_val_col = 'P-value'
    df_metrics = pd.DataFrame.from_dict(
        {'DW-MRI sequence': DWMRI_SEQUENCES, roc_auc_col: roc_aucs,
         p_val_col: p_vals},).sort_values(
        roc_auc_col, ascending=False)
    p_vals = df_metrics[p_val_col].tolist()
    p_vals = [float(x) for x in p_vals]
    _, p_vals_corr, _, _ = multipletests(p_vals, method='holm-sidak')
    df_metrics['P-value corr'] = p_vals_corr
    return df_metrics, dfs
