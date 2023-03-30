"""Scripts for visualization"""

from collections import Counter
import io

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from scripts.utils.constants import CSV_ORG_OVERVIEW
from scripts.utils.io_utils import get_masks, get_orig_imgs
from scripts.utils.dw_mri import get_metrics_global_cyst_seg_dw_mri
from scripts.utils.global_cyst_classification import get_compactness


def plot_compactness(df, roc_auc_compactness, save_to=''):
    """Swarmplot of compactness, x = low / high quality, y = compactness. One point per sample.

    Args:
        df (pd.DataFrame): one row per organoid, with columns 'compactness' and 'Organoid quality'
        roc_auc_compactness (float): ROC for writing it in the title
        save_to (str, optional): path to save plot. Defaults to ''.
    """
    sns.set_style('white')
    plt.figure(figsize=(4, 3), facecolor='white')
    sns.swarmplot(data=df.sort_values('Organoid quality',
                  ascending=True), y='compactness', x='Organoid quality')
    plt.ylabel('Compactness')
    plt.title(f'ROC AUC: {roc_auc_compactness}')
    plt.grid()
    sns.despine()
    plt.tight_layout()

    if save_to != '':
        plt.savefig(save_to, dpi=300)


def get_df_cyst_sizes():
    """Get dataframe comprising cyst size for every organoid.

    Returns:
        x (pd.DataFrame): dataframe comprising cyst size for every organoid
    """
    # calculate cysticity
    cyst_masks = get_masks(kind='gt_cyst_loc_all45')
    org_ids, cyst_sizes = [], []
    for k, v in cyst_masks.items():
        c = Counter(v.flatten())[1]
        org_ids.append(k)
        cyst_sizes.append(c)
    x = pd.DataFrame.from_dict({'org_id': org_ids, 'cyst_size': cyst_sizes})
    return x


def plot_correlation_compactness_cysticity(save_to):
    """Scatterplot with x = cyst size y = compactness.

    Args:
        save_to (str, optional): path to save plot. Defaults to ''.
    """
    df = get_compactness()
    x = get_df_cyst_sizes()

    df = df.merge(x)

    # plot
    plt.figure(figsize=(4, 4), facecolor='white')
    sns.scatterplot(data=df, x='cyst_size', y='compactness')
    pearson = df['cyst_size'].corr(df['compactness'], method='pearson')
    plt.title(f'Correlation: {pearson:.2f} (Pearson)')
    plt.grid()
    sns.despine()
    plt.ylabel('Compactness')
    plt.xlabel('Cyst size in voxels')

    plt.tight_layout()
    if save_to != '':
        plt.savefig(save_to, dpi=300)


def plot_trace_lq_hq_mean_org_int(save_to=''):
    """Swarmplot of mean organoid intensity, x = low / high quality, y = mean organoid intensity. One point per sample.

    Args:
        save_to (str, optional): path to save plot. Defaults to ''.
    """
    _, dfs_dwmri = get_metrics_global_cyst_seg_dw_mri()
    df_trace = dfs_dwmri[1]
    plt.figure(figsize=(4, 3), facecolor='white')
    sns.swarmplot(data=df_trace.sort_values(
        'Organoid quality', ascending=True),
        y='mean_organoid_intensity', x='Organoid quality')
    plt.ylabel('$\mu_{int}$')
    plt.xlabel('Organoid quality', fontsize=11)
    plt.grid()
    plt.title(r'Sequence Trace (p < .001)', fontsize=11)
    sns.despine()
    plt.tight_layout()
    if save_to != '':
        plt.savefig(save_to, dpi=300)


def get_org_boundaries(org_locs_planes, i):
    """Extracts organoid boundaries.

    Args:
        org_locs_planes (list): containing the to-be-visualized planes and plane boundaries.
                                Example: [(['88-117', '59', '32-67'], ['88-117', '60', '32-67']),
                                          (['40-69', '41', '40-75'], ['40-69', '45', '40-75']),
                                          (['5-55', '58', '15-90'], ['5-55', '61', '15-90']),
                                          (['17-67', '36', '15-85'], ['17-67', '53', '15-85'])]
        i (int): in range(4)

    Returns:
        list: list of 10 values, e.g. for example above and i = 0: [88, 117, 59, 32, 67, 88, 117, 60, 32, 67]
    """
    a, b = org_locs_planes[i][0][0].split('-')
    c = org_locs_planes[i][0][1]
    d, e = org_locs_planes[i][0][2].split('-')

    f, g = org_locs_planes[i][1][0].split('-')
    h = org_locs_planes[i][1][1]
    j, k = org_locs_planes[i][1][2].split('-')
    x = [a, b, c, d, e, f, g, h, j, k]
    x = [int(z) for z in x]
    return (*x,)


def plot_examples_lq_hq_organoids(
        org_ids, compactnesses, org_locs_planes, save_to=''):
    """Creates 2x2 plot with two examples per organoid and four organoids in total

    Args:
        org_ids (list): list of four organoid ids
        compactnesses (list): list of four compactnesses corresponding to the org_ids
        org_locs_planes (list): containing the to-be-visualized planes and plane boundaries.
                                Example: [(['88-117', '59', '32-67'], ['88-117', '60', '32-67']),
                                          (['40-69', '41', '40-75'], ['40-69', '45', '40-75']),
                                          (['5-55', '58', '15-90'], ['5-55', '61', '15-90']),
                                          (['17-67', '36', '15-85'], ['17-67', '53', '15-85'])]
        save_to (str, optional): path to save plot. Defaults to ''.
    """
    assert len(org_ids) == 4
    imgs = get_orig_imgs(kind='mri')
    extract_imgs = []
    img_bufs = [io.BytesIO() for i in range(4)]
    i = 0
    for org_id in org_ids:
        #     print('org_id', org_id)
        fig, axs = plt.subplots(1, 2, figsize=(5, 5))
        a, b, c, d, e, f, g, h, j, k = get_org_boundaries(org_locs_planes, i)
        axs[0].imshow(np.rot90(imgs[org_id][a:b, c, d:e]*-1, 3), cmap='Greys')
        axs[1].imshow(np.rot90(imgs[org_id][f:g, h, j:k]*-1, 3), cmap='Greys')
        axs[0].axis('off')
        axs[1].axis('off')
        plt.tight_layout()
        plt.savefig(img_bufs[i])
        plt.close()
        i += 1

    extract_imgs = [Image.open(img_buf) for img_buf in img_bufs]
    subplot_prefixes = ['(a)', '(b)', '(c)', '(d)']
    fig, axs = plt.subplots(2, 2, facecolor='white')
    axs = axs.T.ravel()
    for i in range(4):
        axs[i].imshow(extract_imgs[i])
        axs[i].axis('off')
        axs[i].set_title(
            f'{subplot_prefixes[i]} {org_ids[i]} (C={compactnesses[i]})', y=-
            0.01)

    axs[0].text(-10.8, 0.01, 'High-quality organoids', fontsize=12,
                verticalalignment='center', horizontalalignment='left',
                fontweight='bold')
    axs[2].text(-10.8, 0.01, 'Low-quality organoids', fontsize=12,
                verticalalignment='center', horizontalalignment='left',
                fontweight='bold')
    plt.tight_layout()
    if save_to != '':
        plt.savefig(save_to, dpi=300)


def plot_organoid_growth_over_time():
    """lineplot with x = days y = organoid volume
    """
    org_ids, org_sizes = [], []
    gt = get_masks(kind='gt_org_loc')
    for org_id, org_gt in gt.items():
        org_ids.append(org_id)
        c = Counter(org_gt.flatten())
        assert len(c) == 2
        assert 0 in c and 1 in c
        org_sizes.append(c[1])

    df = pd.DataFrame.from_dict({'org_id': org_ids, 'org_size': org_sizes})
    df = df.merge(pd.read_csv(CSV_ORG_OVERVIEW),
                  on='org_id').sort_values('org_id')
    df['org_nr'] = df['org_nr'].astype('str')

    sns.lineplot(data=df, x='day', y='org_size', hue='org_nr')
    plt.grid()
    plt.ylabel('Organoid volume in voxel')
    plt.xlabel('Day')
    plt.legend(title='Organoid', loc='center left', bbox_to_anchor=(1, 0.5))
    sns.despine(bottom=True, top=True, left=True, right=True)
    plt.tight_layout()
    return df


def get_green_binary_colors(x):
    """Converts 1-values of binary array to green RGB color.

    Args:
        x (ndarray): binary numpy array

    Returns:
        x (ndarray): binary numpy array with 1-values converted to green color.
    """
    x = x[:, :, np.newaxis]
    x = np.where(x == 0.0, [1.0, 1.0, 1.0, 0.0], x)
    x = np.where(x == 1.0, [60/255, 179/255, 113/255, 0.6], x)
    return x
