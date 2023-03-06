from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
import io
from PIL import Image

from scripts.utils.io_utils import get_masks, get_orig_imgs
from scripts.utils.dwmri import get_metrics_global_cyst_seg_dw_mri
from scripts.utils.global_cyst_classification import get_compactness


def plot_compactness(df, roc_auc_compactness, save_to=''):
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


def plot_trace_lq_hq_mean_org_int(save_to='', norm_by_max=True):
    _, dfs_dwmri = get_metrics_global_cyst_seg_dw_mri()
    df_trace = dfs_dwmri[1]
    trace_max_intensity = 32766
    df_trace['mean_org_intensity_norm'] = df_trace['mean_organoid_intensity'] / \
        trace_max_intensity
    plt.figure(figsize=(4, 3), facecolor='white')
    if norm_by_max:
        sns.swarmplot(
            data=dfs_dwmri[1].sort_values('Organoid quality', ascending=True), y='mean_org_intensity_norm', x='Organoid quality')
    else:
        sns.swarmplot(
            data=dfs_dwmri[1].sort_values('Organoid quality', ascending=True), y='mean_organoid_intensity', x='Organoid quality')
    plt.ylabel('$\mu_{int}$')
    plt.xlabel('Organoid quality', fontsize=11)
    plt.grid()
    plt.title(r'Sequence Trace (P=$4x10^{-7}$)', fontsize=11)
    sns.despine()
    plt.tight_layout()
    if save_to != '':
        plt.savefig(save_to, dpi=300)


def get_org_boundaries(org_locs_planes, i):
    a, b = org_locs_planes[i][0][0].split('-')
    c = org_locs_planes[i][0][1]
    d, e = org_locs_planes[i][0][2].split('-')

    f, g = org_locs_planes[i][1][0].split('-')
    h = org_locs_planes[i][1][1]
    j, k = org_locs_planes[i][1][2].split('-')
    x = [a, b, c, d, e, f, g, h, j, k]
    x = [int(z) for z in x]
    return (*x,)


def plot_examples_lq_hq_organoids(org_ids, compactnesses, org_locs_planes, save_to=''):
    """
    Creates 2x2 plot with two examples per organoid and four organoids in total
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
            f'{subplot_prefixes[i]} {org_ids[i]} (C={compactnesses[i]})', y=-0.01)

    axs[0].text(-10.8, 0.01, 'High-quality organoids', fontsize=12,
                verticalalignment='center', horizontalalignment='left', fontweight='bold')
    axs[2].text(-10.8, 0.01, 'Low-quality organoids', fontsize=12,
                verticalalignment='center', horizontalalignment='left', fontweight='bold')
    plt.tight_layout()
    if save_to != '':
        plt.savefig(save_to, dpi=300)


def plot_organoid_growth_over_time():
    org_ids, org_sizes = [], []
    gt = get_masks(kind='gt_org_loc')
    for org_id, org_gt in gt.items():
        org_ids.append(org_id)
        c = Counter(org_gt.flatten())
        assert len(c) == 2
        assert 0 in c and 1 in c
        org_sizes.append(c[1])

    df = pd.DataFrame.from_dict({'org_id': org_ids, 'org_size': org_sizes})
    df = df.merge(pd.read_csv('mri_paper_data/data_overview.csv'),
                  on='org_id').sort_values('org_id')
    df['org_nr'] = df['org_nr'].astype('str')

    sns.lineplot(data=df, x='day', y='org_size', hue='org_nr')
    plt.grid()
    plt.ylabel('Organoid volume in voxel')
    plt.xlabel('Day')
    plt.legend(title='Organoid', loc='center left', bbox_to_anchor=(1, 0.5))
    sns.despine(bottom=True, top=True, left=True, right=True)
    plt.tight_layout()


def get_green_binary_colors(x):
    x = x[:, :, np.newaxis]
    x = np.where(x == 0.0, [1.0, 1.0, 1.0, 0.0], x)
    x = np.where(x == 1.0, [60/255, 179/255, 113/255, 0.6], x)
    return x


def plot_test_performance(df, save_to=''):
    assert 'org_nr' in df.columns
    assert 'org_nr' in df.columns
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))

    # barplot
    # create copy of dataframe to display 'Overall' performance in plot
#     df_copy=df.copy()
#     df_copy['org_nr'] = 'Overall'
#     df_bar = pd.concat([df, df_copy])

#     sns.barplot(data=df_bar, x='org_nr', y='Test Dice', ax=axs[0])
#     axs[0].set_ylim(0.0, 1.0)
#     axs[0].set_xlabel('Organoid')
#     sns.despine(top=True, left=True, bottom=True, right=True, ax=axs[0])

    # boxplot
    sns.boxplot(data=df, x='org_nr', y='Test Dice', ax=axs[0])
    axs[0].set_ylim(0.0, 1.0)
    axs[0].set_xlabel('Organoid')
    sns.despine(top=True, left=True, bottom=True, right=True, ax=axs[0])

    # lineplot
    df['org_nr'] = df['org_nr'].astype('str')
    sns.lineplot(data=df, x="day", y="Test Dice",
                 hue="org_nr", sort=True, ax=axs[1])
    axs[1].set_ylabel('Test Dice')
    axs[1].set_xlabel('Day')
    axs[1].set_ylim((0.0, 1.0))
    sns.despine(bottom=True, top=True, left=True, right=True, ax=axs[1])
    axs[1].legend(title='Organoid', loc='center left', bbox_to_anchor=(1, 0.5))

#     df['org_id_readable'] = 'Org ' + df['org_nr'].astype('str')+', Day '+df['day'].astype('str')
#     sns.barplot(data=df, x='org_id_readable', y='Test Dice', color='grey', ax=axs[2])
#     axs[2].tick_params(labelrotation=90)
#     axs[2].set_xlabel('')
#     axs[2].set_ylim(0.0, 1.0)

    plt.tight_layout()
    if save_to != '':
        plt.savefig(save_to, dpi=400)
