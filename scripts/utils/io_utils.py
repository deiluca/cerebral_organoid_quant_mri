import os
import numpy as np
import h5py
from os.path import join as opj

from scripts.utils.constants import MRI_IMG_DIR, DWMRI_IMG_DIR, MRI_ORG_LOC_GT_DIR, DWMRI_ORG_LOC_GT_DIR, MRI_CYST_LOC_GT_DIR, MRI_CYST_LOC_GT_DIR_ALL45


def get_orig_imgs(kind, seq=None):
    """Get MRI / DTI images

    Args:
        kind (str): 'mri', 'dwmri'
        seq (int, optional): specific sequence for DTI. Defaults to None.

    Returns:
        dict: key is org_id, value ndarray representing image
    """
    assert kind in ['mri', 'dwmri']
    if kind == 'mri':
        d = MRI_IMG_DIR
    else:
        assert seq is not None
        d = opj(DWMRI_IMG_DIR, f'seq{seq}')
    imgs = {}

    for f in os.listdir(d):
        if not f.endswith('.npy'):
            continue
        org_id = f.replace('.npy', '')
        filename = os.path.join(d, f)
        imgs[org_id] = np.load(filename)

    return imgs


def get_masks(kind):
    """Get masks of predicted / GT, organoid / cyst segmentation in MRI / DTI.

    Args:
        kind (str): Options 'gt_org_loc', 'gt_cyst_loc', 'gt_cyst_loc_all45', 'predicted_org_loc', 'gt_org_loc_dwmri'

    Returns:
        dict: key is org_id, value ndarray representing mask
    """
    assert kind in ['gt_org_loc', 'gt_cyst_loc', 'gt_cyst_loc_all45',
                    'predicted_org_loc', 'gt_org_loc_dwmri']
    if kind == 'gt_org_loc':
        d = MRI_ORG_LOC_GT_DIR
        suffix = '.npy'
    elif kind == 'gt_org_loc_dwmri':
        d = DWMRI_ORG_LOC_GT_DIR
        suffix = '.npy'
    elif kind == 'gt_cyst_loc':
        d = MRI_CYST_LOC_GT_DIR
        suffix = '.npy'
    elif kind == 'gt_cyst_loc_all45':
        d = MRI_CYST_LOC_GT_DIR_ALL45
        suffix = '.npy'
    elif kind == 'predicted_org_loc':
        d = 'results/organoid_segmentation/checkpoint_dirs/all_predictions_on_test_sets'
        suffix = '_predictions.npy'
    all_gt = dict()
    for f in os.listdir(d):
        if not f.endswith('.npy'):
            continue
        filename = os.path.join(d, f)
        org = f.replace(suffix, '')
        gt_file = opj(d, f'{org}{suffix}')
        gt = np.load(gt_file)
        all_gt[org] = gt
    return all_gt


def write_h5(filename, raw, label):
    """Writes h5 file to disk.

    Args:
        filename (str): _description_
        raw (ndarray): image
        label (ndarray): mask (e.g. organoid segmentation or cyst segmentation mask)
    """
    with h5py.File(filename, "w") as data_file:
        data_file.create_dataset("raw", data=raw)
        data_file.create_dataset("label", data=label)
