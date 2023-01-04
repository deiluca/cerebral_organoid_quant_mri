import os
import numpy as np
import h5py
from os.path import join as opj

from scripts.utils.constants import MRI_IMG_DIR, DWMRI_IMG_DIR, MRI_ORG_LOC_GT_DIR, DWMRI_ORG_LOC_GT_DIR

def get_orig_imgs(kind, seq=None):
    assert kind in ['mri', 'dwmri']
    if kind=='mri':
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


def get_organoid_locations(kind):
    assert kind in ['gt', 'predicted', 'gt_dwmri']
    if kind == 'gt':
        d = MRI_ORG_LOC_GT_DIR
        suffix = '.npy'
    elif kind=='gt_dwmri':
        d = DWMRI_ORG_LOC_GT_DIR
        suffix = '.npy'
    else:
        d = 'mri_paper_results/organoid_segmentation/checkpoint_files_3DUNet/all_predictions_on_test_sets'
        suffix = '_predictions.npy'
    all_gt = dict()
    for f in os.listdir(d):
        filename = os.path.join(d, f)        
        org = f.replace(suffix, '')
        gt_file = opj(d, f'{org}{suffix}')
        gt = np.load(gt_file)
        all_gt[org] = gt
    return all_gt

def write_h5(filename, raw, label):
    with h5py.File(filename, "w") as data_file:
        data_file.create_dataset("raw", data=raw)
        data_file.create_dataset("label", data=label)