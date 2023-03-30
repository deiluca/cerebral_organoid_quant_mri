import os
import numpy as np
import pandas as pd
from os.path import join as opj
from sklearn.model_selection import train_test_split
import shutil
from collections import Counter

from scripts.utils.io_utils import write_h5
from scripts.utils.constants import MRI_IMG_DIR, MRI_ORG_LOC_GT_DIR, MRI_CYST_LOC_GT_DIR, MRI_ORG_SEG_FILES_3DUNET, MRI_CYST_SEG_FILES_3DUNET, CSV_ORG_OVERVIEW, ORG_ID, MRI_CYST_SEG_SPLITS, MRI_ORG_SEG_SPLITS


class DataPreparerSegmentation(object):
    def __init__(self,
                 imgdir=MRI_IMG_DIR,
                 org_seg_gtdir=MRI_ORG_LOC_GT_DIR,
                 org_seg_files_3dunet=MRI_ORG_SEG_FILES_3DUNET,
                 cyst_seg_gtdir=MRI_CYST_LOC_GT_DIR,
                 cyst_seg_files_3dunet=MRI_CYST_SEG_FILES_3DUNET,
                 fixed_splits=True):
        """Initialize DataPreparerSegmentation

        Args:
            imgdir (str, optional): directory of extracted MRI/DTI images. Defaults to MRI_IMG_DIR.
            org_seg_gtdir (str, optional): directory of ground truth organoid segmentation files. Defaults to MRI_ORG_LOC_GT_DIR.
            org_seg_files_3dunet (str, optional): directory of output directory for h5files for 3D U-Net training on organoid segmentation. Defaults to MRI_ORG_SEG_FILES_3DUNET.
            cyst_seg_gtdir (str, optional): directory of ground truth cyst segmentation files. Defaults to MRI_CYST_LOC_GT_DIR.
            cyst_seg_files_3dunet (str, optional): directory of output directory for h5files for 3D U-Net training on cyst segmentation. Defaults to MRI_CYST_SEG_FILES_3DUNET.
            fixed_splits (bool, optional): used pre-defined trainin-validation or not. Defaults to True.
        """
        self.imgdir = imgdir
        self.org_seg_gtdir = org_seg_gtdir
        self.org_seg_files_3dunet = org_seg_files_3dunet
        self.cyst_seg_gtdir = cyst_seg_gtdir
        self.cyst_seg_files_3dunet = cyst_seg_files_3dunet
        self.fixed_splits = fixed_splits

    def create_h5files_organoid_seg(self):
        """Generates one h5 file per sample. This h5 file encompasses the image and the ground truth mask for organoid segmentation.
        """
        print('Creating H5 files for organoid seg...', end=' ')
        os.makedirs(self.org_seg_files_3dunet, exist_ok=True)
        for f in os.listdir(self.imgdir):
            if not f.endswith('.npy'):
                continue
            img_file = opj(self.imgdir, f)
            org_id = f.replace('.npy', '')
            gt_file = opj(self.org_seg_gtdir, f'{org_id}.npy')

            img = np.load(img_file)
            gt = np.load(gt_file)
            assert img.shape == gt.shape
            write_h5(os.path.join(
                self.org_seg_files_3dunet, f"{org_id}.h5"), img, gt)

        print('done')

    def get_arr(self, x):
        """Transforms string of numpy array to numpy array.

        Args:
            x (str): String in the form '['org5_0609', 'org6_0518', 'org4_0609']'

        Returns:
            ndarray: For example above: ['org5_0609', 'org6_0518', 'org4_0609']
        """
        return [y.replace('[', '').replace(']', '').replace('\'', '').strip() for y in x.split(',')]

    def create_loocv_splits_organoid_seg(self):
        """Performs leave-one-out cross validation per organoid, resulting in nine splits in total.
        """

        df = pd.read_csv(CSV_ORG_OVERVIEW)
        print('Creating LOOCV splits for organoid seg...', end=' ')

        for i in range(1, 10):
            outdir = opj(self.org_seg_files_3dunet, f'LOO_org{i}')
            if os.path.isdir(outdir):
                shutil.rmtree(outdir)
            os.makedirs(outdir)
            if self.fixed_splits:
                splits = pd.read_csv(MRI_ORG_SEG_SPLITS)
                test = splits[splits['split'] == f'LOO_org{i}']['test'].tolist()[
                    0]
                train = splits[splits['split'] == f'LOO_org{i}']['train'].tolist()[
                    0]
                val = splits[splits['split'] == f'LOO_org{i}']['val'].tolist()[
                    0]
                train, val, test = self.get_arr(
                    train), self.get_arr(val), self.get_arr(test)
            else:
                test_cond = df['org_nr'] == i
                test = df[test_cond]
                train, val = train_test_split(
                    df[~test_cond], test_size=0.2, random_state=0)
                train = train[ORG_ID].tolist()
                val = val[ORG_ID].tolist()
                test = test[ORG_ID].tolist()

            target_dir = opj(outdir, 'train')
            os.makedirs(target_dir)
            for train_org in train:
                shutil.copyfile(
                    opj(self.org_seg_files_3dunet, train_org + ".h5"),
                    opj(target_dir, train_org + ".h5"))

            target_dir = opj(outdir, 'val')
            os.makedirs(target_dir)
            for val_org in val:
                shutil.copyfile(opj(self.org_seg_files_3dunet,
                                val_org+".h5"), opj(target_dir, val_org+".h5"))

            target_dir = opj(outdir, 'test')
            os.makedirs(target_dir)
            for test_org in test:
                shutil.copyfile(
                    opj(self.org_seg_files_3dunet, test_org + ".h5"),
                    opj(target_dir, test_org + ".h5"))

        print('done')

    def create_h5files_cyst_seg(self):
        """Generates one h5 file per sample. This h5 file encompasses the image and the ground truth mask for cyst segmentation.
        """
        print('Creating H5 files for local cyst seg...', end=' ')
        os.makedirs(self.cyst_seg_files_3dunet, exist_ok=True)
        for f in os.listdir(self.cyst_seg_gtdir):
            if not f.endswith('.npy'):
                continue
            gt_file = opj(self.cyst_seg_gtdir, f)
            org_id = f.replace('.npy', '')
            img_file = opj(self.imgdir, f'{org_id}.npy')

            img = np.load(img_file)
            gt = np.load(gt_file)

            assert img.shape == gt.shape
            write_h5(os.path.join(
                self.cyst_seg_files_3dunet, f"{org_id}.h5"), img, gt)

        print('done')

    def create_loocv_splits_local_cyst_seg(self):
        """Generates one h5 file per sample. This h5 file encompasses the image and the ground truth mask for cyst segmentation.
        """
        print('Creating LOOCV splits for local cyst seg...', end=' ')

        orgs_local_cyst_seg = [x.replace('.npy', '')
                               for x in os.listdir(self.cyst_seg_gtdir)]
        for org in range(1, 9):
            outdir = opj(self.cyst_seg_files_3dunet, f'LOO_org{org}')
            if os.path.isdir(outdir):
                shutil.rmtree(outdir)
            os.makedirs(outdir)
            if self.fixed_splits:
                splits = pd.read_csv(MRI_CYST_SEG_SPLITS)
                orgs_test = splits[splits['split'] == f'LOO_org{org}']['test'].tolist()[
                    0]
                train = splits[splits['split'] == f'LOO_org{org}']['train'].tolist()[
                    0]
                val = splits[splits['split'] == f'LOO_org{org}']['val'].tolist()[
                    0]
                train, val, orgs_test = self.get_arr(
                    train), self.get_arr(val), self.get_arr(orgs_test)
            else:
                orgs_test = [
                    x for x in orgs_local_cyst_seg if f'org{org}' in x]
                orgs_remaining = [
                    x for x in orgs_local_cyst_seg if x not in orgs_test]
                train, val = train_test_split(
                    orgs_remaining, test_size=0.2, random_state=0)

            # no test organoid available for LOO org 9
            if org == 9:
                continue
            target_dir = opj(outdir, 'test')
            os.makedirs(target_dir)
            for test_org in orgs_test:
                shutil.copyfile(
                    opj(self.cyst_seg_files_3dunet, test_org + ".h5"),
                    opj(target_dir, test_org + ".h5"))

            target_dir = opj(outdir, 'train')
            os.makedirs(target_dir)
            for train_org in train:
                shutil.copyfile(
                    opj(self.cyst_seg_files_3dunet, train_org + ".h5"),
                    opj(target_dir, train_org + ".h5"))

            target_dir = opj(outdir, 'val')
            os.makedirs(target_dir)
            for val_org in val:
                shutil.copyfile(opj(self.cyst_seg_files_3dunet,
                                val_org+".h5"), opj(target_dir, val_org+".h5"))
        print('done')
