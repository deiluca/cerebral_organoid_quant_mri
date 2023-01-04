import os
import numpy as np
import pandas as pd
from os.path import join as opj
from sklearn.model_selection import train_test_split
import shutil

from scripts.utils.io_utils import write_h5
from scripts.utils.constants import MRI_IMG_DIR, MRI_ORG_LOC_GT_DIR, MRI_ORG_SEG_FILES_3DUNET, CSV_ORG_OVERVIEW, ORG_ID

class DataPreparerSegmentation(object):
    def __init__(self,
                 imgdir = MRI_IMG_DIR,
                 org_seg_gtdir=MRI_ORG_LOC_GT_DIR,
                 org_seg_files_3dunet=MRI_ORG_SEG_FILES_3DUNET):
        self.imgdir = imgdir
        self.org_seg_gtdir = org_seg_gtdir
        self.org_seg_files_3dunet = org_seg_files_3dunet

    def create_h5files_organoid_seg(self):
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
            write_h5(os.path.join(self.org_seg_files_3dunet, f"{org_id}.h5"), img, gt)

        print('done')

    def create_loocv_splits_organoid_seg(self):
        df = pd.read_csv(CSV_ORG_OVERVIEW)
        print('Creating LOOCV splits for organoid seg...', end=' ')

        for i in range(1, 10):
            test_cond = df['org_nr']==i
            test = df[test_cond]
            train, val = train_test_split(df[~test_cond], test_size=0.2, random_state=0)
    #         print(f'LOOCV split {i}: train.shape {train.shape}, val.shape {val.shape}, test.shape {test.shape}')
            outdir = opj(self.org_seg_files_3dunet, f'LOO_org{i}')
            if os.path.isdir(outdir):
                shutil.rmtree(outdir)
            os.makedirs(outdir)
            for _, row in train.iterrows():
                target_dir = opj(outdir, 'train')
                os.makedirs(target_dir, exist_ok=True)
                shutil.copyfile(opj(self.org_seg_files_3dunet, row[ORG_ID]+".h5"), opj(target_dir, row[ORG_ID]+".h5"))
            for _, row in val.iterrows():
                target_dir = opj(outdir, 'val')
                os.makedirs(target_dir, exist_ok=True)
                shutil.copyfile(opj(self.org_seg_files_3dunet, row[ORG_ID]+".h5"), opj(target_dir, row[ORG_ID]+".h5"))
            for _, row in test.iterrows():
                target_dir = opj(outdir, 'test')
                os.makedirs(target_dir, exist_ok=True)
                shutil.copyfile(opj(self.org_seg_files_3dunet, row[ORG_ID]+".h5"), opj(target_dir, row[ORG_ID]+".h5"))
        print(' done')