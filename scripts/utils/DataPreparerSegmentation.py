from os.path import dirname, join
from pprint import pprint
import os
import pydicom
from PIL import Image
import numpy as np
import ipyplot
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import threshold_otsu, threshold_multiotsu
import pandas as pd
from sklearn.model_selection import KFold
import collections, numpy
import h5py
import seaborn as sns
import SimpleITK as sitk
from scipy.stats import ttest_ind
from scipy import stats
import statsmodels.api as sm
import pylab as py
from sklearn.preprocessing import minmax_scale
from os.path import join as opj
from collections import Counter
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve
from sklearn.model_selection import train_test_split
import shutil
import io
from PIL import Image

from scripts.utils.io_utils import write_h5
from constants import MRI_IMG_DIR, DWMRI_IMG_DIR, MRI_ORG_LOC_GT_DIR, MRI_ORG_SEG_FILES_3DUNET, CSV_ORG_OVERVIEW, ORG_ID

class DataPreparerSegmentation(object):
    def __init__(self,
                 imgdir = MRI_IMG_DIR,
                 org_seg_gtdir=MRI_ORG_LOC_GT_DIR,
                 org_seg_files_3dunet=MRI_ORG_SEG_FILES_3DUNET):
        self.imgdir = imgdir
        self.org_seg_gtdir = org_seg_gtdir
        self.org_seg_files_3dunet = org_seg_files_3dunet

    def create_h5files_organoid_seg(self):
        for f in os.listdir(self.imgdir):
            if not f.endswith('.npy'):
                continue
            img_file = opj(self.imgdir, f)        
            org_id = f.replace('.npy', '')
            gt_file = opj(self.gtdir, f'{org_id}.npy')
            
            img = np.load(img_file)
            gt = np.load(gt_file)
            assert img.shape == gt.shape
            write_h5(os.path.join(self.org_seg_files_3dunet, f"{org_id}.h5"), img, gt)

        print('Done')

    def create_loocv_splits_organoid_seg(self):
        df = pd.read_csv(CSV_ORG_OVERVIEW)
        
        for i in range(1, 10):
            test_cond = df['org_nr']==i
            test = df[test_cond]
            train, val = train_test_split(df[~test_cond], test_size=0.2, random_state=0)
    #         print(f'LOOCV split {i}: train.shape {train.shape}, val.shape {val.shape}, test.shape {test.shape}')
            outdir = opj(self.org_seg_files_3dunet, f'LOO_org{i}')
            if os.path.isdir:
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