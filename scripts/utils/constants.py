"""Defining all constant variables"""

WORKING_DIR = '/home/ws/oc9627/cerebral_organoid_quant_mri'

MRI_ROOTDIR = 'data/data_zenodo/MRI_raw_data'
MRI_IMG_DIR = 'data/T2_star_images/'
DWMRI_IMG_DIR = 'data/DW-MRI_images'

# annotations
MRI_ORG_LOC_GT_DIR = 'data/data_zenodo/annotations/MRI_ground_truth_organoid_locations'
MRI_CYST_LOC_GT_DIR = 'data/data_zenodo/annotations/MRI_ground_truth_cyst_locations/cyst_size_greater_1000'
MRI_CYST_LOC_GT_DIR_ALL45 = 'data/data_zenodo/annotations/MRI_ground_truth_cyst_locations'
DWMRI_ORG_LOC_GT_DIR = 'data/data_zenodo/annotations/DW-MRI_ground_truth_organoid_locations'
CSV_GLOBAL_CYST_ANNOT = 'data/data_zenodo/annotations/global_cyst_classification_annotations.csv'

# files for segmentation
MRI_ORG_SEG_FILES_3DUNET = 'data/organoid_seg_files_3D-U-Net'
MRI_CYST_SEG_FILES_3DUNET = 'data/cyst_seg_files_3D-U-Net'
MRI_ORG_SEG_SPLITS = 'data/splits_org_seg.csv'
MRI_CYST_SEG_SPLITS = 'data/splits_local_cyst_seg.csv'


CSV_ORG_OVERVIEW = 'data/data_zenodo/data_overview.csv'

ORG_ID = 'org_id'

IGNORE_SAMPLES_AT_DATES = ["org9_0530",
                           "org9_0523",
                           "org8_0530",
                           "org5_0615",
                           "org6_0615",
                           "org6_0609",
                           "org3_0630",
                           "org3_0707",
                           "org2_0707"]

DWMRI_SEQUENCES = ['Fractional Anisotropy',
                   'Trace',
                   'Intensity',
                   'Trace Weighted Image',
                   'Tensor Component Dxx',
                   'Tensor Component Dyy',
                   'Tensor Component Dzz',
                   'Tensor Component Dxy',
                   'Tensor Component Dxz',
                   'Tensor Component Dyz',
                   '1st Eigenvalue',
                   '2nd Eigenvalue',
                   '3rd Eigenvalue',
                   '1st Eigenvector x',
                   '1st Eigenvector y',
                   '1st Eigenvector z',
                   '2nd Eigenvector x',
                   '2nd Eigenvector y',
                   '2nd Eigenvector z',
                   '3rd Eigenvector x',
                   '3rd Eigenvector y',
                   '3rd Eigenvector z']

T2STAR_DIRS = [
    '1-3/1805/DS_organoid2022_1bis3_DS_organoid2022_1bis3_20220518__E13_P1',
    '1-3/2305/DS_organoid2022_1bis3_DS_organoid2022_1bis3_20220523__E13_P1',
    '1-3/3005/DS_organoid2022_1bis3_DS_organoid2022_1bis3_20220530__E6_P1',
    '1-3/0206/DS_organoid2022_1bis3_DS_organoid2022_1bis3_20220602__E26_P1',
    '1-3/0906/DS_organoid2022_1bis3_DS_organoid2022_1bis3_20220609__E6_P1',
    '1-3/1506/DS_organoid2022_1bis3_DS_organoid2022_1bis3_20220615__E6_P1',
    '1-3/2206/DS_organoid2022_1bis3_DS_organoid2022_1bis3_20220622__E6_P1',
    '1-3/3006/DS_organoid2022_1bis3_DS_organoid2022_1bis3_20220630__E6_P1',
    '1-3/0707/DS_organoid2022_1bis3_DS_organoid2022_1bis3_20220707__E6_P1',
    '4-6/1805/DS_organoid2022_4bis6_DS_organoid2022_4bis6_20220518__E13_P1',
    '4-6/2305/DS_organoid2022_4bis6_DS_organoid2022_4bis6_20220523__E6_P1',
    '4-6/3005/DS_organoid2022_4bis6_DS_organoid2022_4bis6_20220530__E6_P1',
    '4-6/0206/DS_organoid2022_4bis6_DS_organoid2022_4bis6_20220602__E6_P1',
    '4-6/0906/DS_organoid2022_4bis6_DS_organoid2022_4bis6_20220609__E6_P1',
    '4-6/1506/DS_organoid2022_4bis6_DS_organoid2022_4bis6_20220615__E6_P1',
    '7-9/1805/DS_organoid2022_7bis9_DS_organoid2022_7bis9_20220518__E13_P1',
    '7-9/2305/DS_organoid2022_7bis9_DS_organoid2022_7bis9_20220523__E6_P1',
    '7-9/3005/DS_organoid2022_7bis9_DS_organoid2022_7bis9_20220530__E6_P1']
