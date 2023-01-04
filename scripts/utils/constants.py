WORKING_DIR = '/home/ws/oc9627/cerebral_organoid_quant_mri'

MRI_ROOTDIR = 'data/MRI_raw_data'
MRI_IMG_DIR='data/T2_star_images/'
DWMRI_IMG_DIR='data/DW-MRI_images'

# annotations
MRI_ORG_LOC_GT_DIR = 'data/annotations/MRI_ground_truth_organoid_locations'
MRI_CYST_LOC_GT_DIR = 'data/annotations/MRI_ground_truth_cyst_locations/cyst_size_greater_1000'
DWMRI_ORG_LOC_GT_DIR = 'data/annotations/DW-MRI_ground_truth_organoid_locations'
CSV_GLOBAL_CYST_ANNOT = 'data/annotations/global_cyst_classification_annotations.csv'

# files for segmentation
MRI_ORG_SEG_FILES_3DUNET = 'data/organoid_seg_files_3D-U-Net'
MRI_CYST_SEG_FILES_3DUNET = 'data/cyst_seg_files_3D-U-Net'

CSV_ORG_OVERVIEW = 'data/data_overview.csv'

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