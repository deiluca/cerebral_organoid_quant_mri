import sys
sys.path.append('/home/ws/oc9627/cerebral_organoid_quant_mri')

from scripts.utils.DICOMImageExtractor import DICOMImageExtractor
from scripts.utils.DataPreparerSegmentation import DataPreparerSegmentation

# extract single images from DICOM images
# die = DICOMImageExtractor()
# die.extract_t2star_imgs()
# die.extract_dw_mri_images()

# prepare images for organoid segmentation and local cyst segmentation
dps = DataPreparerSegmentation()
dps.create_h5files_organoid_seg()
dps.create_loocv_splits_organoid_seg()