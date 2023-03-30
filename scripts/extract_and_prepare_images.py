"""Extracts and prepares all MRI images"""

from scripts.utils.data_preparer_segmentation import DataPreparerSegmentation  # noqa
from scripts.utils.dicom_image_extractor import DICOMImageExtractor  # noqa

# extract single images from DICOM images
die = DICOMImageExtractor()
die.extract_t2star_imgs()
die.extract_dw_mri_images()

# prepare images for organoid segmentation and local cyst segmentation
dps = DataPreparerSegmentation()
dps.create_h5files_organoid_seg()
dps.create_loocv_splits_organoid_seg()
dps.create_h5files_cyst_seg()
dps.create_loocv_splits_local_cyst_seg()
