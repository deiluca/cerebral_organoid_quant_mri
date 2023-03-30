"""Extract DICOM images """
import os
from os.path import join as opj
import numpy as np
import pydicom


from scripts.utils.constants import IGNORE_SAMPLES_AT_DATES, MRI_ROOTDIR, MRI_IMG_DIR, DWMRI_IMG_DIR, T2STAR_DIRS
from scripts.utils.minor_utils import min_max_norm


class DICOMImageExtractor():
    """"Extract DICOM images
    """

    def __init__(self,
                 mri_rootdir=MRI_ROOTDIR,
                 outdir_mri=MRI_IMG_DIR,
                 outdir_dwmri=DWMRI_IMG_DIR):
        """Initialize DICOMImageExtractor

        Args:
            mri_rootdir (str, optional): root directory of MRI raw data. Defaults to MRI_ROOTDIR.
            outdir_mri (str, optional): output directory for MRI (T2*) image extraction. Defaults to MRI_IMG_DIR.
            outdir_dwmri (str, optional): output directory for DTI image extraction. Defaults to DWMRI_IMG_DIR.
        """
        assert os.path.isdir(mri_rootdir)

        self.mri_rootdir = mri_rootdir
        self.outdir_mri = outdir_mri
        self.outdir_dwmri = outdir_dwmri

    def extract_t2star_imgs(self):
        """Saves one npy file per organoid in folder self.outdir_mri
        """
        os.makedirs(self.outdir_mri, exist_ok=True)
        # those EV are just filled with medium to ensure consistency between MRI runs (don't contain organoids)

        for t2_star_dir in T2STAR_DIRS:
            print('Processing t2star DICOM dir: ' + t2_star_dir)
            # extract organoid ids
            org_nrs = range(int(t2_star_dir[0]), int(t2_star_dir[2])+1)
            org_date = t2_star_dir.split(
                "/")[1][2:] + t2_star_dir.split("/")[1][:2]
            ArrayDicom, _ = self._get_dcm_arr(
                os.path.join(self.mri_rootdir, t2_star_dir))
            img1, img2, img3 = self._split_image_three(ArrayDicom)
            org1_id = f"org{org_nrs[0]}_{org_date}"
            org2_id = f"org{org_nrs[1]}_{org_date}"
            org3_id = f"org{org_nrs[2]}_{org_date}"

            assert img1.shape == img2.shape and img2.shape == img3.shape
            # save one npy file per organoid
            np.save(os.path.join(self.outdir_mri, f"{org1_id}.npy"), img1)
            if org2_id not in IGNORE_SAMPLES_AT_DATES:
                np.save(os.path.join(self.outdir_mri, f"{org2_id}.npy"), img2)
            if org3_id not in IGNORE_SAMPLES_AT_DATES:
                np.save(os.path.join(self.outdir_mri, f"{org3_id}.npy"), img3)
        print('Done')

    def extract_dw_mri_images(self):
        """Saves one npy file per organoid and sequence in folder self.outdir_dwmri/seq[i]
        """
        os.makedirs(self.outdir_dwmri, exist_ok=True)
        org_locs = dict()
        for root, _, files in os.walk(self.mri_rootdir):
            if len(files) == 264:
                time = os.path.basename(os.path.dirname(root))
                org = os.path.basename(os.path.dirname(os.path.dirname(root)))
                if org not in org_locs:
                    org_locs[org] = dict()
                org_locs[org][time] = root
        for i in range(0, 22):
            print(f'Extracting DW-MRI sequence {i}')
            outdir2 = opj(self.outdir_dwmri, f'seq{i}')
            os.makedirs(outdir2, exist_ok=True)

            org_mri = dict()
            for k, v in org_locs.items():
                for k2, v2 in v.items():
                    org, time = k, k2
                    orgid1, orgid3 = int(org[0]), int(org[2])
                    orgid2 = orgid3-1
                    time = time[2:] + time[:2]
                    orgname1 = f'org{orgid1}_{time}'
                    orgname2 = f'org{orgid2}_{time}'
                    orgname3 = f'org{orgid3}_{time}'

                    ArrayDicom, _ = self._get_dcm_arr(v2, kind='dwmri')
                    img0, img1, img2 = self._split_image_three(
                        ArrayDicom, do_min_max_norm=False)
                    img0, img1, img2 = img0[:, :, i::22], img1[:,
                                                               :, i::22], img2[:, :, i::22]
                    if orgname1 not in IGNORE_SAMPLES_AT_DATES:
                        org_mri[orgname1] = img0
                        np.save(os.path.join(outdir2, f'{orgname1}.npy'), img0)
                    if orgname2 not in IGNORE_SAMPLES_AT_DATES:
                        org_mri[orgname2] = img1
                        np.save(os.path.join(outdir2, f'{orgname2}.npy'), img1)
                    if orgname3 not in IGNORE_SAMPLES_AT_DATES:
                        org_mri[orgname3] = img2
                        np.save(os.path.join(outdir2, f'{orgname3}.npy'), img2)
        print('Done')

    def _dicom_dataset_to_dict(self, dicom_header):
        """Helper function for DICOM preparation. Adapted from: https://github.com/cxr-eye-gaze/eye-gaze-dataset/blob/master/DataProcessing/DataPreparation/image_preparation.py

        Args:
            dicom_header (pydicom.dataset.Dataset): Pydicom dataset

        Returns:
            dict: Dictionary with the same information
        """
        dicom_dict = {}
        repr(dicom_header)
        for dicom_value in dicom_header.values():
            if dicom_value.tag == (0x7fe0, 0x0010):
                # discard pixel data
                continue
            if type(dicom_value.value) == pydicom.dataset.Dataset:
                dicom_dict[dicom_value.tag] = self._dicom_dataset_to_dict(
                    dicom_value.value)
            else:
                v = self._convert_value(dicom_value.value)
                dicom_dict[dicom_value.tag] = v
        return dicom_dict

    def _get_dcm_arr(self, filepath, kind='t2star'):
        """Extracts 3D array containing the pixel values from directory of DICOM files.

        Args:
            filepath (str): directory of DICOM files
            kind (str, optional): either 't2star', 'dwmri'. Defaults to 't2star'.
        """
        assert kind in ['t2star', 'dwmri']
        PathDicom = filepath
        lstFilesDCM = []  # create an empty list
        for dirName, _, fileList in os.walk(PathDicom):
            for filename in sorted(fileList):
                if ".dcm" in filename.lower():  # check whether the file's DICOM
                    lstFilesDCM.append(os.path.join(dirName, filename))
    #                 print(filename)

        # Get ref file
        RefDs = pydicom.read_file(lstFilesDCM[0])
        if kind == 't2star':
            if "T2Star_FLASH_3D_TE_18ms_15kHz_80_80_80_15mm" not in self._dicom_dataset_to_dict(RefDs).values():
                raise Exception
        else:
            pass
        # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
        ConstPixelDims = (int(RefDs.Rows), int(
            RefDs.Columns), len(lstFilesDCM))

        # The array is sized based on 'ConstPixelDims'
        if kind == 't2star':
            ArrayDicom = np.zeros(
                ConstPixelDims, dtype=RefDs.pixel_array.dtype)
        else:
            ArrayDicom = np.zeros(ConstPixelDims, dtype='float32')
        # loop through all the DICOM files
        for filenameDCM in lstFilesDCM:
            # read the file
            ds = pydicom.read_file(filenameDCM)
            # store the raw image data
            px = ds.pixel_array
            if kind == 'dwmri':
                # convert to DTI units
                px = px * ds.RescaleSlope + ds.RescaleIntercept
            ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = px
        return ArrayDicom, RefDs

    def _split_image_three(self, dcmarr, do_min_max_norm=True):
        """Splits MRI image array into three equal sized arrays.

        Args:
            dcmarr (ndarray): 3D array containing pixel values of MRI image
            do_min_max_norm (bool, optional): perform min-max normalization. Defaults to True.

        Returns:
            ndarray, ndarray, ndarray: Three equal sized arrays, each containing one organoid
        """
        if do_min_max_norm:
            dcmarr = min_max_norm(dcmarr)
        dim_length = int(len(dcmarr[:, 0, 0]))
        arrs = []
        for i in range(0, dim_length, int(dim_length/3)):
            arrs.append(dcmarr[i:i+int(dim_length/3), :, :])
        return arrs[0], arrs[1], arrs[2]

    def _convert_value(self, v):
        """Helper function for DICOM preparation. Source: https://github.com/cxr-eye-gaze/eye-gaze-dataset/blob/master/DataProcessing/DataPreparation/image_preparation.py

        Args:
            v (can take multiple types): can take multiple types
        """
        t = type(v)
        if t in (list, int, float):
            cv = v
        elif t == str:
            cv = self._sanitise_unicode(v)
        elif t == bytes:
            s = v.decode('ascii', 'replace')
            cv = self._sanitise_unicode(s)
        elif t == pydicom.valuerep.DSfloat:
            cv = float(v)
        elif t == pydicom.valuerep.IS:
            cv = int(v)
    #     elif t == pydicom.valuerep.PersonName3:
    #         cv = str(v)
        else:
            cv = repr(v)
        return cv

    def _sanitise_unicode(self, s):
        """Removes unicode. Helper function for DICOM preparation. Source: https://github.com/cxr-eye-gaze/eye-gaze-dataset/blob/master/DataProcessing/DataPreparation/image_preparation.py

        Args:
            s (str): string containing unicode

        Returns:
            str: string without unicode
        """
        return s.replace(u"\u0000", "").strip()
