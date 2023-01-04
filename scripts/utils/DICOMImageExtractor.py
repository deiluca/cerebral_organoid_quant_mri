import os
import pydicom
import numpy as np
import numpy
from os.path import join as opj

from scripts.utils.constants import IGNORE_SAMPLES_AT_DATES, MRI_ROOTDIR, MRI_IMG_DIR, DWMRI_IMG_DIR
from scripts.utils.minor_utils import min_max_norm

class DICOMImageExtractor(object):
    def __init__(self,
                mri_rootdir=MRI_ROOTDIR,
                outdir_mri=MRI_IMG_DIR,
                outdir_dwmri=DWMRI_IMG_DIR):

        assert os.path.isdir(mri_rootdir)

        self.mri_rootdir = mri_rootdir
        self.outdir_mri=outdir_mri
        self.outdir_dwmri=outdir_dwmri

    def extract_t2star_imgs(self):
        """
        Saves one npy file per organoid in folder self.outdir_mri
        """
        os.makedirs(self.outdir_mri, exist_ok=True)
        #TODO also try it the same way as for DW-MRI extraction (without hardcoding) 
        t2_star_dirs = ['1-3/1805/DS_organoid2022_1bis3_DS_organoid2022_1bis3_20220518__E13_P1',
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
        # those EV are just filled with medium to ensure consistency between MRI runs (don't contain organoids)

        for t2_star_dir in t2_star_dirs:
            print('Processing t2star DICOM dir: ' + t2_star_dir)
            # extract organoid ids
            org_nrs = range(int(t2_star_dir[0]), int(t2_star_dir[2])+1)
            org_date = t2_star_dir.split("/")[1][2:] + t2_star_dir.split("/")[1][:2]
            ArrayDicom, _ = self._get_dcm_arr(os.path.join(self.mri_rootdir, t2_star_dir))
            img1, img2, img3 = self._split_image_three(ArrayDicom)
            org1_id = "org{}_{}".format(org_nrs[0], org_date)
            org2_id = "org{}_{}".format(org_nrs[1], org_date) 
            org3_id = "org{}_{}".format(org_nrs[2], org_date)
            
            assert img1.shape== img2.shape and img2.shape==img3.shape
            # save one npy file per organoid
            np.save(os.path.join(self.outdir_mri, f"{org1_id}.npy"), img1)
            if org2_id not in IGNORE_SAMPLES_AT_DATES:
                np.save(os.path.join(self.outdir_mri, f"{org2_id}.npy"), img2)
            if org3_id not in IGNORE_SAMPLES_AT_DATES:
                np.save(os.path.join(self.outdir_mri, f"{org3_id}.npy"), img3)
        print('Done')

    def extract_dw_mri_images(self):
        """
        Saves one npy file per organoid and sequence in folder self.outdir_dwmri/seq[i]
        """
        os.makedirs(self.outdir_dwmri, exist_ok=True)
        org_locs = dict()
        for root, dirs, files in os.walk(self.mri_rootdir):
            if len(files)==264:
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
                    img0, img1, img2 = self._split_image_three(ArrayDicom, do_min_max_norm=False)
                    img0, img1, img2 = img0[:,:,i::22], img1[:,:,i::22], img2[:,:,i::22]
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
        """
        Adapted from: https://github.com/cxr-eye-gaze/eye-gaze-dataset/blob/master/DataProcessing/DataPreparation/image_preparation.py
        """
        dicom_dict = {}
        repr(dicom_header)
        for dicom_value in dicom_header.values():
            if dicom_value.tag == (0x7fe0, 0x0010):
                # discard pixel data
                continue
            if type(dicom_value.value) == pydicom.dataset.Dataset:
                dicom_dict[dicom_value.tag] = self._dicom_dataset_to_dict(dicom_value.value)
            else:
                v = self._convert_value(dicom_value.value)
                dicom_dict[dicom_value.tag] = v
        return dicom_dict
    
    def _get_dcm_arr(self, filepath, kind='t2star'):
        assert kind in ['t2star', 'dwmri']
        PathDicom = filepath
        lstFilesDCM = []  # create an empty list
        for dirName, subdirList, fileList in os.walk(PathDicom):
            for filename in sorted(fileList):
                if ".dcm" in filename.lower():  # check whether the file's DICOM
                    lstFilesDCM.append(os.path.join(dirName,filename))
    #                 print(filename)

        # Get ref file
        RefDs = pydicom.read_file(lstFilesDCM[0])
        if kind == 't2star':
            if "T2Star_FLASH_3D_TE_18ms_15kHz_80_80_80_15mm" not in self._dicom_dataset_to_dict(RefDs).values():
                raise Exception
        else:
            pass
        # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
        ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

        # Load spacing values (in mm)
        ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

        # The array is sized based on 'ConstPixelDims'
        ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

        # loop through all the DICOM files
        for filenameDCM in lstFilesDCM:
            # read the file
            ds = pydicom.read_file(filenameDCM)
            # store the raw image data
            ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array
        return ArrayDicom, RefDs
        
    def _split_image_three(self, dcmarr, do_min_max_norm=True):
        if do_min_max_norm:
            dcmarr = min_max_norm(dcmarr)
        split_dim = 0
        dim_length = int(len(dcmarr[:, 0, 0]))
        arrs = []
        for i in range(0, dim_length, int(dim_length/3)):
            arrs.append(dcmarr[i:i+int(dim_length/3), :, :])
        return arrs[0], arrs[1], arrs[2]
    
    def _convert_value(self, v):
        """
        Source: https://github.com/cxr-eye-gaze/eye-gaze-dataset/blob/master/DataProcessing/DataPreparation/image_preparation.py
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
        """
        Source: https://github.com/cxr-eye-gaze/eye-gaze-dataset/blob/master/DataProcessing/DataPreparation/image_preparation.py
        """
        return s.replace(u"\u0000", "").strip()





