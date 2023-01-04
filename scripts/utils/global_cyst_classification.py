from PIL import Image
import numpy as np
from skimage.filters import threshold_otsu
import pandas as pd

from scripts.utils.constants import CSV_ORG_OVERVIEW, CSV_GLOBAL_CYST_ANNOT
from scripts.utils.minor_utils import make_mask_arr_visible
from scripts.utils.io_utils import get_orig_imgs, get_organoid_locations

class GlobalCystClassifier(object):
    def __init__(self):
        pass

    

def get_otsu_mask(image, thresh=None):
    if thresh is None:
        thresh = threshold_otsu(image)
    binary = image > thresh
        
    pil_img = Image.fromarray((binary*(255)).astype("uint8"))
    
    return binary, pil_img


def get_eppendorf(imgs, org_loc, key, drop_last=False):
    eppendorf_masks = []
    for x in range(imgs[key].shape[-1]):
        # only do otsu in 2D slices containing organoids
        if np.max(org_loc[key][:,:,x])>0:
            y = make_mask_arr_visible(get_otsu_mask(image = imgs[key][:,:,x])[0])
        else:
            y = np.zeros((imgs[key].shape[:2]))
        eppendorf_masks.append(y)
    if drop_last:
        for i, x in enumerate(eppendorf_masks):
            if i+1 < len(eppendorf_masks):
                if np.max(x) > 0 and np.max(eppendorf_masks[i+1]) == 0:
                    eppendorf_masks[i] = np.zeros(eppendorf_masks[0].shape)

    eppendorf_masks_stacked = np.moveaxis(np.stack(np.array(eppendorf_masks)), 0, -1)
    
    return eppendorf_masks, eppendorf_masks_stacked

def get_all_otsu_masks(imgs, org_loc, drop_last=True):
    otsu, otsu_vis = {}, {}
    for key, x in imgs.items():
        otsu_non_stacked, otsu_stacked = get_eppendorf(imgs, org_loc, key, drop_last=drop_last)
        otsu[key]= otsu_stacked
        otsu_vis[key]= otsu_non_stacked
    
    return otsu, otsu_vis

def get_org_mean_and_compactness(imgs, org_loc, otsu, mult_255=True, only_org_mean=False):
    
    org_ids, org_nrs, org_dates, compactness, org_mean = [], [], [], [], []
    
    # calculate mean organoid intensity and compactness for each organoid
    for org_id, _ in imgs.items():
        org_nr, org_date = org_id.split('_')
        org_nr = org_nr[3]
        org_mask = org_loc[org_id].astype('bool')
        # EV mask is Otsu segmentation excluding organoid location
        if not only_org_mean:
            eppi_mask = otsu[org_id].astype('bool') & ~org_loc[org_id].astype('bool')
        
        if mult_255:
            org_pixel = (imgs[org_id]*255)[org_mask].ravel()
            if not only_org_mean:
                eppi_pixel = (imgs[org_id]*255)[eppi_mask].ravel()
        else:
            org_pixel = (imgs[org_id])[org_mask].ravel()
            if not only_org_mean:
                eppi_pixel = (imgs[org_id])[eppi_mask].ravel()

        # one array per column in the latter dataframe
        if not only_org_mean:
            compactness.append(abs(np.mean(org_pixel)- np.mean(eppi_pixel)))
        org_mean.append(np.mean(org_pixel))
        org_ids.append(org_id)
        org_dates.append(org_date)
        org_nrs.append(org_nr)
    if not only_org_mean:
        df = pd.DataFrame.from_dict({'org_id': org_ids,
                                     'org_nr': org_nrs,
                                     'date': org_dates,
                                     'mean_organoid_intensity':org_mean,
                                     'compactness':compactness})
    else:
        df = pd.DataFrame.from_dict({'org_id': org_ids,
                             'org_nr': org_nrs,
                             'date': org_dates,
                             'mean_organoid_intensity':org_mean})
    # for human readability
    df.merge(pd.read_csv(CSV_ORG_OVERVIEW)[['org_id', 'date_readable', 'day']], on='org_id')

    
    return df

def get_compactness():
    imgs = get_orig_imgs(kind='mri')
    gt = get_organoid_locations(kind='gt')
    otsu, _ = get_all_otsu_masks(imgs, gt)

    df = get_org_mean_and_compactness(imgs, 
                                    org_loc=gt, 
                                    otsu=otsu)
    quality_annot = pd.read_csv(CSV_GLOBAL_CYST_ANNOT)
    df = df.merge(quality_annot)
    return df