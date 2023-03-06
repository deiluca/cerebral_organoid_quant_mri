
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as opj
import io
from PIL import Image, ImageDraw, ImageFont

from scripts.utils.metrics import dice


class SegmentationOverlayVisualizer(object):
    """
    Visualizes the original image, ground truth and prediction for one organoid
    """

    def __init__(self,
                 orig_imgs,
                 org_id,
                 org_loc,
                 pred_dir,
                 gt_dir,
                 dw_mri=False,
                 px_threshold=10,
                 planes='coronal',
                 size_per_org_x=5,
                 size_per_org_y=5,
                 keep_planes=None,
                 set_title=False,
                 mult_neg=True,
                 rot_img=0,
                 save_to='',
                 add_labels=True,
                 pred_color='green'):

        assert planes in ['coronal', 'axial', 'sagittal']
        assert pred_color in ['orange', 'green']

        gt_file = opj(gt_dir, f'{org_id}.npy')
        pred_file = opj(pred_dir, f'{org_id}_predictions.npy')

        self.gt = np.load(gt_file)
        self.pred = np.load(pred_file)

        self.orig_imgs = orig_imgs
        self.org_id = org_id
        self.orgnr = self.org_id[3]
        self.ol = org_loc
        self.dw_mri = dw_mri
        self.px_threshold = px_threshold
        self.planes = planes
        self.size_per_org_x = size_per_org_x
        self.size_per_org_y = size_per_org_y
        self.keep_planes = keep_planes
        self.set_title = set_title
        self.mult_neg = mult_neg
        self.rot_img = rot_img
        self.save_to = save_to
        self.add_labels = add_labels
        self.pred_color = pred_color

    def get_green_binary_colors(self, x):
        x = x[:, :, np.newaxis]
        x = np.where(x == 0.0, [1.0, 1.0, 1.0, 0.0], x)
        x = np.where(x == 1.0, [60/255, 179/255, 113/255, 0.6], x)
        return x

    def get_orange_binary_colors(self, x):
        x = x[:, :, np.newaxis]
        x = np.where(x == 0.0, [1.0, 1.0, 1.0, 0.0], x)
        x = np.where(x == 1.0, [245/255, 167/255, 66/255, 0.6], x)
        return x

    def get_indices_pred_gt_greater0(self):
        indices_dices = dict()
        for i in range(self.gt.shape[-1]):
            idx_valid = False
            if np.count_nonzero(self.gt[:, :, i] > 0) > self.px_threshold:
                idx_valid = True
            elif np.count_nonzero(self.pred[:, :, i] > 0) > self.px_threshold:
                idx_valid = True
            if idx_valid:
                indices_dices[i] = dice(self.pred[:, :, i], self.gt[:, :, i])

        return indices_dices

    def get_indices_pred_gt_greater0_axial(self):
        indices_dices = dict()
        for i in range(self.gt.shape[-3]):
            idx_valid = False
            if np.count_nonzero(self.gt[i, :, :] > 0) > self.px_threshold:
                idx_valid = True
            elif np.count_nonzero(self.pred[i, :, :] > 0) > self.px_threshold:
                idx_valid = True
            if idx_valid:
                indices_dices[i] = dice(self.pred[i, :, :], self.gt[i, :, :])

        return indices_dices

    def get_indices_pred_gt_greater0_sagittal(self):
        indices_dices = dict()
        for i in range(self.gt.shape[-2]):
            idx_valid = False
            if np.count_nonzero(self.gt[:, i, :] > 0) > self.px_threshold:
                idx_valid = True
            elif np.count_nonzero(self.pred[:, i, :] > 0) > self.px_threshold:
                idx_valid = True
            if idx_valid:
                indices_dices[i] = dice(self.pred[:, i, :], self.gt[:, i, :])
        return indices_dices

    def plot_segmentation_overlays(self):
        img_buf = io.BytesIO()
        ol = self.ol
    #     print(gt.shape, pred.shape)
        if self.planes == 'coronal':
            indices_dices = self.get_indices_pred_gt_greater0()
        elif self.planes == 'sagittal':
            indices_dices = self.get_indices_pred_gt_greater0_sagittal()
        else:
            indices_dices = self.get_indices_pred_gt_greater0_axial()
        nr_rows = 3
        if self.keep_planes is not None:
            indices_dices = {
                k: v for k, v in indices_dices.items() if k in self.keep_planes}
        fig, axs = plt.subplots(len(indices_dices), nr_rows, figsize=(
            self.size_per_org_x*nr_rows, self.size_per_org_y*len(indices_dices)), facecolor='white')
        plane_numbers = []
    #     print(len(indices_dices), indices_dices)
        for j, i in enumerate(sorted(list(indices_dices.keys()), reverse=True)):
            plane_numbers.append(i)
            if self.planes == 'coronal':
                gt_i = self.gt[ol[2]:ol[3], ol[0]:ol[1], i]
                pred_i = self.pred[ol[2]:ol[3], ol[0]:ol[1], i]
            elif self.planes == 'sagittal':
                gt_i = self.gt[ol[2]:ol[3], i, ol[0]:ol[1]]
                pred_i = self.pred[ol[2]:ol[3], i, ol[0]:ol[1]]
            else:
                gt_i = self.gt[i, ol[2]:ol[3], ol[0]:ol[1]]
                pred_i = self.pred[i, ol[2]:ol[3], ol[0]:ol[1]]
            if self.dw_mri:
                orig_img = np.moveaxis(self.orig_imgs[self.org_id], 0, -1)
                if self.planes == 'coronal':
                    img_i = orig_img[ol[2]:ol[3], ol[0]:ol[1], i]
                elif self.planes == 'sagittal':
                    img_i = orig_img[ol[2]:ol[3], i, ol[0]:ol[1]]
                else:
                    img_i = orig_img[i, ol[2]:ol[3], ol[0]:ol[1]]

                axs[j, 0].imshow(
                    img_i*-1 if self.mult_neg else img_i, cmap='Greys')
                axs[j, 1].imshow(
                    img_i*-1 if self.mult_neg else img_i, cmap='Greys')
                axs[j, 2].imshow(
                    img_i*-1 if self.mult_neg else img_i, cmap='Greys')
            else:
                if self.planes == 'coronal':
                    img_i = self.orig_imgs[self.org_id][ol[2]
                        :ol[3], ol[0]:ol[1], i]
                elif self.planes == 'sagittal':
                    img_i = self.orig_imgs[self.org_id][ol[2]:ol[3], i, ol[0]:ol[1]]
                else:
                    img_i = self.orig_imgs[self.org_id][i,
                                                        ol[2]:ol[3], ol[0]:ol[1]]

                axs[j, 0].imshow(
                    img_i*-1 if self.mult_neg else img_i, cmap='Greys')
                axs[j, 1].imshow(
                    img_i*-1 if self.mult_neg else img_i, cmap='Greys')
                axs[j, 2].imshow(
                    img_i*-1 if self.mult_neg else img_i, cmap='Greys')

            axs[j, 1].imshow(self.get_green_binary_colors(
                gt_i), cmap='Set2', alpha=0.5)
            if self.pred_color == 'green':
                axs[j, 2].imshow(self.get_green_binary_colors(
                    pred_i), cmap='Set2', alpha=0.5)
            else:
                axs[j, 2].imshow(self.get_orange_binary_colors(
                    pred_i), cmap='Set2', alpha=0.5)
            axs[j, 0].axis('off')
            axs[j, 1].axis('off')
            axs[j, 2].axis('off')
            if self.set_title:
                axs[j, 0].set_title(f'plane {i}')

        fig.tight_layout()
        plt.tight_layout()
        plt.savefig(img_buf, format='png')
        plt.close()

        if self.rot_img != 0:
            im = Image.open(img_buf)
            im = im.rotate(self.rot_img)
            img_buf.close()
            img_buf2 = io.BytesIO()
            im.save(img_buf2, format='png')
            img_buf = img_buf2
        im = Image.open(img_buf)
        if self.add_labels:
            im = self.add_overlay_labels(im)
        if self.save_to != '':
            im.save(self.save_to)
        return im

    def add_overlay_labels(self, img):

        tim = Image.new('RGBA', (img.height, img.width), (0, 0, 0, 0))
        label_size = {'Prediction': 5, 'GT': 1, 'Image': 2}
        for i, text in enumerate(['Prediction', 'GT', 'Image']):
            dr = ImageDraw.Draw(tim)
            ft = ImageFont.truetype('scripts/utils/arial.ttf', 35)
            xpos = int((img.height/3)*(i+1) - (img.height/3) /
                       2 - label_size[text]*10)
            dr.text((xpos, 20), text, fill=(0, 0, 0), font=ft)

        tim = tim.rotate(90,  expand=1)

        img.paste(tim, (0, 0), tim)
        return img
