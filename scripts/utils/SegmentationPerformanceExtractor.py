import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from os.path import join as opj

from scripts.utils.constants import CSV_ORG_OVERVIEW
from scripts.utils.metrics import get_dice


class SegmentationPerformanceExtractor(object):
    def __init__(self,
                 pred_dir='',
                 gt_dir=''):

        self.pred_dir = pred_dir
        self.gt_dir = gt_dir

    def extract_test_performance(self):
        org_ids, test_dices = [], []
        for i in range(1, 10):
            # get test performance
            dice_scores = self.get_dice_scores()
            for org_id, v in dice_scores.items():
                org_ids.append(org_id)
                test_dices.append(v['dice'])
        df = pd.DataFrame.from_dict(
            {'org_id': org_ids, 'Test Dice': test_dices})
        df = df.merge(pd.read_csv(CSV_ORG_OVERVIEW))
        self.df = df

    def print_test_dice_mean_sd(self):
        assert self.df is not None
        mean = self.df.drop_duplicates().mean()['Test Dice']
        sd = self.df.drop_duplicates().std()['Test Dice']
        print(f'Test Dice {mean:.2f}'+u"\u00B1" +
              f'{sd:.2f} (mean'+u"\u00B1"+'SD)')

    def get_dice_scores(self):

        all_pred = sorted([opj(self.pred_dir, x) for x in os.listdir(
            self.pred_dir) if x.endswith('predictions.npy')])
        all_gt = sorted([opj(self.gt_dir, x) for x in os.listdir(self.gt_dir) if x.endswith(
            '.npy') and f'{x.replace(".npy", "")}_predictions.npy' in os.listdir(self.pred_dir)])
        all_img_pairs = dict(zip(all_pred, all_gt))
        dice_scores = dict()
        for pred, gt in all_img_pairs.items():

            org = os.path.basename(pred)[:9]
            dice = get_dice(pred, gt)

            dice_scores[org] = dict()
            dice_scores[org]['dice'] = dice
            dice_scores[org]['pred_file'] = pred
            dice_scores[org]['gt_file'] = gt
        return dice_scores

    def plot_test_performance(self, save_to=''):
        sns.set_style("whitegrid")
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))

        # barplot
        # create copy of dataframe to display 'Overall' performance in plot
    #     df_copy=df.copy()
    #     df_copy['org_nr'] = 'Overall'
    #     df_bar = pd.concat([df, df_copy])

    #     sns.barplot(data=df_bar, x='org_nr', y='Test Dice', ax=axs[0])
    #     axs[0].set_ylim(0.0, 1.0)
    #     axs[0].set_xlabel('Organoid')
    #     sns.despine(top=True, left=True, bottom=True, right=True, ax=axs[0])

        # boxplot
        sns.boxplot(data=self.df, x='org_nr', y='Test Dice', ax=axs[0])
        axs[0].set_ylim(0.0, 1.0)
        axs[0].set_xlabel('Organoid')
        sns.despine(top=True, left=True, bottom=True, right=True, ax=axs[0])

        # lineplot
        self.df['org_nr'] = self.df['org_nr'].astype('str')
        sns.lineplot(data=self.df, x="day", y="Test Dice",
                     hue="org_nr", sort=True, ax=axs[1])
        axs[1].set_ylabel('Test Dice')
        axs[1].set_xlabel('Day')
        axs[1].set_ylim((0.0, 1.0))
        sns.despine(bottom=True, top=True, left=True, right=True, ax=axs[1])
        axs[1].legend(title='Organoid', loc='center left',
                      bbox_to_anchor=(1, 0.5))

    #     df['org_id_readable'] = 'Org ' + df['org_nr'].astype('str')+', Day '+df['day'].astype('str')
    #     sns.barplot(data=df, x='org_id_readable', y='Test Dice', color='grey', ax=axs[2])
    #     axs[2].tick_params(labelrotation=90)
    #     axs[2].set_xlabel('')
    #     axs[2].set_ylim(0.0, 1.0)

        plt.tight_layout()
        if save_to != '':
            plt.savefig(save_to, dpi=400)
