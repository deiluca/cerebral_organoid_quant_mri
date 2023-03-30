"""Train 3D U-Net"""
import sys
import os
from os.path import join as opj
import yaml

from scripts.utils.constants import WORKING_DIR, MRI_ORG_SEG_FILES_3DUNET, MRI_CYST_SEG_FILES_3DUNET  # noqa

os.chdir(WORKING_DIR)


def create_conf_files(kind, create_yml=False):
    """Create 3D U-Net config files for training. One config file per LOOCV test set.

    Args:
        create_yml (bool, optional): whether to serialize config to disk. Defaults to False.

    Returns:
        list: 8 config locations (organoid 9 is excluded because of too small cysts)
    """
    assert kind in ['org_seg', 'local_cyst_seg']
    if kind == 'org_seg':
        config_dir = 'pytorch-3dunet/resources/3d_organoid_seg/train'
        input_files = MRI_ORG_SEG_FILES_3DUNET
        max_org_idx = 10
        max_iterations = 2000
    else:
        config_dir = 'pytorch-3dunet/resources/3d_cyst_seg/train'
        input_files = MRI_CYST_SEG_FILES_3DUNET
        max_org_idx = 9
        max_iterations = 5000

    with open("pytorch3dunet/resources/general_train_config.yml", "r") as stream:
        try:
            ref_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    config_dir = 'pytorch3dunet/resources/3d_organoid_seg/train'
    os.makedirs(config_dir, exist_ok=True)
    config_locs = []
    for org in range(1, max_org_idx):
        ref_config['trainer'][
            'checkpoint_dir'] = f'results/organoid_segmentation/checkpoint_dirs/LOO_org{org}'
        ref_config['trainer']['max_num_iterations'] = max_iterations
        ref_config['loaders']['train']['file_paths'] = [
            f'{input_files}/LOO_org{org}/train']
        ref_config['loaders']['val']['file_paths'] = [
            f'{input_files}/LOO_org{org}/val']
        config_loc = opj(config_dir, f'LOO_org{org}.yml')
        config_locs.append(config_loc)
        if create_yml:
            with open(config_loc, 'w') as outfile:
                yaml.dump(ref_config, outfile, default_flow_style=False)
    return config_locs


def train_all_models(config_locs):
    """Train models of all LOOCV splits.

    Args:
        config_locs (list): list of all config files (one config file per LOOCV test set)
    """
    for i, cl in enumerate(config_locs):
        print('Training with config:', cl)
        os.system(f'python pytorch3dunet/train.py --config {cl}')


if __name__ == '__main__':
    kind = sys.argv[0]
    assert kind in ['org_seg', 'local_cyst_seg']
    config_locs = create_conf_files(kind=kind, create_yml=True)
    train_all_models(config_locs)
