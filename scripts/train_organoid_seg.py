import yaml
from os.path import join as opj
import os
import sys
sys.path.insert(0, '/home/ws/oc9627/cerebral_organoid_quant_mri')
from scripts.utils.constants import WORKING_DIR, MRI_ORG_SEG_FILES_3DUNET  # noqa

os.chdir(WORKING_DIR)


def create_conf_files(create_yml=False):
    """Create 3D U-Net config files for organoid segmentation training. One config file per LOOCV test set.

    Args:
        create_yml (bool, optional): whether to serialize config to disk. Defaults to False.

    Returns:
        list: 8 config locations (organoid 9 is excluded because of too small cysts)
    """
    with open("pytorch3dunet/resources/general_train_config.yml", "r") as stream:
        try:
            ref_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    config_dir = 'pytorch3dunet/resources/3d_organoid_seg/train'
    os.makedirs(config_dir, exist_ok=True)
    config_locs = []
    for i in range(1, 10):
        ref_config['trainer'][
            'checkpoint_dir'] = f'results/organoid_segmentation/checkpoint_dirs/LOO_org{i}'
        ref_config['trainer']['max_num_iterations'] = 2000
        ref_config['loaders']['train']['file_paths'] = [
            f'{MRI_ORG_SEG_FILES_3DUNET}/LOO_org{i}/train']
        ref_config['loaders']['val']['file_paths'] = [
            f'{MRI_ORG_SEG_FILES_3DUNET}/LOO_org{i}/val']
        config_loc = opj(config_dir, f'LOO_org{i}.yml')
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
    config_locs = create_conf_files(create_yml=True)
    train_all_models(config_locs)
