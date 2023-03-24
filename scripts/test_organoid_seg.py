import shutil
import yaml
from os.path import join as opj
import os
import sys
sys.path.insert(0, '/home/ws/oc9627/cerebral_organoid_quant_mri')
from scripts.utils.constants import WORKING_DIR, MRI_ORG_SEG_FILES_3DUNET  # noqa

os.chdir(WORKING_DIR)


def create_conf_files_test(create_yml=False):
    """Create 3D U-Net config files for organoid segmentation testing. One config file per LOOCV test set.

    Args:
        create_yml (bool, optional): whether to serialize config to disk. Defaults to False.

    Returns:
        list: 9 config locations
    """
    with open("pytorch3dunet/resources/general_test_config.yml", "r") as stream:
        try:
            ref_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    config_dir = 'pytorch3dunet/resources/3d_organoid_seg/test'
    config_locs = []

    for i in range(1, 10):
        ckp_dir = f'results/organoid_segmentation/checkpoint_dirs/org{i}'
        outdir = opj(ckp_dir, 'test_out')
        ref_config['model_path'] = opj(ckp_dir, 'best_checkpoint.pytorch')
        ref_config['loaders']['test']['file_paths'] = [
            f'{MRI_ORG_SEG_FILES_3DUNET}/LOO_org{i}/test']
        ref_config['loaders']['output_dir'] = outdir

        config_loc = opj(config_dir, f'LOO_org{i}.yml')
        if create_yml:
            with open(config_loc, 'w') as outfile:
                yaml.dump(ref_config, outfile, default_flow_style=False)
        config_locs.append(config_loc)

    return config_locs


def move_test_files_to_common_dir():
    """Goal: one directory containing all numpy files of predictions on all LOOCV test sets
    """
    outdir = 'results/organoid_segmentation/checkpoint_dirs/all_predictions_on_test_sets'
    os.makedirs(outdir, exist_ok=True)
    for i in range(1, 10):
        ckp_dir = f'results/organoid_segmentation/checkpoint_dirs/org{i}/test_out'
        for f in os.listdir(ckp_dir):
            if f.endswith('predictions.npy'):
                shutil.copyfile(opj(ckp_dir, f), opj(outdir, f))


def test_all_models(config_locs):
    """Test models of all LOOCV splits.

    Args:
        config_locs (list): list of all config files (one config file per LOOCV test set)
    """
    for i, cl in enumerate(config_locs):
        print('Training with config:', cl)
        os.system(f'python pytorch3dunet/predict.py --config {cl}')


if __name__ == '__main__':
    config_locs = create_conf_files_test(create_yml=True)
    test_all_models(config_locs)
    move_test_files_to_common_dir()
