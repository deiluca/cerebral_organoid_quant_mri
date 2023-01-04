# 1. create config_files
import yaml
from os.path import join as opj
import os

def create_conf_files(create_yml=False):
    with open("../pytorch3dunet/resources/general_train_config.yml", "r") as stream:
        try:
            ref_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    config_dir = '/home/ws/oc9627/pytorch-3dunet/resources/3d_cyst_seg/nested_cv_loo_34annot'
    os.makedirs(config_dir, exist_ok=True)
    print(ref_config)
    config_locs = []
    for org in range(1, 10):
        ref_config['trainer']['checkpoint_dir'] = f'CHECKPOINT_DIRS_cyst_seg_34annot/org{org}'
        ref_config['trainer']['max_num_iterations'] = 5000
        ref_config['loaders']['train']['file_paths'] = [f'/home/ws/oc9627/Dokumente/MRT_segmentations_cysts/annotations_15_12_2022_34_orgs_with_cyst_sizes_greater_1000/h5_files/LOO_org{org}/train']
        ref_config['loaders']['val']['file_paths'] = [f'/home/ws/oc9627/Dokumente/MRT_segmentations_cysts/annotations_15_12_2022_34_orgs_with_cyst_sizes_greater_1000/h5_files/LOO_org{org}//val']
        config_loc = opj(config_dir, f'LOO_org{org}.yml')

        config_locs.append(config_loc)
        if create_yml:
            with open(config_loc, 'w') as outfile:
                yaml.dump(ref_config, outfile, default_flow_style=False)
    return config_locs


def train_all_models(config_locs):
    for i, cl in enumerate(config_locs):
        print('Training with config:', cl)
        os.system(f'python pytorch3dunet/train.py --config {cl}')

config_locs = create_conf_files(create_yml=True)
train_all_models(config_locs)