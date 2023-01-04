# 1. create config_files
import yaml
from os.path import join as opj
import os


def create_conf_files_test(create_yml=False):
    with open("/home/ws/oc9627/pytorch-3dunet/resources/3d_organoid_seg/inference_all_samples_incl0707_new_model.yml", "r") as stream:
        try:
            ref_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    config_dir = '/home/ws/oc9627/pytorch-3dunet/resources/3d_cyst_seg/nested_cv_loo_34annot_test'
    os.makedirs(config_dir, exist_ok=True)
    config_locs = []

    for org in range(1, 10):
        ckp_dir = f'CHECKPOINT_DIRS_cyst_seg_34annot/org{org}'
        outdir = opj(ckp_dir, 'test_out_incl_raw_pred')
        ref_config['model_path'] = opj(ckp_dir, 'best_checkpoint.pytorch')
        ref_config['loaders']['test']['file_paths'] = [
            f'/home/ws/oc9627/Dokumente/MRT_segmentations_cysts/annotations_15_12_2022_34_orgs_with_cyst_sizes_greater_1000/h5_files/LOO_org{org}/test']
        ref_config['loaders']['output_dir'] = outdir

        config_loc = opj(config_dir, f'LOO_{org}.yml')
        if create_yml:
            with open(config_loc, 'w') as outfile:
                yaml.dump(ref_config, outfile, default_flow_style=False)
        config_locs.append(config_loc)

    return config_locs


def test_all_models(config_locs):
    for i, cl in enumerate(config_locs):
        print('Training with config:', cl)
        os.system(f'python pytorch3dunet/predict.py --config {cl}')


config_locs = create_conf_files_test(create_yml=True)
# config_locs = config_locs[:6]
# print(config_locs)
test_all_models(config_locs)
