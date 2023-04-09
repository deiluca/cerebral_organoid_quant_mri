# An AI-based segmentation and analysis pipeline for high-field MR monitoring of cerebral organoids
[![Pylint](https://github.com/deiluca/cerebral_organoid_quant_mri/actions/workflows/pylint.yml/badge.svg)](https://github.com/deiluca/cerebral_organoid_quant_mri/actions/workflows/pylint.yml)

This repository reproduces the results published in the paper [*An AI-based segmentation and analysis pipeline for high-field MR monitoring of cerebral organoids*](https://doi.org/10.1101/2023.04.07.535822) [1]

Specifically, it especially implements three tasks:
- Organoid segmentation
- Global cysticity classification
- Local cyst segmentation

For the implementation of the 3D U-Net, the full credit goes to Adrian Wolny (https://github.com/wolny/pytorch-3dunet).

## Prerequisites

* Operating system: Windows or Linux (tested on Ubuntu 20.04)
* [Install Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#installing-conda-on-a-system-that-has-other-python-installations-or-packages)

## Installation
```
git clone https://github.com/deiluca/cerebral_organoid_quant_mri
```
Install conda environment
```
cd path/to/cerebral_organoid_quant_mri
conda env create -f environment.yml
```

Activate the conda environment:

```
conda activate co_quant_mri
```

Add this line to ~/.bashrc to permanently add the repository to PYTHONPATH
```
export PYTHONPATH="${PYTHONPATH}:path/to/cerebral_organoid_quant_mri"
```


## Data preparation

1. Download the data from [Zenodo](https://zenodo.org/record/7805426) and unpack it in data/data_zenodo.

2. Image extraction and data preparation
    ```
    python scripts/extract_and_prepare_images.py
    ```

## Organoid segmentation

1. Train and test 3D U-Net. (can be skipped: checkpoints from previous run are located [here](results/organoid_segmentation/checkpoint_dirs_trained_previously))
     ```
     python scripts/train.py org_seg
     python scripts/test.py org_seg
     ```
2. Extract and inspect results using [scripts/data_analysis.ipynb](scripts/data_analysis.ipynb)

**Model performance** (Test Dice 0.92&#177;0.06 [mean&#177;SD])

<img src="results/organoid_segmentation/plots/organoid_seg_performance.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

**Example of segmentation performance (org7_0530)**

<img src='results/organoid_segmentation/plots/organoid_seg_overlay_org5_0530_pred-orange.png'
     alt="Markdown Monster icon"
     style="float: center; margin-right: 10px; width:500px" />

## Global cysticity classification
See [scripts/data_analysis.ipynb](scripts/data_analysis.ipynb)


**Performance of *Compactness* and examples of low- and high-quality organoids**

<img src='results/global_cyst_classification/compactness_separates_lq_hq_organoids.png'
     alt="Markdown Monster icon"
     style="float: center; margin-right: 10px; height:300px" />
<img src='results/global_cyst_classification/examples_lq_hq_organoids.png'
     alt="Markdown Monster icon"
     style="float: center; margin-right: 10px; height:300px" />

**DW-MRI: Higher diffusion of low-quality organoids**

<img src='results/global_cyst_classification/trace_mean_org_intensities.png'
     alt="Markdown Monster icon"
     style="float: center; margin-right: 10px; height:300px" />

## Local cyst segmentation
1. Train and test 3D U-Net. (can be skipped: checkpoints from previous run are located [here](results/local_cyst_segmentation/checkpoint_dirs_trained_previously))
     ```
     python scripts/train.py local_cyst_seg
     python scripts/test.py local_cyst_seg
     ```
2. Extract and inspect results using [scripts/data_analysis.ipynb](scripts/data_analysis.ipynb)

**Model performance** (Test Dice 0.63&#177;0.15 [mean&#177;SD])

<img src='results/local_cyst_segmentation/plots/local_cyst_seg_performance.png'
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;width=300px" />

**Example of segmentation performance (org7_0530)**

<img src='results/local_cyst_segmentation/plots/cyst_seg_overlay_org7_0530_pred-orange.png'
     alt="Markdown Monster icon"
     style="float: center; margin-right: 10px; width:500px" />

**Is compactness a predictor of organoid cysticity?**

Yes, high correlation. Extract and inspect results using [scripts/data_analysis.ipynb](scripts/data_analysis.ipynb)

<img src='results/local_cyst_segmentation/plots/correlation_compactness_cysticity.png'
     alt="Markdown Monster icon"
     style="float: center; margin-right: 10px; width:500px" />


Please note that repeated 3D U-Net training runs might lead to slightly different results. This is caused by random initialization of 3D U-Net weights.


**Cite**

If you find this useful, please consider citing our work:

[1] Deininger, L. et al. (2023) “An AI-based segmentation and analysis pipeline for high-field mr monitoring of cerebral organoids.” Available at: https://doi.org/10.1101/2023.04.07.535822.

``` 
@article{Deininger2023,
  doi = {10.1101/2023.04.07.535822},
  url = {https://doi.org/10.1101/2023.04.07.535822},
  year = {2023},
  month = apr,
  publisher = {Cold Spring Harbor Laboratory},
  author = {Luca Deininger and Sabine Jung-Klawitter and Petra Richter and Manuel Fischer and Kianush Karimian-Jazi and Michael Breckwoldt and Martin Bendszus and Sabine Heiland and Jens Kleesiek and Ralf Mikut and Daniel Huebschmann and Daniel Schwarz},
  title = {An {AI}-based segmentation and analysis pipeline for high-field {MR} monitoring of cerebral organoids}
}
```