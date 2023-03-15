# Cerebral organoid quantification in MRI

This repository quantifies cerebral organoids in MRI. It especially implements three tasks:
- Organoid segmentation
- Global cysticity classification
- Local cyst segmentation

These tasks are presented in the paper Cerebral organoid quantification in MRI.

For the implementation of the 3D U-Net, the full credit goes to Adrian Wolny (https://github.com/wolny/pytorch-3dunet).

## Prerequisites

* Operating system: Windows or Linux (tested on Ubuntu 18.04) **Adapt**
* [Anaconda Distribution](https://www.anaconda.com/distribution/#download-section) **Adapt**

## Installation
```
git clone https://github.com/deiluca/cerebral_organoid_quant_mri
```
Install requirements
```
cd path/to/cerebral_organoid_quant_mri
conda env create -f requirements.yml
```

Activate the conda environment:

```
conda activate co_quant_mri
```


## Data preparation

1. Download the data from x: Describe how to unpack exactly and where

2. Image extraction and data preparation
    ```
    python scripts/extract_and_prepare_images.py
    ```

## Organoid segmentation

1. Train and test 3D U-Net. [can be skipped: results from previous run are located in results/organoid_segmentation]
     ```
     python scripts/train_organoid_seg.py
     python scripts/test_organoid_seg.py
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
1. Train and test 3D U-Net. [can be skipped: results from previous run are located in results/local_cyst_segmentation]
     ```
     python scripts/train_local_cyst_seg.py
     python scripts/test_local_cyst_seg.py
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

**Can compacntess predict cysticity?**

Yes, high correlation. Extract and inspect results using [scripts/data_analysis.ipynb](scripts/data_analysis.ipynb)

<img src='results/local_cyst_segmentation/plots/correlation_compactness_cysticity.png'
     alt="Markdown Monster icon"
     style="float: center; margin-right: 10px; width:500px" />

Please note that repeated 3D U-Net training runs might lead to slightly different results. This is caused by random initialization of 3D U-Net weights.

If you find this useful, please consider citing our work: