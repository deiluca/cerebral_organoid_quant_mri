# Cerebral organoid quantification in MRI

This repository reproduces the results from the paper MRI

This repository quantifies cerebral organoids in MRI. It especially implements three tasks:
- organoid segmentation
- global cyst classification
- local cyst segmentation

## Prerequisites

* Operating system: Windows (tested on Windows 10) or Linux (tested on Ubuntu 18.04) **Adapt**
* [Anaconda Distribution](https://www.anaconda.com/distribution/#download-section) **Adapt**
* For GPU use: a CUDA capable GPU **Adapt**
* Minimum / recommended RAM/VRAM: 8 GiB / 16 GiB **Adapt**

## Installation
```
git clone https://github.com/deiluca/cerebral_organoid_quant_mri
```

```
cd path_to_your_cloned_repository
conda env create -f requirements.yml
```

Activate the co_quant_mri_ve:

```
conda activate co_quant_mri_ve
```

## Data preparation

1. Download the data from x: Describe how to unpack exactly and where

2. Image extraction and data preparation
    ```
    python scripts/extract_and_prepare_images.py
    ```

## Organoid segmentation

1. train and test 3D U-Net. [can be skipped: results from previous run are located in results/organoid_segmentation)]
2. inspect results using scripts/data_analysis.ipynb

**Model performance**

<img src="results/organoid_segmentation/plots/organoid_seg_performance.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

**Example of segmentation performance (org7_0530)**

<img src='results/organoid_segmentation/plots/organoid_seg_overlay_org5_0530.png'
     alt="Markdown Monster icon"
     style="float: center; margin-right: 10px; width:500px" />

## Global cyst classification
See scripts/data_analysis.ipynb

**Performance of *Compactness* and examples of low- and high-quality organoids**

<img src='results/global_cyst_classification/compactness_separates_lq_hq_organoids.png'
     alt="Markdown Monster icon"
     style="float: center; margin-right: 10px; height:300px" />
<img src='results/global_cyst_classification/examples_lq_hq_organoids.png'
     alt="Markdown Monster icon"
     style="float: center; margin-right: 10px; height:300px" />



## Local cyst segmentation
1. train and test 3D U-Net. [can be skipped: results from previous run are located in results/local_cyst_segmentation]
2. generate and inspect results using scripts/data_analysis.ipynb

**Model performance**

<img src='results/local_cyst_segmentation/plots/local_cyst_seg_performance.png'
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;width=300px" />

**Example of segmentation performance (org7_0530)**

<img src='results/local_cyst_segmentation/plots/cyst_seg_overlay_org7_0530.png'
     alt="Markdown Monster icon"
     style="float: center; margin-right: 10px; width:500px" />
     