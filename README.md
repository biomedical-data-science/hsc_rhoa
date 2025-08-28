# RhoA Role in HSC Aging
This repository hosts the source code developed for the imaging part of the paper *'Targeting RhoA activity rejuvenates aged hematopoietic stem cells by reducing nuclear stretching'*.

We analyze confocal microscopy images of Hematopoietic Stem Cell (HSC) nuclei stained with the DAPI fluorescent marker. The raw data, obtained from a ZEISS microscope in `.czi` format, is first pre-processed into `.npz` format using `numpy`. Subsequent analyses are performed on these files, including intensity by distance profiles and multidimensional analysis of extracted image features.

The project is a collaboration between the Biomedical Data Science lab at ISGlobal, led by Paula Petrone, and the Stem Cell Aging at lab at IDIBELL led by Carolina Florian. Both centers are located in Barcelona, Spain.

## Source data

The microscopy image data used in these analyses can be downloaded from [this link](https://doi.org/10.34810/data697). (Link will be made public after publication).

## Source code

Python scripts:
- **methods/hsc_spatial_stats.py**: Contains main functions and utils used in the Jupyter Notebooks.
- **methods/czi_preprocessing.py**: Pre-process `.czi` raw images to generate `.npz` processed matrices.
- **methods/seg_utils.py**: Contains util functions for the assessment and visualization of image segmentation.
- **methods/utils.py**: Contains short util functions used throughout the code.

Jupyter Notebooks:
- **01_czi_exploration_and_outlier_detection.ipynb**: Explores ZEISS `.czi` images format and metadata, includes outlier detection thresholds and plots. We use `AICSImageIO`, a Python library that facilitates working with microscopy images.
- **02_czi_preprocessing.ipynb**: Contains a call to `czi_preprocessing.py` to transform raw ZEISS `.czi` images into preprocessed `.npz` matrices.
- **03_inensity_profiles.ipynb**: Contains the analyses of intensity profiles as a function of distance from the segmented nuclear border.
- **04_extract_features.ipynb**: Contains the extraction of "manual" features from `.npz` matrices and the creation of a feature data table using `pandas`.
- **05_comparison_boxplots.ipynb**: Contains a series of boxplots showing differences in univariate features among biological conditions of HSCs.
- **06A_aged_young_umap_analysis**: Contains the identification of relevant features among aged and young HSCs via statistical significance and correlation analyses.
- **06B_aged_agedri_umap_analysis**: Contains the identification of relevant features among aged and aged treated with RhoA inhibitor HSCs via statistical significance and correlation analyses.
- **07_feature_umap_analysis.ipynb**: Contains multidimensional UMAP analyses and K-Means clustering on the identified relevant features for different biological conditions of HSCs.
- **07B_clustering_validation.ipynb**: Contains quality checks on the K-Means clustering.
- **08_feature_umap_analysis_MP.ipynb**: Contains multidimensional UMAP analyses and K-Means clustering on the identified relevant features for HSCs and myeloid progenitor (MP) cell nuclei.
- **09A_seg_manual_validation.ipynb**: Contains the comparison of automatic segmentation using Otsu vs. manual annotation on 3D nuclei of young, aged, and aged + Ri HSC nuclei images. Manual annotation was performed using `Napari`.
- **09B_seg_visualization.ipynb**: Contains visual comparisons of automatic vs. manual segmentation of HSC images.

## Installation and Use Guide

No installation is needed; simply download a copy of this repository and update the data folder paths according to your directory structure.

The Jupyter Notebooks contain all necessary steps to replicate the analyses from the paper, along with the expected results. Code execution is generally fast, except for `02_czi_preprocessing.ipynb` and `04_extract_features.ipynb`, which, depending on dataset size, might take a few hours of computing time on a standard CPU.

## Software Requirements

Reproducing the hosted Jupyter Notebooks requires only a standard PC with sufficient RAM to support the in-memory operations, no extra hardware is needed. The code was developed and tested on a Linux Operative System, specifically Ubuntu 22.04.4, with a 13th Gen Intel® Core™ i7 CPU with 16 GB of RAM. 

## Python Dependencies

The code was developed using a conda environment with `Python==3.11.4` and the following Python packages:

```
aicsimageio==4.9.2

matplotlib==3.7.1

numpy==1.23.5

pandas==1.5.3

scikit-image==0.20.0

scikit-learn==1.1.3

scipy==1.11.4

seaborn==0.12.2

statannotations==0.6.0

umap-learn==0.5.5

napari==0.6.1
```
