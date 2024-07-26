# RhoA role in HSC aging
Repository hosting the source code developed for the paper *'Targeting RhoA activity rejuvenates aged hematopoietic stem cells by reducing nuclear stretching'*.

Here, we analize confocal microscopy images of Hematopoietic Stem Cells (HSCs) nuclei tainted with DAPI fluorescent marker. The raw data coming from a ZEISS microscope in `.czi` format is first pre-processed into `numpy`'s `.npz` format. Downstream analyses are carried on these files, which includes intensity by distance profiles and multidimensional analysis from extracted image features. The Jupyter Notebooks include all the necessary steps to replicate the analyses from the paper, along with the expected results.

The analyses from this part of the project relies on a colaboration between the Biomedical Data Science lab at ISGlobal, led by Paula Petrone, and the Stem Cell Aging at lab at IDIBELL led by Carolina Florian. Both centers are located in Barcelona, Spain.


## Source code

Python scripts:
- **methods/hsc_spatial_stats.py**: Contains main functions and utils used in the Jupyter Notebooks.
- **methods/czi_preprocessing.py**: Contains a script to pre-process `.czi` raw images and originate `.npz` processed matrices.

Jupyter Notebooks:
- **01_czi_exploration_and_outlier_detection.ipynb**: Contains a basic exploration of ZEISS `.czi` images format and metadata, along with outlier detection thresholds and plots. We use `AICSImageIO`, a Python library that facilitates working with microscopy images.
- **02_czi_preprocessing.ipynb**: Contains a call to `czi_preprocessing.py` to transform raw ZEISS `.czi` images to preprocessed `.npz` matrices.
- **03_inensity_profiles.ipynb**: Contains the analyses of intensity profiles as a function of distance from the segmented nuclear border.
- **04_extract_features.ipynb**: Contains the extraction of "manual" features from `.npz` matrices and the creation of a feature data table using `pandas`.
- **05_comparison_boxplots.ipynb**: Contains a series of boxplots showing differences in univariate features among biological conditions of HSCs.
- **06A_aged_young_umap_analysis_zscore**: Contains the identification of relevant features among aged and young HSCs via statistical significance and correlation analyses.
- **06B_aged_agedri_umap_analysis_zscore**: Contains the identification of relevant features among aged and aged treated with RhoA inhibitor HSCs via statistical significance and correlation analyses.
- **07_feature_umap_analysis_zscore.ipynb**: Contains the multidimensional UMAP analyses and K-Means clustering on the identified relevant features for the different biological conditions of HSCs.
- **08_feature_umap_analysis_zscore_MP.ipynb**: Contains the multidimensional UMAP analyses and K-Means clustering on the identified relevant features for HSCs and myeloid progenitor (MP) cell nuclei.


## Software Requirements

Reproducing the hosted Jupyter Notebooks requires only a standard PC with enough RAM to support the in-memory operations. The code was developed and tested in a Linux Operative System, with Ubuntu 22.04.4 version and 13th Gen Intel® Core™ i7 CPU with 16 GB of RAM. 

## Python Dependencies

The code was developed using a conda environment with `Python==3.11.4` the following Python packages and versions:

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
```
