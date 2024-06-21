# RhoA role in HSC aging
Repository hosting the source code developed for the paper *'Targeting RhoA activity rejuvenates aged hematopoietic stem cells by reducing nuclear stretching'*.

The analyses from this part of the project relies on a colaboration between the Biomedical Data Science lab at ISGlobal, led by Paula Petrone, and the Stem Cell Aging at lab at IDIBELL led by Carolina Florian.

Here, we analize microscope images of Hematopoietic Stem Cells (HSCs) nuclei tainted with DAPI fluorescent marker. The raw data coming from a ZEISS microscope in `.czi` format is first pre-processed into `numpy`'s `.npz` format. Downstream analyses are carried on these files, which includes intensity by distance profiles and multidimensional analysis from extracted image features.

## Source code

Python scripts:
- **hsc_spatial_stats.py**: Contains main functions and utils used in the Jupyter Notebooks.
- **czi_preprocessing.py**: Contains a script to pre-process `.czi` raw images and originate `.npz` processed matrices.

Jupyter Notebooks:
- **czi_exploration_and_outlier_detection.ipynb**: Contains a basic exploration of ZEISS `.czi` images format and metadata, along with outlier detection thresholds and plots. We use `AICSImageIO`, a Python library that facilitates working with microscopy images.
- **czi_preprocessing.ipynb**: Contains a call to `czi_preprocessing.py` to transform from raw ZEISS `.czi` images to preprocessed `.npz` matrices.
- **inensity_profiles.ipynb**: Contains the creation of intensity profiles as a function of distance from the segmented nuclear border.
- **extract_features.ipynb**: Contains the extraction of manual features from `.npz` matrices and the creation of a feature data table using `pandas`.
- **feature_pca_analysis.ipynb**: Contains the multidimensional PCA analyses.
- **feature_umap_analysis.ipynb**: Contains the multidimensional UMAP analyses and K-Means clustering.
- **young_aged_treated_plot.ipynb**: Contains code to generate comparison of distribution of each feature across Young, Aged and Treated + RhoA inhibition treatment HSCs.


## Dependencies

The code was developed in a local machine using a conda environment with the following packages and versions:

`seaborn==0.11.2`

`scipy==1.9.3`

`scikit-image==0.19.2`

`scikit-learn==1.1.3`

`pandas==1.5.1`

`numpy==1.22.3`

`matplotlib==3.5.3`

`aicsimageio==4.9.2`

`statannotations==0.5.0`

`umap-learn==0.5.3`
