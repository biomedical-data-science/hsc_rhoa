# # Notebook to load CZI microscope images, 
# # preprocess them and save them as NPZ

import os
from glob import glob
import pandas as pd

#loads necessary code functions from hsc_spatial_stats.pys
#all documentation for algorithms are found in the python code
from hsc_spatial_stats import *


## Choose the new resolution and normalization for CZI preprocessing
RESOLUTION = 0.05
normalization = "none"

# Biological conditions of HSC to consider
conditions = ['old', 'treated_RhoAi', 'treated_NaB, 'compressed_8um',
             'compressed_3um', 'RhoA_KO', 'young', 'compressed_5um']

#containing folder with .czi images inside
outliers = pd.read_csv("hsc_raw_data/czi_image_outliers.csv")['path'].values
input_dir = 'hsc_raw_data'
output_dir = f'segmented_nucleus/res_{RESOLUTION}_{normalization}_norm'

hist_df = pd.DataFrame()

# Create output directory if it does not exist already
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for cond in conditions:
    # Create condition folder if it does not exist already    
    output_dir = f'segmented_nucleus/res_{RESOLUTION}_{normalization}_norm/{cond}'
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    img_list = glob(f'{input_dir}/{cond}/**/*.czi', recursive=True)

    # For each image within this condition
    for i, img_path in enumerate(img_list, 1):

        out_path = get_output_filename_with_subfolder(
            img_path, output_dir, suffix='XY')

        # If this cell was already processed - skip
        if os.path.exists(out_path):
            continue

        # skip outliers
        if img_path in outliers:
            print('Outlier identified - skipping')
            continue

        # Log information to the user
        print(img_path)
        print("Image Number : "+ str(i))
        
        #DAPI signal usually is named "Ch1-T3" 
        img, nuc_mask = czi_image_preprocessing(
            img_path, output_dir, channel_name='Ch1-T3', new_res=RESOLUTION, 
            resize=True, normalization=normalization)

        if img=='':
        #if missing then search for "Ch1-T2" channel
            img, nuc_mask = czi_image_preprocessing(
                img_path, output_dir, channel_name='Ch1-T2', new_res=RESOLUTION, 
                resize=True, normalization=normalization)
                
            if img=='':
                continue

