#%%
__author__ = 'jieqing jiao'
__contact__ = 'jieqing.jiao@gmail.com'

# Ref: doi: 10.1007/s12021-022-09616-0
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import numpy as np
import pandas as pd
import nibabel as nib
import subprocess
from niftypad.tac import Ref
from niftypad import basis
from niftypad.image_process.parametric_image import image_to_parametric
from niftypad.models import get_model_inputs
from niftypad.image_process.regions import extract_regional_values_image_data
from niftypad.image_process.transform_image import dicom4_to_nifti3, load_4D
import niftypad.linearity as li
from niftypad.image_process.slice_visualize import vol_heatmap, vol_crop

#%% file structure
wd = r"F:\Data\Image\NHP\NiftyPAD\M6\Postop"

pet_basename = r"M6-Postop-PET-DTBZ-CTAC-Dynamic"


os.makedirs(wd, exist_ok=True)
os.chdir(wd)

os.makedirs("raw", exist_ok=True)
raw_folder = os.path.join(wd, "raw")

os.makedirs("reg", exist_ok=True)
reg_folder = os.path.join(wd, "reg")

os.makedirs("mask", exist_ok=True)
mask_folder = os.path.join(wd, "mask")

os.makedirs("resample", exist_ok=True)
resample_folder = os.path.join(wd, "resample")

os.makedirs("tac-ref", exist_ok=True)
tac_ref_folder = os.path.join(wd, "tac-ref")

os.makedirs("tac-roi", exist_ok=True)
tac_roi_folder = os.path.join(wd, "tac-roi")

os.makedirs("linearity", exist_ok=True)
linearity = os.path.join(wd, "linearity")

os.makedirs("vol-param", exist_ok=True)
vol_param_folder = os.path.join(wd, "vol-param")

os.makedirs("vol-param-pre", exist_ok=True)
vol_param_pre_folder = os.path.join(wd, "vol-param-pre")

os.makedirs("params", exist_ok=True)
param_folder = os.path.join(wd, "params")

os.makedirs("show-fig", exist_ok=True)
show_fig_folder = os.path.join(wd, "show-fig")

# %% visualize the parametric images

img_bp_file = os.path.join(show_fig_folder, rf'{pet_basename}_mrtm_k2p_bp.nii.gz')
img_bp = nib.load(img_bp_file)
img_bp_data = img_bp.get_fdata()
mask_file = os.path.join(show_fig_folder, r'BrainMask_dilate.nii.gz')
img_mask = nib.load(mask_file)
img_data_mask = img_mask.get_fdata()
img_bp_file.split('.',1)[0]
save_base = os.path.join(show_fig_folder, os.path.basename(img_bp_file.split('.',1)[0]))

# crop the volume to the smallest bounding box
img_data_mask, crop_indices = vol_crop(img_data_mask)
img_bp_data = img_bp_data[crop_indices]

barlim = (0, 5) # set the color bar limit for visualization

# Visualize and save an axial slice (slice 50)
# nslice =[15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
nslice =np.arange(0, img_data_mask.shape[2], 5) # every 5 slices
for i in nslice:
    vol_heatmap(img_bp_data, nslice=i, orient='ax', mask=img_data_mask, 
            barlim=barlim, colormap='viridis', save_base=save_base)
    
nslice = np.arange(0, img_data_mask.shape[1], 5) # every 5 slices
for i in nslice:
    vol_heatmap(img_bp_data, nslice=i, orient='cor', mask=img_data_mask,
            barlim=barlim, colormap='viridis', save_base=save_base)
    
nslice = np.arange(0, img_data_mask.shape[0], 5) # every 5 slices
for i in nslice:
    vol_heatmap(img_bp_data, nslice=i, orient='sag', mask=img_data_mask,
            barlim=barlim, colormap='viridis', save_base=save_base)


# %%
img_bp_file = os.path.join(show_fig_folder, rf'{pet_basename}_suvr.nii.gz')
img_bp = nib.load(img_bp_file)
img_bp_data = img_bp.get_fdata()
mask_file = os.path.join(show_fig_folder, r'BrainMask_dilate.nii.gz')
img_mask = nib.load(mask_file)
img_data_mask = img_mask.get_fdata()
img_bp_file.split('.',1)[0]
save_base = os.path.join(show_fig_folder, os.path.basename(img_bp_file.split('.',1)[0]))

# crop the volume to the smallest bounding box
img_data_mask, crop_indices = vol_crop(img_data_mask)
img_bp_data = img_bp_data[crop_indices]

barlim = (0, 5) # set the color bar limit for visualization

# Visualize and save an axial slice (slice 50)
# nslice =[15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
nslice =np.arange(0, img_data_mask.shape[2], 5) # every 5 slices
for i in nslice:
    vol_heatmap(img_bp_data, nslice=i, orient='ax', mask=img_data_mask, 
            barlim=barlim, colormap='viridis', save_base=save_base)
    
nslice = np.arange(0, img_data_mask.shape[1], 5) # every 5 slices
for i in nslice:
    vol_heatmap(img_bp_data, nslice=i, orient='cor', mask=img_data_mask,
            barlim=barlim, colormap='viridis', save_base=save_base)
    
nslice = np.arange(0, img_data_mask.shape[0], 5) # every 5 slices
for i in nslice:
    vol_heatmap(img_bp_data, nslice=i, orient='sag', mask=img_data_mask,
            barlim=barlim, colormap='viridis', save_base=save_base)
# %%
