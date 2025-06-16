
#%%
__author__ = 'jieqing jiao'
__contact__ = 'jieqing.jiao@gmail.com'

import numpy as np
import pandas as pd
import nibabel as nib
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from niftypad.tac import Ref
from niftypad.image_process.transform_image import load_4D
from niftypad.image_process.parametric_image import image_to_suvr_with_reference_tac
from niftypad.image_process.regions import extract_regional_values_image_data
from niftypad.image_process.slice_visualize import vol_heatmap, vol_crop


#%% file structure
wd = r"F:\Data\Image\NHP\NiftyPAD\M4\PostExp"
os.chdir(wd)

os.makedirs("raw", exist_ok=True)
raw_folder = os.path.join(wd, "raw")

os.makedirs("resample", exist_ok=True)
resample_folder = os.path.join(wd, "resample")

os.makedirs("vol-param", exist_ok=True)
vol_param_folder = os.path.join(wd, "vol-param")

os.makedirs("params", exist_ok=True)
param_folder = os.path.join(wd, "params")

os.makedirs("show-fig", exist_ok=True)
show_fig_folder = os.path.join(wd, "show-fig")

#%% load PET image
basename = r"M6-Postop-PET-DTBZ-CTAC-Dynamic"
pet_folder = os.path.join(resample_folder, basename)
img_data_pet, img_pet = load_4D(pet_folder) # img_data is tensor of shape (x, y, z, t), img is nibabel image


#%%
# dt

dt = np.array([[0, 30, 60, 90, 120, 150, 180, 240, 300, 360, 480, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300],
               [30, 60, 90, 120, 150, 180, 240, 300, 360, 480, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]])


#%% use the mask file to extract the ref region first then get the TAC
# ref mask file
ref_file = os.path.join(resample_folder, r'CerebellumMask.nii.gz')
img_ref = nib.load(ref_file)
img_data_ref = img_ref.get_fdata()

#%% get the ref TAC

regions_data, regions_label = extract_regional_values_image_data(img_data_pet, img_data_ref)

ref_data = regions_data[1] #regions_label=1
ref = Ref(ref_data, dt)

#%% get the image suvr

frame_index = np.arange(img_data_pet.shape[-1])


filename = os.path.basename(os.path.splitext(img_pet.get_filename())[0]).split('.',1)[0]
save_base = os.path.join(vol_param_folder, filename)

save_ext = '.nii.gz'


image_suvr = image_to_suvr_with_reference_tac(pet_image=img_data_pet, dt=dt, reference_regional_tac=ref.tac,
                                              selected_frame_index=frame_index[-2:])
nib.save(nib.Nifti1Image(image_suvr, img_pet.affine), save_base +
             '_suvr' + save_ext)


#%% read the suvr image

suvr_file = os.path.join(vol_param_folder, rf'{basename}_suvr.nii.gz')
img_sur_f = nib.load(suvr_file)
img_data_suvr_f = img_sur_f.get_fdata()

#%% set the paths of roi mask files

nigra_r_mask = os.path.join(wd, r'resample\NigraRmask.nii.gz')
nigra_l_mask = os.path.join(wd, r'resample\NigraLmask.nii.gz')

caudate_r_mask = os.path.join(wd, r'resample\CaudateRmask.nii.gz')
caudate_l_mask = os.path.join(wd, r'resample\CaudateLmask.nii.gz')

putamen_r_mask = os.path.join(wd, r'resample\PutamenRmask.nii.gz')
putamen_l_mask = os.path.join(wd, r'resample\PutamenLmask.nii.gz')

accumbens_r_mask = os.path.join(wd, r'resample\AccumbensRmask.nii.gz')
accumbens_l_mask = os.path.join(wd, r'resample\AccumbensLmask.nii.gz')

striatumdorsal_r_mask = os.path.join(wd, r'resample\StriatumDarsalRmask.nii.gz')
striatumdorsal_l_mask = os.path.join(wd, r'resample\StriatumDarsalLmask.nii.gz')

striatumdorsalNAc_r_mask = os.path.join(wd, r'resample\NigraStriatumNAcRmask.nii.gz')
striatumdorsalNAc_l_mask = os.path.join(wd, r'resample\NigraStriatumNAcLmask.nii.gz')

roi_list_files = [nigra_r_mask, nigra_l_mask, caudate_r_mask, caudate_l_mask, putamen_r_mask, putamen_l_mask,
                  accumbens_r_mask, accumbens_l_mask, striatumdorsal_r_mask, striatumdorsal_l_mask,
                  striatumdorsalNAc_r_mask, striatumdorsalNAc_l_mask]

# %% calculate the mean params of the ROIs
suvr_list = {}

for roi_f_file in roi_list_files: # f means final
    img_roi_f = nib.load(roi_f_file)
    img_data_roi_f = img_roi_f.get_fdata()
    regions_data, regions_label = extract_regional_values_image_data(img_data_suvr_f, img_data_roi_f)
    bp_data = regions_data[1] # regions_label=1
    bp_mean = np.mean(bp_data)

    roi_name = os.path.basename(roi_f_file)
    suvr_list[roi_name] = bp_mean        

#%% Save the params to csv files
df_suvr = pd.DataFrame.from_dict(suvr_list, orient="index", columns=["Value"])

# Save the DataFrame to CSV
df_savename = os.path.join(param_folder, 'suvr_list.csv')
df_suvr.to_csv(df_savename)

#%% adjust the param img 
# adjust the parametric images orientation in 3D slicer
# note: adjust the orientation of brain mask image to the same as the parametric images
# do not forget to crop after adjust the orientation
# for brain mask,  dilate 3 mm then guassian smooth 3 mm in 3D slicer
# save/copy the result to show-fig folder

# %% visualize the parametric images

img_bp_file = os.path.join(show_fig_folder, rf'{basename}_suvr.nii.gz')
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

# Visualize and save an axial slice (slice 50)
# nslice =[15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
nslice =np.arange(0, img_data_mask.shape[2], 5) # every 5 slices
for i in nslice:
    vol_heatmap(img_bp_data, nslice=i, orient='ax', mask=img_data_mask, 
            barlim=(0,6), colormap='viridis', save_base=save_base)
    
nslice = np.arange(0, img_data_mask.shape[1], 5) # every 5 slices
for i in nslice:
    vol_heatmap(img_bp_data, nslice=i, orient='cor', mask=img_data_mask,
            barlim=(0,6), colormap='viridis', save_base=save_base)
    
nslice = np.arange(0, img_data_mask.shape[0], 5) # every 5 slices
for i in nslice:
    vol_heatmap(img_bp_data, nslice=i, orient='sag', mask=img_data_mask,
            barlim=(0,6), colormap='viridis', save_base=save_base)
# %%
