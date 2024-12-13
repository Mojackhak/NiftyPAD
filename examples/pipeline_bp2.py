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
from niftypad.image_process.slice_visualize import vol_heatmap

#%% file structure
wd = r"F:\Data\Image\NHP\NiftyPAD\M4\Postop2"
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

# #%% Paths provided in the query
# dicom_folder = r"E:\Chen Lab\Data\NHP\Image\M4\postmodel\200165_PET\200165-M4_preop_PET_2024-09-08\PET-CT\4660"
# output_file = r"F:\Data\Image\NHP\NiftyPAD\M4\raw-file\M4-Postop-PET-DTBZ-CTAC-Dynamic"

# #%% Call the function with the provided paths
# pet_file = convert_dicom_to_nifti_conda(dicom_folder, output_file)


#%% 
# extract reference region 
# use lead dbs to register the atlas to the image
# do not use lead dbs to register the PET images to correct motion, the registration is not accurate!
# use 3d slicer to create the mask

#%% Split the 4D PET image into 3D images
# note that direct save the 4D image to nii file will cause the image orientation inconsistent with raw dicom image (do not know why)
# have try pkg: pydicom, dcm2niix, SimpleITK all failed
dicom_folder = r"E:\Chen Lab\Data\NHP\Image\M6\200169-M6_preop_PET_20241208\200169\20241208000682\9212"
output_base = os.path.join(raw_folder, r'M6-Postop-PET-DTBZ-CTAC-Dynamic')
dicom4_to_nifti3(dicom_folder, output_base)

#%% transform, crop and resample and crop image to the same size in 3D slicer

# Transform the atlas and mask to anchor space.
# note: first resample to the same resolution, then crop to the same size
# note: resample well make the images size consistent
# note: the interpolation method for mask should be the nearest neighbor
# note: the interpolation method for image should be the linear

#%%
# img file
# file_path = '/Users/Himiko/data/amsterdam_data/'
# pet_file = file_path + 'E301_FLUT_AC1_combined_ID_cleared.nii'
# img = nib.load(pet_file)
# pet_image = img.get_fdata()

# pet_folder = r"F:\Data\Image\NHP\NiftyPAD\M4\resample\M4-Postop-PET-DTBZ-CTAC-Dynamic"
pet_folder = os.path.join(resample_folder, r'M4-Postop-PET-DTBZ-CTAC-Dynamic')
img_data_pet, img_pet = load_4D(pet_folder) # img_data is tensor of shape (x, y, z, t), img is nibabel image


#%%
# dt indicates the start and end times of each frame in dynamic PET imaging. It is defined as a NumPy array structured as:
# dt = np.array([[start_time seqs], [end_time seqs]])
# dt = np.array([[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 120, 180, 240, 300, 450, 600, 900, 1200, 1800, 2400, 3000, 5396,
#                 5696, 5996, 6296],
#                [5, 10, 15, 20, 25, 30, 40, 50, 60, 120, 180, 240, 300, 450, 600, 900, 1200, 1800, 2400, 3000, 3600,
#                 5696, 5996, 6296, 6596]])

dt = np.array([[0, 30, 60, 90, 120, 150, 180, 240, 300, 360, 480, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300],
               [30, 60, 90, 120, 150, 180, 240, 300, 360, 480, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]])

#%% Way1: directly input ref TAC
# ref represents the time-activity curve (TAC) of the reference region in dynamic PET imaging. 
# It is an array where each value corresponds to the TAC value for a specific frame.
# cer_right_GM = np.array([0, 0, 0, 0, 592.221900000000, 2487.12200000000, 4458.12800000000, 5239.16900000000, 5655.47800000000, 6740.88200000000, 7361.56300000000, 7315.28400000000, 7499.59700000000, 7067.78900000000, 6663.79200000000, 5921.15200000000, 5184.79900000000, 4268.11900000000, 3431.98000000000, 2886.08500000000, 2421.19200000000, 1687.55500000000, 1538.81800000000, 1440.42100000000, 1439.46900000000])
# cer_left_GM = np.array([0, 0, 0, 0, 915.895900000000, 3751.55300000000, 5377.27800000000, 5896.48700000000, 6752.62900000000, 7299.80200000000, 7566.03600000000, 7440.07100000000, 7539.30500000000, 7271.21300000000, 6646.04300000000, 6109.30200000000, 5246.95700000000, 4447.90400000000, 3464.82700000000, 2863.01500000000, 2445.84400000000, 1658.95300000000, 1525.59300000000, 1382.73300000000, 1363.93900000000])
# cer_GM = cer_right_GM/2 + cer_left_GM/2
# ref = Ref(cer_GM, dt)
# ref.interp_1cubic()
# ref.run_feng_srtm()


#%% Way2: use the mask file to extract the ref region first then get the TAC
# ref mask file
ref_file = os.path.join(resample_folder, r'CerebellumMask.nii.gz')
img_ref = nib.load(ref_file)
img_data_ref = img_ref.get_fdata()

#%% get the ref TAC

regions_data, regions_label = extract_regional_values_image_data(img_data_pet, img_data_ref)

ref_data = regions_data[1] #regions_label=1
ref = Ref(ref_data, dt)

#%% Interpolate the TAC (interplate to 1 second interval)

ref.interp_1()
ref.interp_1cubic()
ref.run_exp1()
ref.run_exp2()
ref.run_exp_am()
ref.run_feng_srtm() # may encounter overflow error, not recommended currently

#%% Visualize the ref TAC

# self.inputf1_feng_srtm = []
# self.inputf1 = []
# self.inputf1cubic = []
# self.inputf1_exp1 = []
# self.inputf1_exp2 = []
# self.inputf1_exp_am = []
# self.input_interp_method = []
# self.input_interp_1 = []


ref.plot_tac('interp_1', tac_ref_folder)
ref.plot_tac('interp_1cubic', tac_ref_folder)
ref.plot_tac('exp1', tac_ref_folder)
ref.plot_tac('exp2', tac_ref_folder)
ref.plot_tac('exp_am', tac_ref_folder)
ref.plot_tac('feng_srtm', tac_ref_folder)

#%%
# choose the best interpolation method!
input_ref = ref.inputf1_exp_am

#%% Read the ROI mask file
roi_file = os.path.join(resample_folder, r'PutamenRmask.nii.gz')
img_roi = nib.load(roi_file)
img_data_roi = img_roi.get_fdata()
mask_roi = np.argwhere(img_data_roi > 0)

#%% get the TAC of the ROI

regions_data, regions_label = extract_regional_values_image_data(img_data_pet, img_data_roi)

roi_data = regions_data[1] #regions_label=1
roi = Ref(roi_data, dt)

#%% Interpolate the TAC (interplate to 1 second interval)

roi.interp_1()
roi.interp_1cubic()
roi.run_exp1()
roi.run_exp2()
roi.run_exp_am()
roi.run_feng_srtm() # may encounter overflow error, not recommended currently

#%% Visualize the ROI TAC

roi.plot_tac('interp_1', tac_roi_folder)
roi.plot_tac('interp_1cubic', tac_roi_folder)
roi.plot_tac('exp1', tac_roi_folder)
roi.plot_tac('exp2', tac_roi_folder)
roi.plot_tac('exp_am', tac_roi_folder)
roi.plot_tac('feng_srtm', tac_roi_folder)

#%%
# choose the best interpolation method!
input_roi = roi.inputf1_feng_srtm

#%% judge the linear phase of models
t = np.arange(0, len(input_roi))
# t_trunc to truncate the initial rising part of the TAC
li.logan_linear_phase(input_roi, input_ref, t, t_trunc=(200, np.inf), save_dir=linearity)
li.mrtm_linear_phase(input_roi, input_ref, t, t_trunc=(200, np.inf), save_dir=linearity)


#%% set the start and end time of the linear phase of the Logan plot
start_l = 700
end_l = None

start_l2 = 1000
end_l2 = None

start_m = 800
end_m = 3200

start_m2 = 1500
end_m2 = None


#%%
# k2p is the tissue-to-plasma clearance 𝑘2′ of the reference tissue of the parent radioligand in dynamic PET imaging (𝑘′2) .
# pre-defined in SRTM and SRTM2 nonlinear models
# k2p * Ct(t) = dCt(t)/dt. Ct(t) is the concentration of the parent radioligand in the tissue at time t.
# REF: http://www.turkupetcentre.net/petanalysis/model_compartmental_ref.html#SRTM
# Coupling of the k'2 (k2 in the reference region) to a common value across brain regions, 
# or to a first-pass estimate (as proposed for the BF implementations), 
# reduces the variance of parameter estimates (Wu & Carson, 2002; Ichise et al., 2003; Endres et al., 2011).

# REF: https://doi.org/10.1097/01.WCB.0000085441.37552.CA. This paper provides a way to estimate k2p.
# Run MRTM or SRTM to ROI TACs with the reference region TAC to get the k2p value.
# Averaging of several k′2 samples from high-BP ROIs might be required to minimize the variability of k′2 estimation. 

k2p = 0.001
r1 = 1.5

#%%
# basis functions are very important for accuracy of the model
# REF: http://www.turkupetcentre.net/petanalysis/model_compartmental_ref.html#SRTM
# GPT4: Basis functions in dynamic PET analysis represent complex time-activity curves (TACs) 
# as linear combinations of simpler, predefined functions, enabling efficient fitting and parameter 
# estimation by capturing the underlying kinetics with reduced computational complexity.
# Basis functions have a set of unknown parameters, and by fitting them to the known TAC values, 
# the optimal parameters are determined. These optimal values are then used to calculate kinetic 
# parameters like BP (Binding Potential), k1, and k2.
# The basis functions represent the time-dependent behavior of the tracer in the tissue (via exponential decays), 
# and they are used to fit kinetic models to the dynamic PET data.



# For make_basis parameters:
# inputf1: the input function of the radioligand in the plasma
# dt: the start and end times of each frame in dynamic PET imaging
# beta_lim: the range of 𝛽 values to be used in the basis functions
# beta is the decay constant of the basis functions. It is the reciprocal of the half-life of the basis functions.
# beta = ln(2)/T1/2. The smaller beta mean slower decay of the basis functions.
# so the range of beta_lim should be set based on the tracer's half-life.
# for example, tracer like 18F-DTBZ has a half-life of 110 minutes (10.3389/fnagi.2022.931015), 
# so beta=0.0063 min-1,  beta_lim can be set as [0.0063*0.1, 0.0063*10] min-1
# n_beta: the number of 𝛽 values to be used in the basis function
# GPT: Multiple basis functions are used in dynamic PET modeling to represent the underlying kinetic 
# processes as a linear combination. Each basis function, typically an exponential decay, 
# is associated with a parameter (β) that controls its decay rate. The fitting process involves 
# finding the optimal coefficients for these basis functions by minimizing the difference 
# between the model’s prediction and the observed Time-Activity Curves (TACs). 
# Once the best-fitting parameters (coefficients) are obtained, 
# they can be used to derive kinetic parameters such as the binding potential (BP), 
# rate constants (k1, k2), and other biological measures based on the model's equations.
# So with larger n_beta, the model can capture more complex kinetics of the tracer in the tissue but with more computational cost.
# beta_space: default is 'log', can be 'natural'. 
# If 'log', the basis functions are created with a logarithmic spacing of 𝛽 values. Recommended for most cases.
# If 'natural', the basis functions are created with a natural spacing of 𝛽 values.
# w: the weights of the basis functions. Default is None.
# k2p: 𝑘′2. Set when the model need pre-defined 𝑘′2. default is None. 
# The b contain the basis functions without pre-defined 𝑘′2 when 𝑘′2 is given.
# basis structure:     
# basis = {'beta': beta, 'basis': basis, 'basis_w': basis_w, 'm': m, 
# 'm_w': m_w, 'basis_k2p': basis_k2p,'input': input, 'w': w, 'k2p': k2p, 'dt': dt}
# fig: whether to plot the basis functions. Default is False.


t1_2 = 110 # half-life of the tracer in minutes
beta = np.log(2)/t1_2
beta_lim = [beta*0.1/60, beta*10/60] # convert to s-1
n_beta = 128
b = basis.make_basis(input_ref, dt, beta_lim=beta_lim, n_beta=n_beta, w=None, k2p=k2p, fig=True)

#%%
# provide all user inputs in one dict here and later 'get_model_inputs' will select the needed ones


user_inputs = {'dt': dt,
               'inputf1': input_ref,
               'w': None,
               'r1': r1, # influx rate 𝑅1
               'k2p': k2p,
               'beta_lim': beta_lim,
               'n_beta': n_beta,
               'b': b, # basis functions
               'linear_phase_start_l': start_l, # logan_ref linear phase start of TAC
               'linear_phase_end_l': end_l,  # logan_ref linear phase end of TAC
               'linear_phase_start_l2': start_l2, # logan_ref_k2p linear phase start of TAC
               'linear_phase_end_l2': end_l2, # logan_ref_k2p linear phase end of TAC
               'linear_phase_start_m': start_m, # mrtm linear phase start of TAC
               'linear_phase_end_m': end_m, # mrtm linear phase end of TAC
               'linear_phase_start_m2': start_m2, # mrtm_k2p linear phase start of TAC
               'linear_phase_end_m2': end_m2, # mrtm_k2p linear phase end of TAC
               'fig': False
               }


#%%
# model
# srtm = simplified reference tissue model SRTM (nonlinear model)
# srtm_k2p = SRTM2 with pre-defined 𝑘′2 version (nonlinear model)
# srtmb_basis = SRTM with pre-calculated basis functions (linear model)
# srtmb_k2p_basis = SRTM2 with pre-calculated basis functions ( pre-defined 𝑘′2 version) (linear model)
# srtmb_asl_basis = SRTM ASK model (linear model). 
# SRTM ASL was a recently developed model (Scott et al. (2019)) for analysing PET data acquired by a simultaneous 
# PET-MR scanner where arterial spin labelling (ASL) is available to provide the perfusion information and derive 
# the relative influx rate 𝑅1.
# logan_ref = Logan graphical analysis with reference region (linear model). Depend on the linear phase of the plot.
# logan_ref_k2p = Logan graphical analysis with reference region and pre-defined 𝑘′2 (linear model). Depend on the linear phase of the plot
# logan methods are less used nowadays
# mrtm = Multilinear reference tissue model (linear model). Higher variance with slightly reduced bias compared to SRTM
# mrtm_k2p = Multilinear reference tissue model with pre-defined 𝑘′2 (linear model). Higher variance with slightly reduced bias compared to SRTM
# MRTM: the equations of Logan graphical analysis are solved with multilinear regression
# REF: doi.org/10.1097/01.WCB.0000085441.37552.CA. This paper provides a table comparing the characteristics of different models.


# In km_outputs BP is binding potential, R1 is influx rate, k2 is efflux rate, tacf is fitted TAC


# models = ['srtm', 'srtm_k2p', 'srtmb_basis', 'srtmb_asl_basis', 'srtmb_k2p_basis', 'mrtm', 'mrtm_k2p']

models = ['srtm', 'srtmb_basis', 'mrtm']
km_outputs = ['r1', 'k2p', 'BP']

filename = os.path.basename(os.path.splitext(img_pet.get_filename())[0]).split('.',1)[0]
save_base = os.path.join(vol_param_pre_folder, filename)
# save_ext = os.path.splitext(img_pet.get_filename())[1] # '.nii' or '.nii.gz'
save_ext = '.nii.gz'

for model_name in models:
    print(model_name)
    model_inputs = get_model_inputs(user_inputs, model_name)
    # if mask is not None, the model will calculate the voxel TAC in the mask region. 
    # Otherwise, it will fit region with value higher than a threshold depend on thr.
    parametric_images_dict, pet_image_fit = image_to_parametric(img_data_pet, dt, model_name, model_inputs, km_outputs,
                                                                mask=None, thr=0.5)
    for kp in parametric_images_dict.keys():
        nib.save(nib.Nifti1Image(parametric_images_dict[kp], img_pet.affine), save_base +
                 '_' + model_name + '_' + kp + save_ext)
    nib.save(nib.Nifti1Image(pet_image_fit, img_pet.affine), save_base +
             '_' + model_name + '_' + 'fit' + save_ext)

# %% load the k2p parametric images
save_ext = '.nii.gz'
filter_str = 'k2p' + save_ext
k2p_files = [os.path.join(vol_param_pre_folder, f) for f in os.listdir(vol_param_pre_folder) if (filter_str in f)]
k2p_list = {}

for k2p_file in k2p_files:
    img_k2p = nib.load(k2p_file)
    img_data_k2p = img_k2p.get_fdata()
    regions_data, regions_label = extract_regional_values_image_data(img_data_k2p, img_data_roi)
    k2p_data = regions_data[1] #regions_label=1
    k2p_mean = np.mean(k2p_data)
    k2p_list.update({os.path.basename(k2p_file): k2p_mean})

r1_files = [os.path.join(vol_param_pre_folder, f) for f in os.listdir(vol_param_pre_folder) if ('r1' in f)]
r1_list = {}

for r1_file in r1_files:
    img_r1 = nib.load(r1_file)
    img_data_r1 = img_r1.get_fdata()
    regions_data, regions_label = extract_regional_values_image_data(img_data_r1, img_data_roi)
    r1_data = regions_data[1] #regions_label=1
    r1_mean = np.mean(r1_data)
    r1_list.update({os.path.basename(r1_file): r1_mean})
    

# %% Set the k2p and r1 values according to the parametric images

k2p = k2p_list['M4-Postop-PET-DTBZ-CTAC-Dynamic_mrtm_k2p.nii.gz']
r1 = r1_list['M4-Postop-PET-DTBZ-CTAC-Dynamic_mrtm_r1.nii.gz']

#%% judge the linear phase of models
t = np.arange(0, len(input_roi))
# t_trunc to truncate the initial rising part of the TAC
li.logan_k2p_linear_phase(input_roi, input_ref, t, k2p, t_trunc=(200, np.inf), save_dir=linearity)
li.mrtm_k2p_linear_phase(input_roi, input_ref, t, k2p, t_trunc=(200, np.inf), save_dir=linearity)


#%% set the start and end time of the linear phase of the Logan plot

start_l2 = 800
end_l2 = None

start_m2 = 1000
end_m2 = None

#%% basis functions
t1_2 = 110 # half-life of the tracer in minutes
beta = np.log(2)/t1_2
beta_lim = [beta*0.1/60, beta*10/60] # convert to s-1
n_beta = 128
b = basis.make_basis(input_ref, dt, beta_lim=beta_lim, n_beta=n_beta, w=None, k2p=k2p, fig=True)

#%%
# provide all user inputs in one dict here and later 'get_model_inputs' will select the needed ones


user_inputs = {'dt': dt,
               'inputf1': input_ref,
               'w': None,
               'r1': r1, # influx rate 𝑅1
               'k2p': k2p,
               'beta_lim': beta_lim,
               'n_beta': n_beta,
               'b': b, # basis functions
               'linear_phase_start_l': start_l, # logan_ref linear phase start of TAC
               'linear_phase_end_l': end_l,  # logan_ref linear phase end of TAC
               'linear_phase_start_l2': start_l2, # logan_ref_k2p linear phase start of TAC
               'linear_phase_end_l2': end_l2, # logan_ref_k2p linear phase end of TAC
               'linear_phase_start_m': start_m, # mrtm linear phase start of TAC
               'linear_phase_end_m': end_m, # mrtm linear phase end of TAC
               'linear_phase_start_m2': start_m2, # mrtm_k2p linear phase start of TAC
               'linear_phase_end_m2': end_m2, # mrtm_k2p linear phase end of TAC
               'fig': False
               }

#%% load brain mask

mask_file = os.path.join(resample_folder, r'BrainMask.nii.gz')
img_mask = nib.load(mask_file)
img_data_mask = img_mask.get_fdata()
mask_brain = np.argwhere(img_data_mask > 0)

# %%
# models = ['srtm_k2p', 'srtmb_asl_basis', 'srtmb_k2p_basis', 'mrtm_k2p']
# models = ['srtmb_asl_basis', 'srtmb_k2p_basis', 'mrtm_k2p']
# models = ['srtm', 'srtmb_basis', 'mrtm']
models = ['srtm', 'srtmb_basis', 'mrtm', 'srtmb_asl_basis', 'srtmb_k2p_basis', 'mrtm_k2p']
km_outputs = ['r1', 'k2', 'k2p', 'BP'] # do not add tacf here, it will be saved as 'fit'

filename = os.path.basename(os.path.splitext(img_pet.get_filename())[0]).split('.',1)[0]
save_base = os.path.join(vol_param_folder, filename)
# save_ext = os.path.splitext(img_pet.get_filename())[1] # '.nii' or '.nii.gz'
save_ext = '.nii.gz'

for model_name in models:
    print(model_name)
    model_inputs = get_model_inputs(user_inputs, model_name)
    # if mask is not None, the model will calculate the voxel TAC in the mask region. 
    # Otherwise, it will fit region with value higher than a threshold depend on thr.
    parametric_images_dict, pet_image_fit = image_to_parametric(img_data_pet, dt, model_name, model_inputs, km_outputs,
                                                                mask=mask_brain, thr=0.5)
    for kp in parametric_images_dict.keys():
        nib.save(nib.Nifti1Image(parametric_images_dict[kp], img_pet.affine), save_base +
                 '_' + model_name + '_' + kp + save_ext)
    nib.save(nib.Nifti1Image(pet_image_fit, img_pet.affine), save_base +
             '_' + model_name + '_' + 'fit' + save_ext)

#%% set the params img path
srtmb_asl_basis_bp_files = os.path.join(vol_param_folder, r'M4-Postop-PET-DTBZ-CTAC-Dynamic_srtmb_asl_basis_bp.nii.gz')
srtmb_k2p_basis_bp_files = os.path.join(vol_param_folder, r'M4-Postop-PET-DTBZ-CTAC-Dynamic_srtmb_k2p_basis_bp.nii.gz')
mrtm_k2p_basis_bp_files = os.path.join(vol_param_folder, r'M4-Postop-PET-DTBZ-CTAC-Dynamic_mrtm_k2p_bp.nii.gz')
srtm_bp_files = os.path.join(vol_param_folder, r'M4-Postop-PET-DTBZ-CTAC-Dynamic_srtm_bp.nii.gz')
srtm_basis_bp_files = os.path.join(vol_param_folder, r'M4-Postop-PET-DTBZ-CTAC-Dynamic_srtm_bp.nii.gz')
mrtm_bp_files = os.path.join(vol_param_folder, r'M4-Postop-PET-DTBZ-CTAC-Dynamic_mrtm_bp.nii.gz')

model_list_files = [srtmb_asl_basis_bp_files, srtmb_k2p_basis_bp_files, mrtm_k2p_basis_bp_files, 
                    srtm_bp_files, srtm_basis_bp_files, mrtm_bp_files]

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
bp_list = {}

for model_file_f in model_list_files: # f means final
    img_model_f = nib.load(model_file_f)
    img_data_model_f = img_model_f.get_fdata()
    # Ensure the model file key exists in the dictionary
    model_name = os.path.basename(model_file_f)
    if model_name not in bp_list:
        bp_list[model_name] = {}    
    for roi_f_file in roi_list_files: # f means final
        img_roi_f = nib.load(roi_f_file)
        img_data_roi_f = img_roi_f.get_fdata()
        regions_data, regions_label = extract_regional_values_image_data(img_data_model_f, img_data_roi_f)
        bp_data = regions_data[1] # regions_label=1
        bp_mean = np.mean(bp_data)

        roi_name = os.path.basename(roi_f_file)
        bp_list[model_name][roi_name] = bp_mean        

#%% Save the params to csv files
df_bp = pd.DataFrame.from_dict(bp_list, orient='index')
df_bp.index.name = 'Model_Name'  # Set index name
df_bp.columns.name = 'ROI_Name'  # Set columns name

# Save the DataFrame to CSV
df_savename = os.path.join(param_folder, 'bp_list.csv')
df_bp.to_csv(df_savename)


#%% adjust the param img (optional)
# adjust the parametric images orientation in 3D slicer
# note: adjust the orientation of brain mask image to the same as the parametric images

# %% visualize the parametric images

img_bp_file = os.path.join(show_fig_folder, r'M6-Preop-PET-DTBZ-CTAC-Dynamic_mrtm_k2p_bp.nii.gz')
img_bp = nib.load(img_bp_file)
img_bp_data = img_bp.get_fdata()
mask_file = os.path.join(show_fig_folder, r'BrainMask.nii.gz')
img_mask = nib.load(mask_file)
img_data_mask = img_mask.get_fdata()
img_bp_file.split('.',1)[0]
save_base = os.path.join(show_fig_folder, os.path.basename(img_bp_file.split('.',1)[0]))
# Visualize and save an axial slice (slice 50)
nslice =[15, 20, 25, 30, 35, 40, 45]
for i in nslice:
    vol_heatmap(img_bp_data, nslice=i, orient='ax', mask=img_data_mask, 
            barlim=(0,5), colormap='viridis', save_base=save_base)
    vol_heatmap(img_bp_data, nslice=i, orient='cor', mask=img_data_mask,
            barlim=(0,5), colormap='viridis', save_base=save_base)
    vol_heatmap(img_bp_data, nslice=i, orient='sag', mask=img_data_mask,
            barlim=(0,5), colormap='viridis', save_base=save_base)

# %%
