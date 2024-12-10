#%%
__author__ = 'jieqing jiao'
__contact__ = 'jieqing.jiao@gmail.com'

# Ref: doi: 10.1007/s12021-022-09616-0
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import numpy as np
import nibabel as nib
import subprocess
from niftypad.tac import Ref
from niftypad import basis
from niftypad.image_process.parametric_image import image_to_parametric
from niftypad.models import get_model_inputs
from niftypad.image_process.regions import extract_regional_values_image_data
from niftypad.image_process.transform_image import dicom4_to_nifti3, load_4D

#%% file structure
wd = r"F:\Data\Image\NHP\NiftyPAD\M4"
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

os.makedirs("vol-param", exist_ok=True)
vol_param_folder = os.path.join(wd, "vol-param")

# #%% Paths provided in the query
# dicom_folder = r"E:\Chen Lab\Data\NHP\Image\M4\postmodel\200165_PET\200165-M4_preop_PET_2024-09-08\PET-CT\4660"
# output_file = r"F:\Data\Image\NHP\NiftyPAD\M4\raw-file\M4-Postop-PET-DTBZ-CTAC-Dynamic"

# #%% Call the function with the provided paths
# pet_file = convert_dicom_to_nifti_conda(dicom_folder, output_file)


#%% 
# extract reference region 
# use lead dbs to register the atlas to the image
# use 3d slicer to create the mask

#%% Split the 4D PET image into 3D images
# note that direct save the 4D image to nii file will cause the image orientation inconsistent with raw dicom image (do not know why)
# have try pkg: pydicom, dcm2niix, SimpleITK all failed
dicom_folder = r"E:\Chen Lab\Data\NHP\Image\M4\postmodel\200165_PET\200165-M4_preop_PET_2024-09-08\PET-CT\4660"
output_base = r"F:\Data\Image\NHP\NiftyPAD\M4\raw\M4-Postop-PET-DTBZ-CTAC-Dynamic"
dicom4_to_nifti3(dicom_folder, output_base)

#%% crop and resample and crop image to the same size in 3D slicer

# note: first crop to smaller size, then resample to the same size, then crop to the same size
# note: resample well make the images size consistent

#%%
# img file
# file_path = '/Users/Himiko/data/amsterdam_data/'
# pet_file = file_path + 'E301_FLUT_AC1_combined_ID_cleared.nii'
# img = nib.load(pet_file)
# pet_image = img.get_fdata()

pet_folder = r"F:\Data\Image\NHP\NiftyPAD\M4\resample\Scene\M4-Postop-PET-DTBZ-CTAC-Dynamic"
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
ref_file = r"F:\Data\Image\NHP\NiftyPAD\M4\resample\Scene\OccipitalMask.nii.gz"
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
inputf1 = ref.inputf1_exp_am

#%% Read the ROI mask file
roi_file = r"F:\Data\Image\NHP\NiftyPAD\M4\resample\Scene\PutamenRmask.nii.gz"
img_roi = nib.load(roi_file)
img_data_roi = img_roi.get_fdata()

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
# roi.run_feng_srtm() # may encounter overflow error, not recommended currently

#%% Visualize the ROI TAC

roi.plot_tac('interp_1', tac_roi_folder)
roi.plot_tac('interp_1cubic', tac_roi_folder)
roi.plot_tac('exp1', tac_roi_folder)
roi.plot_tac('exp2', tac_roi_folder)
roi.plot_tac('exp_am', tac_roi_folder)
# roi.plot_tac('feng_srtm', tac_roi_folder)


#%% set the start and end time of the linear phase of the Logan plot
start_t = 500
end_t = None

#%%
# k2p is the tissue-to-plasma clearance 𝑘2′ of the reference tissue of the parent radioligand in dynamic PET imaging (𝑘′2) .
# pre-defined in SRTM and SRTM2 nonlinear models
# k2p * Ct(t) = dCt(t)/dt. Ct(t) is the concentration of the parent radioligand in the tissue at time t.
# REF: http://www.turkupetcentre.net/petanalysis/model_compartmental_ref.html#SRTM
# Coupling of the k'2 (k2 in the reference region) to a common value across brain regions, 
# or to a first-pass estimate (as proposed for the BF implementations), 
# reduces the variance of parameter estimates (Wu & Carson, 2002; Ichise et al., 2003; Endres et al., 2011).

# REF: https://doi.org/10.1097/01.WCB.0000085441.37552.CA. This paper provides a way to estimate k2p.
# Run MRTM with the reference region TAC to get the k2 value of the reference region.
# averaging of several k′2 samples from high-BP ROIs might be required to minimize the variability of k′2 estimation. 


k2p = 0.000250
r1 = 0.905
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
b = basis.make_basis(ref.inputf1cubic, dt, beta_lim=beta_lim, n_beta=n_beta, w=None, k2p=k2p, fig=True)

#%%
# provide all user inputs in one dict here and later 'get_model_inputs' will select the needed ones


user_inputs = {'dt': dt,
               'inputf1': inputf1,
               'w': None,
               'r1': r1, # influx rate 𝑅1
               'k2p': k2p,
               'beta_lim': beta_lim,
               'n_beta': n_beta,
               'b': b, # basis functions
               'linear_phase_start_l': start_t, # logan_ref linear phase start of TAC
               'linear_phase_end_l': end_t,  # logan_ref linear phase end of TAC
               'linear_phase_start_l2': start_t, # logan_ref_k2p linear phase start of TAC
               'linear_phase_end_l2': end_t, # logan_ref_k2p linear phase end of TAC
               'linear_phase_start_m': start_t, # mrtm linear phase start of TAC
               'linear_phase_end_m': end_t, # mrtm linear phase end of TAC
               'linear_phase_start_m2': start_t, # mrtm_k2p linear phase start of TAC
               'linear_phase_end_m2': end_t, # mrtm_k2p linear phase end of TAC
               'fig': False
               }

#%%
# model
# srtm_fun = simplified reference tissue model SRTM (nonlinear model)
# srtm_fun_k2p = SRTM2 with pre-defined 𝑘′2 version (nonlinear model)
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


models = ['srtm_fun', 'srtm_fun_k2p', 'srtmb_basis', 'srtmb_asl_basis', 'srtmb_k2p_basis', 'mrtm', 'mrtm_k2p']
km_outputs = ['R1', 'k2', 'BP']

save_base = os.path.join(vol_param_folder, os.path.basename(os.path.splitext(img_pet.get_filename())[0]))
# save_ext = os.path.splitext(img_pet.get_filename())[1] # '.nii' or '.nii.gz'
save_ext = '.nii.gz'

for model_name in models:
    print(model_name)
    model_inputs = get_model_inputs(user_inputs, model_name)
    # if mask is not None, the model will fit TAC in the mask region. 
    # Otherwise, it will fit region with value higher than a threshold depend on thr.
    parametric_images_dict, pet_image_fit = image_to_parametric(img_data_pet, dt, model_name, model_inputs, km_outputs,
                                                                mask=None, thr=0.5)
    for kp in parametric_images_dict.keys():
        nib.save(nib.Nifti1Image(parametric_images_dict[kp], img_pet.affine), save_base +
                 '_' + model_name + '_' + kp + save_ext)
    nib.save(nib.Nifti1Image(pet_image_fit, img_pet.affine), os.path.splitext(img_pet.get_filename())[0] +
             '_' + model_name + '_' + 'fit' + save_ext)

# %%
