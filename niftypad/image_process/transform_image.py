#%%

import os
import subprocess
import glob
import nibabel as nib
import numpy as np
import pydicom

def dicom4_to_nifti3(dicom_folder, output_base):
    """
    Splits a 4D DICOM image into a series of 3D NIfTI images and saves them with a specified naming pattern.

    Args:
        dicom_folder (str): Path to the 4D DICOM folder.
        output_base (str): Base path of the 3D NIfTI files.

    Returns:
        None
    """
    # Ensure output directory exists
    output_dir = output_base
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert DICOM folder to a single NIfTI file using dcm2niix
    temp_nifti_path = os.path.join(output_dir, "temp_4d.nii")
    subprocess.run([
        "dcm2niix", "-o", output_dir, "-f", "temp_4d", dicom_folder
    ], check=True)

    # Load the 4D NIfTI file
    nifti_4d = nib.load(temp_nifti_path)
    data_4d = nifti_4d.get_fdata()
    affine = nifti_4d.affine

    # Split into 3D volumes and save each as a separate NIfTI file
    for i in range(data_4d.shape[3]):
        data_3d = data_4d[:, :, :, i]
        nifti_3d = nib.Nifti1Image(data_3d, affine)

        # Create output filename
        output_filename = os.path.join(output_dir, f"{os.path.basename(output_base)}_{i:03d}.nii")

        # Save 3D NIfTI file
        nib.save(nifti_3d, output_filename)

    # Clean up temporary 4D NIfTI file
    os.remove(temp_nifti_path)

    print(f"3D NIfTI images saved in {output_dir}")



def convert_dynamic_dicom_to_nii(dicom_folder: str, output_nii_file: str):
    """
    Converts dynamic DICOM PET data to a NIfTI file.

    Parameters:
        dicom_folder (str): Path to the folder containing DICOM files.
        output_nii_file (str): Path to save the output NIfTI file.

    Returns:
        None: Saves the NIfTI file at the specified location.
    """
    # Ensure the output folder exists
    output_dir = os.path.dirname(output_nii_file)
    os.makedirs(output_dir, exist_ok=True)

    # Construct dcm2niix command
    command = [
        "dcm2niix",
        "-z", "y",  # Compress NIfTI file with gzip
        "-f", os.path.basename(output_nii_file).replace('.nii', ''),  # File name without extension
        "-o", output_dir,  # Output folder
        dicom_folder  # Input folder with DICOM files
    ]

    # Run the command and capture output
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"Conversion successful: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e.stderr}")
        raise RuntimeError("Failed to convert DICOM to NIfTI.") from e



def combine_nifti_files(input_folder, output_path):
    """
    Combines all NIfTI files (.nii or .nii.gz) in a given folder into a single 4D NIfTI file.
    
    Parameters
    ----------
    input_folder : str
        Path to the folder containing NIfTI files.
    output_path : str
        Path (including filename) to the output 4D NIfTI file.
        
    Notes
    -----
    - All input NIfTI files must have the same dimensions, voxel sizes, and orientation.
    - The combined output will be a 4D dataset, where each 3D volume from the input
      files is stored along the 4th dimension.
    """

    # Search for all nii and nii.gz files in the input folder
    nifti_paths = sorted(glob.glob(os.path.join(input_folder, '*.nii')) + 
                         glob.glob(os.path.join(input_folder, '*.nii.gz')))

    if not nifti_paths:
        raise FileNotFoundError(f"No NIfTI files found in {input_folder}.")

    # Load the first image to get reference affine and header
    first_img = nib.load(nifti_paths[0])
    reference_affine = first_img.affine
    reference_header = first_img.header
    first_data = first_img.get_fdata()

    # Initialize a list to hold data arrays
    data_list = [first_data]

    # Load subsequent images and verify consistent shape
    for fp in nifti_paths[1:]:
        img = nib.load(fp)
        data = img.get_fdata()

        # Check that shapes match
        if data.shape != first_data.shape:
            raise ValueError(f"Image shape mismatch between {nifti_paths[0]} and {fp}. "
                             f"All images must have the same dimensions.")
        data_list.append(data)

    # Stack along the 4th dimension to create a 4D array
    combined_data = np.stack(data_list, axis=-1)

    # Create a new NIfTI image
    combined_img = nib.Nifti1Image(combined_data, affine=reference_affine, header=reference_header)

    # Save to the specified output path
    nib.save(combined_img, output_path)
    print(f"Combined 4D NIfTI file saved at: {output_path}")


def load_4D(folder_path):
    """
    Load all NIfTI files in a folder, extract their image data,
    and concatenate them along the fourth dimension to create a 4D image.

    Parameters:
        folder_path (str): Path to the folder containing NIfTI files.

    Returns:
        nib.Nifti1Image: A 4D NIfTI image created by concatenating all the input images.
    """
    img_list = []

    # Iterate through all files in the folder
    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        
        # Check if the file is a NIfTI file
        if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
            # Load the NIfTI file
            img = nib.load(file_path)
            # Extract image data
            img_data = img.get_fdata()
            # Append the image data to the list
            img_list.append(img_data)

    # Concatenate along the fourth dimension
    if not img_list:
        raise ValueError("No NIfTI files found in the specified folder.")

    concatenated_data = np.stack(img_list, axis=-1)

    # Create a new NIfTI image from the concatenated data
    concatenated_img = nib.Nifti1Image(concatenated_data, affine=img.affine, header=img.header)

    filename = folder_path + ".nii"
    concatenated_img.set_filename(filename)

    return concatenated_data, concatenated_img

# # %%
# dicom_folder = r"E:\Chen Lab\Data\NHP\Image\M4\postmodel\200165_PET\200165-M4_preop_PET_2024-09-08\PET-CT\4660"
# output_nii_file = r"F:\Data\Image\NHP\NiftyPAD\M4\raw\M4-Postop-PET-DTBZ-CTAC-Dynamic.nii"
# convert_dynamic_dicom_to_nii(dicom_folder, output_nii_file)
# # %%
# nii_folder = r"F:\Data\Image\NHP\NiftyPAD\M4\raw\M4-Postop-PET2-DTBZ-CTAC-Dynamic"
# output_nii_file = r"F:\Data\Image\NHP\NiftyPAD\M4\raw\M4-Postop-PET3-DTBZ-CTAC-Dynamic.nii"
# combine_nifti_files(nii_folder, output_nii_file)

# # %%
# dicom_folder = r"E:\Chen Lab\Data\NHP\Image\M4\postmodel\200165_PET\200165-M4_preop_PET_2024-09-08\PET-CT\4660"
# output_base = r"F:\Data\Image\NHP\NiftyPAD\M4\raw\M4-Postop-PET-DTBZ-CTAC-Dynamic"
# dicom4_to_nifti3(dicom_folder, output_base)
# # %%
# folder_path = r"F:\Data\Image\NHP\NiftyPAD\M4\resample\Scene\M4-Postop-PET-DTBZ-CTAC-Dynamic"
# img4d = load_4D(folder_path)
# # %%
