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
        output_filename = os.path.join(output_dir, f"{os.path.basename(output_base)}_{i:03d}.nii.gz")

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

    filename = folder_path + ".nii.gz"
    concatenated_img.set_filename(filename)

    return concatenated_data, concatenated_img

def resample_nifti(input_file, zoom_factor, save_dir):
    """
    Resample a 3D NIfTI image volume.

    Args:
        input_file (str): Path to the input NIfTI file.
        zoom_factor (int): Factor for resampling the image.
                           >1 for higher resolution, <1 for lower resolution.
        save_dir (str): Directory to save the resampled image.

    Returns:
        None
    """
    # Load the image and get the data
    img = nib.load(input_file)
    img_data = img.get_fdata()
    affine = img.affine
    header = img.header

    if zoom_factor > 1:
        # Increase resolution
        new_shape = tuple(dim * zoom_factor for dim in img_data.shape)
        resampled_data = np.repeat(
            np.repeat(
                np.repeat(img_data, zoom_factor, axis=0), zoom_factor, axis=1
            ), zoom_factor, axis=2
        )
    elif zoom_factor < 1:
        # Decrease resolution
        zoom_factor = 1 / zoom_factor
        pad_size = [(0, (zoom_factor - dim % zoom_factor) % zoom_factor) for dim in img_data.shape]
        padded_data = np.pad(img_data, pad_size, mode="constant")
        grouped_data = padded_data.reshape(
            padded_data.shape[0] // zoom_factor, zoom_factor,
            padded_data.shape[1] // zoom_factor, zoom_factor,
            padded_data.shape[2] // zoom_factor, zoom_factor
        )
        resampled_data = grouped_data.mean(axis=(1, 3, 5))

    # Save the resampled image
    resampled_img = nib.Nifti1Image(resampled_data, affine, header)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"resampled_zoom_{zoom_factor}.nii.gz")
    nib.save(resampled_img, save_path)


# import os
# import nibabel as nib
# import numpy as np
# import pydicom
# from pydicom.dataset import Dataset, FileMetaDataset
# import datetime
# from pathlib import Path

# def nii_series_to_dicom(input_folder, output_folder):
#     """
#     Convert a series of NIfTI (.nii or .nii.gz) files in a folder representing a 3D+t dynamic medical image
#     to a single DICOM series in another folder.

#     Args:
#         input_folder (str): Path to the folder containing NIfTI files (assumed to be in temporal order).
#         output_folder (str): Path to the folder to save the combined DICOM series.
#     """
#     # Ensure output folder exists
#     os.makedirs(output_folder, exist_ok=True)

#     # Sort the NIfTI files in temporal order
#     nii_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.nii') or f.endswith('.nii.gz')])

#     if not nii_files:
#         print("No NIfTI files found in the input folder.")
#         return

#     combined_data = []
#     affine = None
#     header = None

#     # Load and combine NIfTI files
#     for file_name in nii_files:
#         file_path = os.path.join(input_folder, file_name)
#         nii = nib.load(file_path)
#         data = nii.get_fdata()

#         if len(data.shape) != 3:
#             print(f"Skipping {file_name}: Not a 3D file.")
#             continue

#         if affine is None:
#             affine = nii.affine
#             header = nii.header

#         combined_data.append(data)

#     if not combined_data:
#         print("No valid 3D NIfTI files found.")
#         return

#     combined_data = np.stack(combined_data, axis=-1)  # Combine into a 4D array
#     num_volumes = combined_data.shape[3]

#     # Generate a series of DICOM files for each time point
#     for t in range(num_volumes):
#         volume_data = combined_data[..., t]

#         # Create a basic DICOM dataset
#         file_meta = FileMetaDataset()
#         file_meta.MediaStorageSOPClassUID = pydicom.uid.UID("1.2.840.10008.5.1.4.1.1.2")  # MR Image Storage
#         file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
#         file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian  # Specify Transfer Syntax UID
#         file_meta.ImplementationClassUID = pydicom.uid.generate_uid()

#         ds = Dataset()
#         ds.file_meta = file_meta

#         # Add patient and study information
#         ds.PatientName = "NIFTI_CONVERTED"
#         ds.PatientID = "123456"
#         ds.StudyInstanceUID = pydicom.uid.generate_uid()
#         ds.SeriesInstanceUID = pydicom.uid.generate_uid()
#         ds.SOPInstanceUID = pydicom.uid.generate_uid()
#         ds.Modality = "MR"
#         ds.ContentDate = datetime.datetime.now().strftime('%Y%m%d')
#         ds.ContentTime = datetime.datetime.now().strftime('%H%M%S')

#         # Add image information
#         ds.ImagePositionPatient = affine[:3, 3].tolist()
#         ds.ImageOrientationPatient = affine[:3, :3].flatten().tolist()
#         ds.PixelSpacing = [header['pixdim'][1], header['pixdim'][2]]
#         ds.SliceThickness = header['pixdim'][3]

#         # Add pixel data
#         ds.Rows, ds.Columns = volume_data.shape[:2]
#         ds.SamplesPerPixel = 1
#         ds.PhotometricInterpretation = "MONOCHROME2"
#         ds.BitsAllocated = 16
#         ds.BitsStored = 16
#         ds.HighBit = 15
#         ds.PixelRepresentation = 1
#         ds.PixelData = (volume_data.astype(np.int16)).tobytes()

#         # Save the DICOM file
#         output_path = Path(output_folder) / f"dynamic_series_t{t}.dcm"
#         pydicom.dcmwrite(str(output_path), ds)

#     print(f"Converted {len(nii_files)} NIfTI files into a DICOM series.")

# Example usage
# nii_series_to_dicom('/path/to/nifti/folder', '/path/to/output/dicom/folder')


# Example usage
# nii_series_to_dicom('/path/to/nifti/folder', '/path/to/output/dicom/folder')


# Example usage
# nii_series_to_dicom('/path/to/nifti/folder', '/path/to/output/dicom/folder')

# Example usage
# nii_to_dicom('/path/to/nifti/folder', '/path/to/output/dicom/folder')
# # %%
# dicom_folder = r"E:\Chen Lab\Data\NHP\Image\M4\postmodel\200165_PET\200165-M4_preop_PET_2024-09-08\PET-CT\4660"
# output_nii_file = r"F:\Data\Image\NHP\NiftyPAD\M4\raw\M4-Postop-PET-DTBZ-CTAC-Dynamic.nii"
# # convert_dynamic_dicom_to_nii(dicom_folder, output_nii_file)
# # %%
# nii_folder = r"F:\Data\Image\NHP\NiftyPAD\M4\PostMPTP\test"
# output_nii_file = r"F:\Data\Image\NHP\NiftyPAD\M4\PostMPTP\test.nii"
# combine_nifti_files(nii_folder, output_nii_file)

# # %%
# dicom_folder = r"E:\Chen Lab\Data\NHP\Image\M4\postmodel\200165_PET\200165-M4_preop_PET_2024-09-08\PET-CT\4660"
# output_base = r"F:\Data\Image\NHP\NiftyPAD\M4\raw\M4-Postop-PET-DTBZ-CTAC-Dynamic"
# dicom4_to_nifti3(dicom_folder, output_base)
# # %%
# folder_path = r"F:\Data\Image\NHP\NiftyPAD\M4\resample\Scene\M4-Postop-PET-DTBZ-CTAC-Dynamic"
# img4d = load_4D(folder_path)
# # %%
