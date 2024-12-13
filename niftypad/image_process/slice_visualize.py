#%%
import numpy as np
import matplotlib.pyplot as plt
import os

def vol_heatmap(vol, nslice, orient, mask=None, barlim=None, colormap='hot', save_base=None):
    """
    Visualize a volume slice with a heatmap.

    Args:
        vol (numpy.ndarray): 3D tensor data of the volume.
        nslice (int): Slice number to visualize.
        orient (str): Slice orientation ('ax' for axial, 'cor' for coronal, 'sag' for sagittal).
        mask (numpy.ndarray, optional): Binary volume mask of the same shape as vol. Only masked regions are shown.
        barlim (tuple, optional): Colorbar limits as (vmin, vmax). If None, auto-scaling is used.
        colormap (str, optional): Matplotlib colormap name for visualization. Default is 'hot'.
        save_base (str, optional): Base name for saving the figure. Final name is save_base + '_nslice_orient.svg'.

    Available colormaps:
        Perceptually Uniform Sequential: ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
        Sequential: ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        Diverging: ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
        Cyclic: ['twilight', 'twilight_shifted', 'hsv']
        Qualitative: ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']
        Miscellaneous: ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral', 'gist_ncar']
    """

    # Crop the volume to the smallest bounding box
    # mask, crop_indices = vol_crop(mask) if mask is not None else None
    # vol = vol[crop_indices]
    vol, crop_indices = vol_crop(vol)
    mask = mask[crop_indices] if mask is not None else None


    # Validate orientation
    if orient not in ['ax', 'cor', 'sag']:
        raise ValueError("Invalid orientation. Use 'ax', 'cor', or 'sag'.")

    # Get the slice based on orientation
    if orient == 'sag':  # Sagittal
        slice_data = vol[nslice, :, :]
        mask_data = mask[nslice, :, :] if mask is not None else None
        slice_data = np.transpose(slice_data)  # Switch x and y 
        mask_data = np.transpose(mask_data) if mask is not None else None       
    elif orient == 'cor':  # Coronal
        slice_data = vol[:, nslice, :]
        mask_data = mask[:, nslice, :] if mask is not None else None
        slice_data = np.transpose(slice_data)  # Switch x and y
        mask_data = np.transpose(mask_data) if mask is not None else None
    elif orient == 'ax':  # Axial
        slice_data = vol[:, :, nslice]
        mask_data = mask[:, :, nslice] if mask is not None else None
        slice_data = np.rot90(slice_data, k=1)
        mask_data = np.rot90(mask_data, k=1) if mask is not None else None

    # Apply mask if provided
    if mask is not None:
        slice_data = np.where(mask_data, slice_data, np.nan)

    # Switch x and y and flip them
    


    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(slice_data, cmap=colormap, interpolation='nearest', origin='lower')

    # Set colorbar limits if provided
    if barlim is not None:
        plt.clim(barlim)

    plt.colorbar(label='Intensity')
    plt.title(f"Slice {nslice} - {orient.upper()}")
    plt.axis('off')

    # Save the figure
    if save_base:
        save_name = f"{save_base}_{nslice}_{orient}.svg"
        plt.savefig(save_name, format='svg', bbox_inches='tight')
        print(f"Figure saved to {save_name}")

    plt.show()


def vol_crop(vol):
    """
    Crop out the zero-value regions of a 3D volume, shrinking it to the smallest bounding box.

    Args:
        vol (numpy.ndarray): 3D tensor data of the volume.

    Returns:
        tuple: (Cropped volume (numpy.ndarray), crop indices (tuple of slices)).
    """
    # Find non-zero regions along each axis
    non_zero_coords = np.argwhere(vol != 0)
    min_coords = non_zero_coords.min(axis=0)
    max_coords = non_zero_coords.max(axis=0) + 1  # +1 because slicing is exclusive at the end

    # Define the crop indices
    crop_indices = (slice(min_coords[0], max_coords[0]),
                    slice(min_coords[1], max_coords[1]),
                    slice(min_coords[2], max_coords[2]))

    # Crop the volume to the bounding box
    vol_cropped = vol[crop_indices[0], crop_indices[1], crop_indices[2]]

    return vol_cropped, crop_indices


#%%
# Example volume and mask
# import nibabel as nib

# img_bp_file = r"F:\Data\Image\NHP\NiftyPAD\M4\Postop2\show-fig\M4-Postop-PET-DTBZ-CTAC-Dynamic_mrtm_k2p_bp.nii.gz"
# img_bp = nib.load(img_bp_file)
# img_bp_data = img_bp.get_fdata()
# mask_file = r"F:\Data\Image\NHP\NiftyPAD\M4\Postop2\show-fig\BrainMask_dilate.nii.gz"
# img_mask = nib.load(mask_file)
# img_data_mask = img_mask.get_fdata()
# save_base = r"F:\Data\Image\NHP\NiftyPAD\M4\Postop2\show-fig\M4-Postop-PET-DTBZ-CTAC-Dynamic_mrtm_k2p_bp"
# # Visualize and save an axial slice (slice 50)
# nslice =[15, 20, 25, 30, 35, 40, 45]
# for i in nslice:
#     vol_heatmap(img_bp_data, nslice=i, orient='ax', mask=img_data_mask, 
#             barlim=(0,5), colormap='viridis', save_base=save_base)
#     vol_heatmap(img_bp_data, nslice=i, orient='cor', mask=img_data_mask,
#             barlim=(0,5), colormap='viridis', save_base=save_base)
#     vol_heatmap(img_bp_data, nslice=i, orient='sag', mask=img_data_mask,
#             barlim=(0,5), colormap='viridis', save_base=save_base)
# %%
