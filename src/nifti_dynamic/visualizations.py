import nibabel as nib
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt, colors
from PIL import Image
import tempfile
import cv2
from scipy.ndimage import median_filter

_segmentation_cmap = colors.ListedColormap([
    plt.cm.tab10(1),
    plt.cm.tab10(0.99),
    plt.cm.Set2(5/7),
    plt.cm.Set2(4/7),
])

WORKING_DIR = Path("/depict/users/hinge/private/hedypet/dynamic")

def _get_centerline(binary_mask_3d, projection_axis_idx):
    true_value_coords = np.where(binary_mask_3d)
    coords_on_last_dim_for_true = true_value_coords[-1]
    coords_on_projection_axis_for_true = true_value_coords[projection_axis_idx]

    unique_last_dim_indices = np.unique(coords_on_last_dim_for_true)
    
    if not unique_last_dim_indices.size:
        return np.zeros(binary_mask_3d.shape[-1], dtype=int)
    
    fp_values = np.array([
        coords_on_projection_axis_for_true[coords_on_last_dim_for_true == v_idx].mean().round()
        for v_idx in unique_last_dim_indices
    ])
    
    x_all_indices = np.arange(binary_mask_3d.shape[-1])
    
    interpolated_centerline = np.interp(x_all_indices, unique_last_dim_indices, fp_values)
    
    smoothed_interpolated_centerline = median_filter(interpolated_centerline, 5)
    
    return smoothed_interpolated_centerline.astype(int)

def plot_aorta_visualizations(pet_array, segments_nifti, rois_nifti, save_path):
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir_path = Path(tempdir)
        
        fig_sagittal, axs_sagittal = plt.subplots(1, 10, figsize=(9, 6))
        xlim_sag, ylim_sag = _plot_single_aorta_view(
            pet_array, segments_nifti, rois_nifti, 
            view_axis=1, slice_definition="max", 
            ax_raw=axs_sagittal[0], ax_overlay=axs_sagittal[1]
        )
        axs_sagittal[0].set_title("MIP")
        axs_sagittal[2].set_title("Unrolled aorta segments", loc="left")
        axs_sagittal[0].set_ylabel("Sagittal")
        for i in range(4): 
            _plot_single_aorta_view(
                pet_array, segments_nifti, rois_nifti, 
                view_axis=1, slice_definition=i + 1, 
                ax_raw=axs_sagittal[(i + 1) * 2], ax_overlay=axs_sagittal[(i + 1) * 2 + 1], 
                xlim=xlim_sag, ylim=ylim_sag
            )
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(tempdir_path / "aorta_sagittal_views.jpg", dpi=300, bbox_inches='tight')
        plt.close(fig_sagittal)

        fig_coronal, axs_coronal = plt.subplots(1, 10, figsize=(9, 6))
        xlim_cor, ylim_cor = _plot_single_aorta_view(
            pet_array, segments_nifti, rois_nifti, 
            view_axis=0, slice_definition="max", 
            ax_raw=axs_coronal[0], ax_overlay=axs_coronal[1]
        )
        axs_coronal[0].set_ylabel("Coronal")
        for i in range(4): 
            _plot_single_aorta_view(
                pet_array, segments_nifti, rois_nifti, 
                view_axis=0, slice_definition=i + 1, 
                ax_raw=axs_coronal[(i + 1) * 2], ax_overlay=axs_coronal[(i + 1) * 2 + 1], 
                xlim=xlim_cor, ylim=ylim_cor
            )
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(tempdir_path / "aorta_coronal_views.jpg", dpi=300, bbox_inches='tight')
        plt.close(fig_coronal)

        img_sagittal = Image.open(tempdir_path / "aorta_sagittal_views.jpg")
        img_coronal = Image.open(tempdir_path / "aorta_coronal_views.jpg")
        combined_img_array = np.concatenate((np.array(img_sagittal), np.array(img_coronal)), axis=0)
        Image.fromarray(combined_img_array).save(save_path)

def _plot_single_aorta_view(pet_array, segments_nifti, rois_nifti, 
                           view_axis, slice_definition, 
                           ax_raw, ax_overlay, xlim=None, ylim=None):
    segments_array = segments_nifti.get_fdata()
    rois_array = rois_nifti.get_fdata()
    
    roi4_mask = (rois_array == 4)
    vmax = pet_array[roi4_mask].mean() * 1.6 if np.any(roi4_mask) else pet_array.max() * 0.5

    pet_view_2d, segments_view_2d, rois_view_2d = None, None, None

    if slice_definition == "max":
        pet_view_2d = np.rot90(pet_array.max(axis=view_axis))
        segments_view_2d = np.rot90(segments_array.max(axis=view_axis))
        rois_view_2d = np.rot90(rois_array.max(axis=view_axis))
    elif isinstance(slice_definition, int): 
        centerline_coords = _get_centerline(segments_array == slice_definition, view_axis)
        depth_indices = np.arange(pet_array.shape[2])
        
        if view_axis == 0: 
            pet_view_2d = np.rot90((pet_array[centerline_coords, :, depth_indices]).T, 1)
            rois_view_2d = np.rot90((rois_array[centerline_coords, :, depth_indices]).T, 1)
            segments_view_2d = np.rot90((segments_array[centerline_coords, :, depth_indices]).T, 1)
        elif view_axis == 1: 
            pet_view_2d = np.rot90(pet_array[:, centerline_coords, depth_indices])
            rois_view_2d = np.rot90(rois_array[:, centerline_coords, depth_indices])
            segments_view_2d = np.rot90(segments_array[:, centerline_coords, depth_indices])
        else:
             raise ValueError(f"Unsupported view_axis for centerline slicing: {view_axis}")
    else:
        raise ValueError(f"Invalid slice_definition argument: {slice_definition}")
    
    active_region_coords_x, active_region_coords_y = (
        np.where(segments_view_2d > 0) if slice_definition == "max" 
        else np.where(segments_view_2d == slice_definition)
    )

    if not active_region_coords_x.size or not active_region_coords_y.size:
        xmin_calc, xmax_calc = 0, pet_view_2d.shape[0]
        ymin_calc, ymax_calc = 0, pet_view_2d.shape[1]
    else:
        xmin_calc = max(0, active_region_coords_x.min() - 10)
        xmax_calc = min(pet_view_2d.shape[0], active_region_coords_x.max() + 10)
        ymin_calc = max(0, active_region_coords_y.min() - 10)
        ymax_calc = min(pet_view_2d.shape[1], active_region_coords_y.max() + 10)

    current_xlim = xlim if xlim is not None else [xmin_calc, xmax_calc]
    current_ylim = ylim if ylim is not None else [ymin_calc, ymax_calc]
    
    current_xlim = [max(0, int(current_xlim[0])), min(pet_view_2d.shape[0], int(current_xlim[1]))]
    current_ylim = [max(0, int(current_ylim[0])), min(pet_view_2d.shape[1], int(current_ylim[1]))]

    pet_cropped = pet_view_2d[current_xlim[0]:current_xlim[1], current_ylim[0]:current_ylim[1]]
    segments_cropped = segments_view_2d[current_xlim[0]:current_xlim[1], current_ylim[0]:current_ylim[1]]
    rois_cropped = rois_view_2d[current_xlim[0]:current_xlim[1], current_ylim[0]:current_ylim[1]]
    
    segment_contours_mask = np.zeros(pet_cropped.shape, dtype=np.uint8)
    for seg_val in [1, 2, 3, 4]: 
        binary_segment_for_contour = (segments_cropped == seg_val).astype("uint8")
        contours, _ = cv2.findContours(binary_segment_for_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(segment_contours_mask, contours, -1, seg_val, thickness=1)
    
    ax_raw.imshow(pet_cropped, cmap="gray_r", vmax=vmax)
    ax_raw.set_xticks([])
    ax_raw.set_yticks([])

    ax_overlay.imshow(pet_cropped, cmap="gray_r", vmax=vmax)
    ax_overlay.imshow(segment_contours_mask, alpha=(segment_contours_mask > 0) * 1.0, interpolation="nearest", vmin=1, vmax=4, cmap=_segmentation_cmap)
    ax_overlay.imshow(segments_cropped, alpha=(segments_cropped > 0) * 0.4, interpolation="nearest", vmin=1, vmax=4, cmap=_segmentation_cmap)
    ax_overlay.imshow(rois_cropped, alpha=(rois_cropped > 0) * 1.0, interpolation="nearest", vmin=1, vmax=4, cmap=_segmentation_cmap)
    ax_overlay.set_xticks([])
    ax_overlay.set_yticks([])

    return current_xlim, current_ylim