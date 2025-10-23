"""Functions for extracting and managing time-activity curves (TACs) from dynamic PET images."""

import numpy as np
import nibabel as nib
from pathlib import Path
import os
import csv


def img_to_array_or_dataobj(img):
    """Convert various image formats to array or dataobj.

    Args:
        img: Image as Nifti1Image, numpy array, ArrayProxy, or path

    Returns:
        Array-like object (dataobj or ndarray)
    """
    if isinstance(img, nib.nifti1.Nifti1Image):
        return img.dataobj
    elif isinstance(img, np.ndarray):
        return img
    elif isinstance(img, nib.arrayproxy.ArrayProxy):
        return img
    elif isinstance(img, Path) or isinstance(img, str):
        return nib.load(img).dataobj
    else:
        raise ValueError("Input must be a Nifti1Image or a numpy array.")


def extract_tac(img, seg, max_roi_size=None):
    """Extract time-activity curve from a single ROI.

    Args:
        img: 4D image data (x, y, z, time) or Nifti1Image
        seg: 3D binary segmentation mask
        max_roi_size: Maximum ROI size (raises error if exceeded)

    Returns:
        tac_mean: Mean TAC values
        tac_std: Standard deviation TAC values
        n_voxels: Number of voxels (constant across timepoints)
    """
    img = img_to_array_or_dataobj(img)
    seg = seg > 0
    nonzero = np.nonzero(seg)
    # Get min and max for each dimension
    xmin, xmax = np.min(nonzero[0]), np.max(nonzero[0])
    ymin, ymax = np.min(nonzero[1]), np.max(nonzero[1])
    zmin, zmax = np.min(nonzero[2]), np.max(nonzero[2])

    ## Vectorized operations can use a lot of memory.
    if max_roi_size is not None and (xmax-xmin)*(ymax-ymin)*(zmax-zmin)*img.shape[-1] > max_roi_size:
        raise ValueError("Segmentation too big, use extract_multiple_tacs")

    img_bb = img[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1,:]
    img_masked = img_bb[seg[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1],:]
    tac_mean = img_masked.mean(axis=0)
    tac_std = img_masked.std(axis=0)
    n_voxels = np.array([seg.sum()]*len(tac_mean))

    return tac_mean, tac_std, n_voxels


def extract_multiple_tacs(img, seg, max_roi_size_factor=2, _rich_progress=None, _rich_task=None):
    """Extract TACs for all ROIs in a segmentation image.

    Args:
        img: 4D image data (x, y, z, time) or Nifti1Image
        seg: 3D segmentation with integer labels
        max_roi_size_factor: Memory optimization factor (higher = faster but more RAM)
        _rich_progress: Internal - Rich Progress object (optional)
        _rich_task: Internal - Rich progress task ID (optional)

    Returns:
        tacs_mean: Dictionary mapping label -> mean TAC
        tacs_std: Dictionary mapping label -> std TAC
        tacs_n: Dictionary mapping label -> n_voxels
    """
    img = img_to_array_or_dataobj(img)

    #handle static images
    if img.ndim == 3:
        img = np.asanyarray(img)
        img = img[:,:,:,np.newaxis]

    n_frames = img.shape[-1]

    targets = list(np.unique(seg))
    if 0 in targets:
        targets.remove(0)

    tacs_mean = {int(x):[] for x in targets}
    tacs_std = {int(x):[] for x in targets}
    tacs_n = {int(x):[] for x in targets}

    #Try 4D cropping - faster but uses too much memory for larger organs
    max_roi_size = max_roi_size_factor*np.prod(seg.shape)

    # Create progress bar for ROI extraction
    roi_task = None
    if _rich_progress is not None:
        roi_task = _rich_progress.add_task("Extracting ROIs (4D cropping)", total=len(tacs_mean))

    for k in tacs_mean.keys():
        try:
            tacs_mean[k], tacs_std[k], tacs_n[k] = extract_tac(img, seg==k, max_roi_size=max_roi_size)
            targets.remove(k)
            if roi_task is not None:
                _rich_progress.advance(roi_task)
        except ValueError as e:
            print("label",k,"too large - will run without 4D cropping")
            if roi_task is not None:
                _rich_progress.advance(roi_task)
            continue

    #Iterate through each frame - slower but uses less memory
    if len(targets) > 0:
        frame_task = None
        if _rich_progress is not None:
            frame_task = _rich_progress.add_task("Extracting large ROIs (per-frame)", total=n_frames)

        for i in range(n_frames):
            frame = img[...,i]
            for k in targets:
                seg_target = seg==k
                arr = frame[seg_target]
                tacs_mean[k].append(arr.mean())
                tacs_std[k].append(arr.std())
                tacs_n[k].append(seg_target.sum())

            if frame_task is not None:
                _rich_progress.advance(frame_task)

        for k in targets:
            tacs_mean[k] = np.array(tacs_mean[k])
            tacs_std[k] = np.array(tacs_std[k])
            tacs_n[k] = np.array(tacs_n[k])

    return tacs_mean, tacs_std, tacs_n


def save_tac(filename, tac_mean, tac_std, n_voxels, time):
    """Save TAC to CSV file with time, mu, std, n_voxels columns.

    Args:
        filename: Output CSV file path
        tac_mean: Mean TAC values
        tac_std: Standard deviation TAC values
        n_voxels: Number of voxels per timepoint
        time: Time points in seconds
    """
    filename = Path(filename)
    os.makedirs(filename.parent, exist_ok=True)

    data = {
        "time": [float(x) for x in time],
        "mu": [float(x) for x in tac_mean],
        "std": [float(x) for x in tac_std],
        "n_voxels": [int(x) for x in n_voxels],
    }

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data.keys())  # Write headers
        writer.writerows(zip(*data.values()))  # Write data rows


def load_tac(filename):
    """Load TAC from CSV file.

    Returns:
        time: Time points
        tac_mean: Mean TAC values
        tac_std: Standard deviation TAC values
        n_voxels: Number of voxels per timepoint
    """
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Read header row
        read_dict = {header: list(column) for header, column in zip(headers, zip(*reader))}

    time = np.array(read_dict["time"]).astype(float) if "time" in read_dict else None
    tac_mean = np.array(read_dict["mu"]).astype(float)
    tac_std = np.array(read_dict["std"]).astype(float)
    n_voxels = np.array(read_dict["n_voxels" if "n_voxels" in read_dict else "n"]).astype(int)

    return time, tac_mean, tac_std, n_voxels


def _pooled_mean_variance(mu1, mu2, n1, n2, v1, v2):
    """Calculate pooled mean and variance from two samples.

    Args:
        mu1, mu2: Means of the two samples
        n1, n2: Sample sizes
        v1, v2: Variances of the two samples

    Returns:
        Combined mean, variance, and sample size
    """
    n_comb = n1+n2
    mu_comb = (mu1*n1+mu2*n2)/(n_comb)
    var_comb = (n1*v1+n2*v2)/n_comb+n1*n2*np.square((mu1-mu2)/n_comb)
    return np.nan_to_num(mu_comb), np.nan_to_num(var_comb), n_comb


def combine_tacs(tacs_paths, tacs_output_path):
    """Combine multiple TACs using pooled mean and variance.

    Args:
        tacs_paths: List of paths to TAC CSV files
        tacs_output_path: Output path for combined TAC
    """
    comb_mu = comb_var = comb_n = 0
    time = None

    for tac_p in tacs_paths:
        t, mu, std, n = load_tac(tac_p)
        if time is None:
            time = t
        else:
            assert np.allclose(time, t), f"Time points do not match when combining TACs from {tac_p}"
        comb_mu, comb_var, comb_n = _pooled_mean_variance(mu, comb_mu, n, comb_n, np.square(std), comb_var)

    save_tac(tacs_output_path, comb_mu, np.sqrt(comb_var), comb_n, time=time)


def load_and_combine_tacs(tacs_paths):
    """Load and combine multiple TACs using pooled mean and variance.

    Args:
        tacs_paths: List of paths to TAC CSV files

    Returns:
        time: Time points
        tac_mean: Combined mean TAC
        tac_std: Combined standard deviation
        n_voxels: Combined number of voxels
    """
    comb_mu = comb_var = comb_n = 0
    time = None

    for tac_p in tacs_paths:
        t, mu, std, n = load_tac(tac_p)
        if time is None:
            time = t
        else:
            assert np.allclose(time, t), f"Time points do not match when combining TACs from {tac_p}"
        comb_mu, comb_var, comb_n = _pooled_mean_variance(mu, comb_mu, n, comb_n, np.square(std), comb_var)

    return time, comb_mu, np.sqrt(comb_var), comb_n
