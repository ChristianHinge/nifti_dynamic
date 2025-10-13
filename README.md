![Banner](banner.jpg)

# nifti_dynamic 🧠⚡

Efficient TAC extraction and Patlak analysis of HUGE PET arrays in NIfTI format. Works with gzipped 4D PET images (saving 90% disk space vs DICOM) 📦. All algorithms load partial chunks to prevent memory crashes when handling massive arrays (440×440×645×69×4 bytes = 32GB).

> Implementation based on Andersen TL, et al. *Diagnostics* 2024;14(15):1590. [doi:10.3390/diagnostics14151590](https://doi.org/10.3390/diagnostics14151590)

## Installation 💾

```bash
pip install nifti_dynamic
```

**Important**: Always ensure `indexed_gzip` is installed for significantly faster reading of gzipped NIfTI arrays (.nii.gz) ⚡

## Performance ⏱️

- Automatic extraction of 4 VOIs: 30 seconds
- TAC for single organs: 10 seconds  
- TACs for all 114 TotalSegmentator organs: 2 minutes
- Full voxel Patlak for 440×440×645×69 array: <1 minute

## Usage 🚀

```python
from nifti_dynamic.patlak import roi_patlak, voxel_patlak
from nifti_dynamic.utils import extract_tac, extract_multiple_tacs
from nifti_dynamic.aorta_rois import pipeline, AortaSegment
import nibabel as nib
from nibabel.processing import resample_from_to
import json 
import numpy as np
from matplotlib import pyplot as plt

# Load dynamic PET and frame times
dynpet = nib.load(".data/dpet.nii.gz")

with open(".data/dpet.json", "r") as handle:
    sidecar = json.load(handle)
    frame_times_start = np.array(sidecar["FrameTimesStart"])
    frame_duration = np.array(sidecar["FrameDuration"])
    frame_time_middle = frame_times_start + frame_duration/2

# Load and resample TotalSegmentator mask to dynamic PET
totalseg = nib.load(".data/totalseg.nii.gz")
totalseg = resample_from_to(totalseg,(dynpet.shape[:3],dynpet.affine),order=0)

# Define aorta mask image
aorta = nib.Nifti1Image((totalseg.get_fdata() == 52).astype("uint8"),affine=totalseg.affine)

# Extract aorta segments and aorta input function VOIs
aorta_segments, aorta_vois = pipeline(
    aorta_mask = aorta,
    dpet = dynpet,
    frame_times_start=frame_times_start,
    cylinder_width=3,
    volume_ml=1,
    image_path=".data/visualization.jpg")

# Use 1-ml bottom descending aorta VOI
descending_bottom_voi = aorta_vois.get_fdata()==AortaSegment.DESCENDING_BOTTOM.value

# Extract TACs
if_mu = extract_tac(dynpet, descending_bottom_voi)
brain_mu = extract_tac(dynpet, totalseg.get_fdata()==90)
liver_mu = extract_tac(dynpet, totalseg.get_fdata()==5)

# Extract multiple TACs efficiently
extract_multiple_tacs(dynpet, totalseg.get_fdata())

# ROI Patlak analysis
brain_slope, brain_intercept, X, Y = roi_patlak(brain_mu,if_mu,frame_time_middle,n_frames_linear_regression=4)

# Voxel Patlak analysis
img_slope, img_intercept = voxel_patlak(
    dynpet, if_mu,
    frame_time_middle,
    n_frames_linear_regression=6,
    gaussian_filter_size=2, # Optional pre-Patlak smoothing due to ultra-lowdose
    axial_chunk_size=32 # Increase to speed up at the cost of RAM
    )
```


