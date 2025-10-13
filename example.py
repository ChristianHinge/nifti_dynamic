
from nifti_dynamic.patlak import roi_patlak, voxel_patlak
from nifti_dynamic.utils import extract_tac, extract_multiple_tacs
from nifti_dynamic.aorta_rois import pipeline, AortaSegment
import nibabel as nib
from nibabel.processing import resample_from_to
import json 
import numpy as np
from matplotlib import pyplot as plt
#Load dynamic pet and frame-times
dynpet = nib.load(".data/dpet.nii.gz")

with open(".data/dpet.json", "r") as handle:
    sidecar = json.load(handle)
    frame_times_start = np.array(sidecar["FrameTimesStart"])
    frame_duration = np.array(sidecar["FrameDuration"])
    frame_time_middle = frame_times_start + frame_duration/2

# Load and desample totalsegmentator mask to dynamic pet
totalseg = nib.load(".data/totalseg.nii.gz")
totalseg = resample_from_to(totalseg,(dynpet.shape[:3],dynpet.affine),order=0)

# Define aorta mask image
aorta = nib.Nifti1Image((totalseg.get_fdata() == 52).astype("uint8"),affine=totalseg.affine)

#Extract aorta segments and aorta input function vois
aorta_segments, aorta_vois = pipeline(
    aorta_mask = aorta,
    dpet = dynpet,
    frame_times_start=frame_times_start,
    cylinder_width=3,
    volume_ml=1,
    image_path=".data/visualization.jpg")

# Use 1-ml bottom descending aorta VOI
descending_bottom_voi = aorta_vois.get_fdata()==AortaSegment.DESCENDING_BOTTOM.value

print("Extract single TACs")
if_mu = extract_tac(dynpet, descending_bottom_voi)
brain_mu = extract_tac(dynpet, totalseg.get_fdata()==90)
liver_mu = extract_tac(dynpet, totalseg.get_fdata()==5)

print("Extract multiple tacs")
_ = extract_multiple_tacs(dynpet,totalseg.get_fdata())

#ROI patlak
brain_slope, brain_intercept, X, Y = roi_patlak(brain_mu,if_mu,frame_time_middle,n_frames_linear_regression=4)

print("Voxel patlak")
img_slope, img_intercept = voxel_patlak(
    dynpet,if_mu,
    frame_time_middle,
    n_frames_linear_regression=6,
    gaussian_filter_size=2, # Optional pre-patlak smoothing due to ultra-lowdose
    axial_chunk_size=32 # Increase to speed up at the cost of RAM
    )


plt.figure(figsize=(3,3))
plt.plot(frame_time_middle,if_mu,'.-',label="input function")
plt.plot(frame_time_middle,brain_mu,'.-',label="brain")
plt.plot(frame_time_middle,liver_mu,'.-',label="liver")
plt.xlabel("time (s)")
plt.ylabel("Activity")
plt.legend()
plt.title("Time-activity curve")
plt.savefig(".data/tacs.jpg",dpi=600)


plt.figure(figsize=(3,3))
plt.plot(X,Y,'.')
patlak = lambda x: brain_slope*x+brain_intercept
plt.plot([np.nanmin(X),np.nanmax(X)],[patlak(np.nanmin(X)),patlak(np.nanmax(X))])
plt.title("Brain Patlak analysis")
plt.savefig(".data/brain_patlak.jpg",dpi=600)


plt.figure()
patlak_mip = img_slope.get_fdata().max(axis=1)
voxel_size = img_slope.header.get_zooms()
aspect_ratio = voxel_size[0]/voxel_size[2]
plt.imshow(np.rot90(patlak_mip),aspect=aspect_ratio, cmap="gray_r",vmin=0,vmax=0.01)
plt.colorbar()
plt.axis("off")
plt.title("Patlak Ki MIP")
plt.savefig(".data/patlak_mip.jpg",dpi=300)
