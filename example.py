## VIGTIGT: installer pip install indexed_gzip = langt hurtigere! 

from pathlib import Path
import os
from scipy.ndimage import binary_erosion
import json
from nifti_dynamic.conversion import convert_dicom_to_nifti
from nifti_dynamic.patlak import voxel_patlak
from nifti_dynamic.utils import extract_tac
from nifti_dynamic.aorta_rois import extract_aorta_vois, AortaSegment
import nibabel as nib
import numpy as np
from nifti_dynamic.utils import img_to_array_or_dataobj
from tqdm import tqdm

ROOT = Path("/homes/hinge/Projects/dynamic/data")
SUBJECT_ID = "sub01"
dcm_dir = ROOT / "dcm_example"

dynpet_path = ROOT / SUBJECT_ID / "dpt.nii.gz"
sidecar_path = ROOT / SUBJECT_ID / "dpt.json"
totalseg_path = ROOT / SUBJECT_ID / "seg_res.nii.gz"
#rhaorta_path = ROOT / SUBJECT_ID / "new_aorta.nii.gz"

os.makedirs(ROOT / SUBJECT_ID, exist_ok=True)

if not os.path.exists(dynpet_path):
    print("Convert DICOM to Nifti")
    convert_dicom_to_nifti(dcm_dir, dynpet_path, sidecar_path)

with open(sidecar_path,'r') as f:
    js = json.load(f)
    FrameTimesStart = np.array(js['FrameTimesStart'])
    FrameDuration = np.array(js['FrameDuration'])
    ts = FrameTimesStart + FrameDuration/2.0

dynpet = nib.load(dynpet_path)
totalseg = nib.load(totalseg_path)

aorta_seg = totalseg.get_fdata() == 52

print("Extracting descending VOI")

tacs = []

# for cylinder_width in [3,4,5]:
#     for volume_ml in [2.5,0.75,1,1.25,1.5,1.75,2.25]:
#         voi_filename = ROOT / SUBJECT_ID / "aorta_vois" / f"aorta_vois_{volume_ml:.2f}ml_{cylinder_width}vox.nii.gz"
#         if not os.path.exists(voi_filename):
#             try:
#                 aorta_vois, aorta_segments = extract_aorta_vois(aorta_seg, totalseg.affine, dynpet, FrameTimesStart, t_threshold=40, volume_ml=volume_ml,cylinder_width=cylinder_width)
#             except:
#                 continue
#             nib.Nifti1Image(aorta_vois.astype("uint8"),totalseg.affine).to_filename(voi_filename)
#         else:
#             aorta_vois = nib.load(voi_filename).get_fdata()
        
#         print("Extracting AIF")
#         for seg in tqdm(AortaSegment):
#             aorta_tac = extract_tac(dynpet, aorta_vois==seg.value)
#             tacs.append({
#                 "volume_ml":volume_ml,
#                 "seg": seg.name,
#                 "cylinder_width":cylinder_width,
#                 "tac":list(aorta_tac)
#             })
# import pandas as pd

aorta_vois, aorta_segments = extract_aorta_vois(aorta_seg, totalseg.affine, dynpet, FrameTimesStart, t_threshold=40, volume_ml=1,cylinder_width=3)
aorta_tac = extract_tac(dynpet, aorta_vois==AortaSegment.DESCENDING_BOTTOM.value)

# df = pd.DataFrame(tacs)
# df.to_csv("out.csv")

print("Running Patlak")
patlak_arr, intercepts_arr = voxel_patlak(dynpet,aorta_tac,ts,axial_chunk_size=64,n_frames_linear_regression=12,gaussian_filter_size=0)

print("Saving Patlak")
nib.Nifti1Image(patlak_arr,dynpet.affine).to_filename(ROOT / SUBJECT_ID / "patlak_12.nii.gz")
nib.Nifti1Image(intercepts_arr,dynpet.affine).to_filename(ROOT / SUBJECT_ID / "intercept.nii.gz")
