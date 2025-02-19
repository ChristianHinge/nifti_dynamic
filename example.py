## VIGTIGT: installer pip install indexed_gzip = langt hurtigere! 

from pathlib import Path
import os
from scipy.ndimage import binary_erosion
import json
from nifti_dynamic.conversion import convert_dicom_to_nifti
from nifti_dynamic.patlak import voxel_patlak
from nifti_dynamic.utils import extract_tac
import nibabel as nib
import numpy as np

ROOT = Path("/homes/hinge/Projects/dynamic/data")
SUBJECT_ID = "sub01"
dcm_dir = ROOT / "dcm_example"

dynpet_path = ROOT / SUBJECT_ID / "dpt.nii.gz"
sidecar_path = ROOT / SUBJECT_ID / "dpt.json"
totalseg_path = ROOT / SUBJECT_ID / "seg_res.nii.gz"

os.makedirs(ROOT / SUBJECT_ID, exist_ok=True)

if not os.path.exists(dynpet_path):
    print("Convert DICOM to Nifti")
    convert_dicom_to_nifti(dcm_dir, dynpet_path, sidecar_path)

with open(sidecar_path,'r') as f:
    ts = np.array(json.load(f)["FrameTimesStart"])

dynpet = nib.load(dynpet_path)
totalseg = nib.load(totalseg_path)

print("Extracting aorta TAC")
aorta_seg = binary_erosion(totalseg.get_fdata() == 52,iterations=2)
aorta_tac = extract_tac(dynpet, aorta_seg)

print("Running Patlak")
patlak_arr = voxel_patlak(dynpet,aorta_tac,ts,axial_chunk_size=32,n_frames_linear_regression=4,gaussian_filter_size=4)

print("Saving Patlak")
nib.Nifti1Image(patlak_arr,dynpet.affine).to_filename(ROOT / SUBJECT_ID / "patlak.nii.gz")