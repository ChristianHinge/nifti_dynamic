
import os
import subprocess
import tempfile
import shutil

_default_dcm2niix_args = ["-m","y","-z","o","-b","y", "-ba","n"]

def convert_dicom_to_nifti(dcmdir, nifti_out,sidecar_out=None):
    assert str(nifti_out).endswith(".nii.gz"), "Output filename must be a nifti gz file e.g. (out.nii.gz)"
    if sidecar_out is None:
        sidecar_out = nifti_out.replace(".nii.gz",".json")

    with tempfile.TemporaryDirectory() as tmp:
        cmd = ["dcm2niix","-o", tmp] + _default_dcm2niix_args + [str(dcmdir)] 
        subprocess.check_output(cmd)
        nifti_tmp = _get_nifti_from_dir(tmp)
        sidecar_tmp = _get_json_from_dir(tmp)

    shutil.move(nifti_tmp,nifti_out)
    shutil.move(sidecar_tmp,sidecar_out)

def _get_nifti_from_dir(tempdir):
    fs = os.listdir(tempdir)
    fs = [x for x in fs if x.endswith(".nii.gz") and "ROI" not in x]
    assert len(fs) == 1, "Too many or few output files; " + ",".join(fs)
    return os.path.join(tempdir,fs[0])

def _get_json_from_dir(tempdir):
    fs = os.listdir(tempdir)
    fs = [x for x in fs if x.endswith(".json") and "ROI" not in x]
    assert len(fs) == 1, "Too many or few output files; " + ",".join(fs)
    return os.path.join(tempdir,fs[0])

