#%%
import glob
import os
from rhnode import RHJob, MultiJobRunner
import os
import json
import datetime
from medical_dataset import MedicalDataset
from hedit import CT, TotalSegmentator

def create_ts_json_sidecar(inputs, outputs):
    in_json  = str(inputs["in_file"]).replace(".nii.gz",".json")

    with open(in_json,"r") as handle:
        inj = json.load(handle)
        ref_uid = inj["SeriesInstanceUID"]
        
    outjson = str(outputs["out_segmentation"]).replace(".nii.gz",".json")

    sidecar = {
            "Version":outputs["out_version"],
            "Args": outputs["out_args"],
            "ReferenceSeriesInstanceUID":ref_uid,
            "CreatedTimestamp": datetime.datetime.now().isoformat(),
            "Sources":[
                f"bids:rawdata:{inputs['in_file'].split('/rawdata/')[-1]}"
                ]
        }

    with open(outjson,"w") as handle:
        json.dump(sidecar,handle,sort_keys=True,indent=4)

MODE = "total"
jobs = []
n_exists = 0

for sub in MedicalDataset("hedit:all:",{"ct":CT(),"seg":TotalSegmentator(MODE)}):
    out_img = sub["seg"]
    out_sidecar = out_img.replace(".nii.gz",".json")

    if os.path.exists(out_sidecar) and os.path.exists(out_img):
        n_exists+=1
        print(out_img)
        print("exists")
        continue

    inputs = {
            "in_file":sub["ct"],
            "out_segmentation":out_img,
            "task":MODE,
            "body_seg":True,
            "force_split":True
        }
    
    job = RHJob(
        node_name="totalsegmentator",
        manager_address="titan2:9040",
        inputs=inputs
    )
    jobs.append(job)

    print(len(jobs),n_exists)
    job_runner = MultiJobRunner(jobs,on_job_finish_handle=create_ts_json_sidecar,queue_length=8)
    job_runner.start()

# %%

