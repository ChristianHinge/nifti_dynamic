import numpy as np
import nibabel as nib
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import os 
import csv 


class OverlappedChunkIterator:
    """
    Iterator for processing array data in overlapping chunks with border handling.
    Useful for operations that have edge effects (like Gaussian filtering).
    """
    def __init__(self, array_size, chunk_size, border_size):
        """
        Initialize the iterator.
        
        Args:
            array_size: Size of the array to be chunked
            chunk_size: Size of each chunk to process
            border_size: Size of the border to overlap (e.g., 3 * gaussian_std)
        """
        self.array_size = array_size
        self.chunk_size = chunk_size
        self.border_size = border_size
        self.effective_chunk_size = chunk_size - 2 * border_size
        
        if self.effective_chunk_size <= 0:
            raise ValueError("Chunk size too small for given border size. "
                           "Increase chunk_size or decrease border_size.")
    
    def __len__(self):
        """
        Calculate total number of chunks that will be processed.
        """
        return (self.array_size + self.effective_chunk_size - 1) // self.effective_chunk_size

    def __iter__(self):
        """
        Returns iterator object (self).
        """
        self.current_pos = 0
        return self
    
    def __next__(self):
        """
        Returns the next chunk information as a tuple:
        (start_index, end_index, valid_start, valid_end, output_start, output_size)
        """
        if self.current_pos >= self.array_size:
            raise StopIteration
        
        # Calculate padding sizes
        pad_before = min(self.border_size, self.current_pos)
        remaining_space = self.array_size - (self.current_pos + self.effective_chunk_size)
        pad_after = min(self.border_size, max(0, remaining_space))
        
        # Calculate chunk indices
        start_idx = self.current_pos - pad_before
        end_idx = self.current_pos + self.effective_chunk_size + pad_after
        
        # Calculate valid region within chunk
        valid_start = pad_before
        valid_end = (end_idx - start_idx) - pad_after
        
        # Calculate output region
        output_start = self.current_pos
        output_size = min(self.effective_chunk_size, self.array_size - self.current_pos)
        
        # Prepare for next iteration
        self.current_pos += self.effective_chunk_size
        
        return (start_idx, end_idx, valid_start, valid_end, output_start, output_size)

def img_to_array_or_dataobj(img):
    if isinstance(img, nib.nifti1.Nifti1Image):
        return img.dataobj
    elif isinstance(img, np.ndarray):
        return img
    elif isinstance(img,nib.arrayproxy.ArrayProxy):
        return img
    elif isinstance(img,Path) or isinstance(img,str):
        return nib.load(img).dataobj
    else:
        raise ValueError("Input must be a Nifti1Image or a numpy array.")

# def extract_tac(img, seg, max_roi_size=None):
#     img = img_to_array_or_dataobj(img)
#     seg = seg > 0
#     nonzero = np.nonzero(seg)
#     # Get min and max for each dimension
#     xmin, xmax = np.min(nonzero[0]), np.max(nonzero[0])
#     ymin, ymax = np.min(nonzero[1]), np.max(nonzero[1])
#     zmin, zmax = np.min(nonzero[2]), np.max(nonzero[2])

#     if max_roi_size is not None and (xmax-xmin)*(ymax-ymin)*(zmax-zmin) > max_roi_size:
#         raise ValueError("Segmentation too big, use extract_multiple:tacs")

#     seg_bb = img[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1,:]
#     tac = seg_bb[seg[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1],:].mean(axis=0)

#     return tac

def extract_tac(img, seg, max_roi_size=None, return_std_n=False):
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

    if return_std_n:
        tac_std = img_masked.std(axis=0)
        n_voxels = np.array([seg.sum()]*len(tac_mean))
        return tac_mean, tac_std, n_voxels
    else:
        return tac_mean



def extract_multiple_tacs(img, seg, max_roi_size_factor=2, return_std_n=False):
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
    for k in tqdm(tacs_mean):
        try: 
            tacs_mean[k], tacs_std[k], tacs_n[k] = extract_tac(img,seg==k,max_roi_size=max_roi_size,return_std_n=True)
            targets.remove(k)
        except ValueError as e:
            print("label",k,"too large - will run without 4D cropping")
            continue

    #Iterate through each frame - slower but uses less memory
    if len(targets) > 0:
        for i in tqdm(range(n_frames)):
            frame = img[...,i]
            for k in targets:
                seg_target = seg==k
                arr = frame[seg_target]
                tacs_mean[k].append(arr.mean())
                tacs_std[k].append(arr.mean())
                tacs_n[k].append(seg_target.sum())

        for k in targets:
            tacs_mean[k] = np.array(tacs_mean[k])
            tacs_std[k] = np.array(tacs_std[k])
            tacs_n[k] = np.array(tacs_n[k])

    if return_std_n:
        return tacs_mean, tacs_std, tacs_n
    else:
        return tacs_mean


def save_tac(filename, tac_mean,tac_std = None, n_voxels = None):
    filename = Path(filename)
    os.makedirs(filename.parent,exist_ok=True)
    data = {
        "mu": [float(x) for x in tac_mean],
        "std": [float(x) if tac_std is not None else None for x in tac_std],
        "n": [int(x) if n_voxels is not None else None for x in n_voxels],
    }
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data.keys()) # Write headers
        writer.writerows(zip(*data.values())) # Write data rows

def load_tac(filename):
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader) # Read header row
        read_dict = {header: list(column) for header, column in zip(headers, zip(*reader))}
    return np.array(read_dict["mu"]).astype(float), np.array(read_dict["std"]).astype(float), np.array(read_dict["n"]).astype(int)

def _pooled_mean_variance(mu1,mu2,n1,n2,v1,v2):
    n_comb = n1+n2
    mu_comb = (mu1*n1+mu2*n2)/(n_comb)
    var_comb = (n1*v1+n2*v2)/n_comb+n1*n2*np.square((mu1-mu2)/n_comb)
    return np.nan_to_num(mu_comb), np.nan_to_num(var_comb), n_comb

def combine_tacs(tacs_paths, tacs_output_path):
    comb_mu = comb_var = comb_n = 0

    for tac_p in tacs_paths:
        mu, std, n = load_tac(tac_p)
        comb_mu, comb_var, comb_n = _pooled_mean_variance(mu,comb_mu,n,comb_n,np.square(std),comb_var)

    save_tac(tacs_output_path,comb_mu,np.sqrt(comb_var),comb_n)

def load_and_combine_tacs(tacs_paths):
    comb_mu = comb_var = comb_n = 0

    for tac_p in tacs_paths:
        mu, std, n = load_tac(tac_p)
        comb_mu, comb_var, comb_n = _pooled_mean_variance(mu,comb_mu,n,comb_n,np.square(std),comb_var)

    return comb_mu,np.sqrt(comb_var),comb_n