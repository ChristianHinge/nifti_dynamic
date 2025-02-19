import numpy as np
import nibabel as nib

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
    else:
        raise ValueError("Input must be a Nifti1Image or a numpy array.")

def extract_tac(img, seg):
    img = img_to_array_or_dataobj(img)
    seg = seg > 0
    nonzero = np.nonzero(seg)
    # Get min and max for each dimension
    xmin, xmax = np.min(nonzero[0]), np.max(nonzero[0])
    ymin, ymax = np.min(nonzero[1]), np.max(nonzero[1])
    zmin, zmax = np.min(nonzero[2]), np.max(nonzero[2])

    seg_bb = img[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1,:]
    tac = seg_bb[seg[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1],:].mean(axis=0)
    return tac
