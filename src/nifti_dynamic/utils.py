"""Utility functions for dynamic PET image processing."""

import numpy as np
from pathlib import Path
import json 


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

def get_sidecar_path(pet_path, sidecar_path=None):
    """Determine sidecar JSON path from PET image path.

    Args:
        pet_path: Path to PET image
        sidecar_path: Optional explicit sidecar path

    Returns:
        Path to sidecar JSON file

    Raises:
        SystemExit: If sidecar file does not exist
    """
    import sys

    pet_path = Path(pet_path)

    if sidecar_path is None:
        sidecar_path = pet_path.with_suffix(".json")
        if pet_path.suffix == ".gz":
            sidecar_path = pet_path.with_suffix("").with_suffix(".json")

        if not sidecar_path.exists():
            print(f"Error: Sidecar JSON not found: {sidecar_path}", file=sys.stderr)
            print(f"Please specify --sidecar explicitly", file=sys.stderr)
            sys.exit(1)
    else:
        sidecar_path = Path(sidecar_path)
        if not sidecar_path.exists():
            print(f"Error: Sidecar JSON not found: {sidecar_path}", file=sys.stderr)
            sys.exit(1)

    return sidecar_path


def load_frame_times(sidecar_path):
    """Load frame timing information from BIDS sidecar JSON.

    Args:
        sidecar_path: Path to sidecar JSON file

    Returns:
        frame_times_start: Array of frame start times in seconds
        frame_duration: Array of frame durations in seconds
        frame_time_middle: Array of frame middle times in seconds
    """
    sidecar_path = Path(sidecar_path)
    with open(sidecar_path, 'r') as f:
        sidecar = json.load(f)
        frame_times_start = np.array(sidecar["FrameTimesStart"])
        frame_duration = np.array(sidecar["FrameDuration"])
        frame_time_middle = frame_times_start + frame_duration / 2
    return frame_times_start, frame_duration, frame_time_middle


def downsample_dynamic_pet_2x(input_path, output_path, use_gpu=False, _rich_progress=None, _rich_task=None):
    """Downsample dynamic PET image by 2x2x2 using mean pooling.

    Fast downsampling using NumPy's reshape trick for mean pooling.
    Processes frame-by-frame to minimize memory usage.

    Args:
        input_path: Path to input 4D PET image
        output_path: Path for output downsampled image
        use_gpu: Unused (kept for API compatibility)
        _rich_progress: Internal - Rich Progress object (optional)
        _rich_task: Internal - Rich progress task ID (optional)

    Returns:
        output_img: The downsampled NIfTI image

    Example:
        downsample_dynamic_pet_2x("dpet.nii.gz", "dpet_2x.nii.gz")
    """
    import nibabel as nib
    import tempfile

    input_path = Path(input_path)
    output_path = Path(output_path)

    # Load input image
    input_img = nib.load(str(input_path))

    if len(input_img.shape) != 4:
        raise ValueError(f"Expected 4D image, got {len(input_img.shape)}D")

    n_frames = input_img.shape[3]
    in_shape = input_img.shape[:3]

    # Calculate output shape (divide by 2, floor)
    out_shape = tuple(s // 2 for s in in_shape)

    # Create memory-mapped array for output
    output_shape = out_shape + (n_frames,)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dat')
    temp_file.close()

    output_data = np.memmap(temp_file.name, dtype=np.float32, mode='w+', shape=output_shape)

    # Create progress bar
    frame_task = None
    if _rich_progress is not None:
        frame_task = _rich_progress.add_task("Downsampling frames", total=n_frames)

    # Timing accumulators
    import time
    time_load = 0
    time_pool = 0
    time_save = 0

    # Process each frame
    for i in range(n_frames):
        # Load frame (no copy - use dataobj directly)
        t0 = time.time()
        frame = input_img.dataobj[..., i]
        time_load += time.time() - t0

        # Apply 2x2x2 mean pooling with stride 2
        t0 = time.time()
        # Crop to even dimensions first
        frame_cropped = frame[:out_shape[0]*2, :out_shape[1]*2, :out_shape[2]*2]

        # Reshape and mean over 2x2x2 blocks
        downsampled = frame_cropped.reshape(
            out_shape[0], 2,
            out_shape[1], 2,
            out_shape[2], 2
        ).mean(axis=(1, 3, 5))
        time_pool += time.time() - t0

        # Save to memmap
        t0 = time.time()
        output_data[..., i] = downsampled
        time_save += time.time() - t0

        # Flush periodically
        if i % 8 == 0:
            output_data.flush()

        if frame_task is not None:
            _rich_progress.advance(frame_task)

    # Print timing breakdown
    print(f"\nTiming breakdown ({n_frames} frames):")
    print(f"  Load from disk:    {time_load:.3f}s ({time_load/n_frames*1000:.1f}ms/frame)")
    print(f"  NumPy pooling:     {time_pool:.3f}s ({time_pool/n_frames*1000:.1f}ms/frame)")
    print(f"  Save to memmap:    {time_save:.3f}s ({time_save/n_frames*1000:.1f}ms/frame)")
    print(f"  Total:             {time_load+time_pool+time_save:.3f}s")

    # Final flush
    output_data.flush()

    # Update affine to reflect 2x voxel size
    new_affine = input_img.affine.copy()
    new_affine[:3, :3] *= 2  # Double voxel size

    # Create output image
    output_img = nib.Nifti1Image(output_data, new_affine)

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output_img, str(output_path))

    # Clean up
    del output_data
    Path(temp_file.name).unlink()

    return nib.load(str(output_path))


