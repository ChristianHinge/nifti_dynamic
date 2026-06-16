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
