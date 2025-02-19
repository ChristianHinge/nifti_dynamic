try:
    import indexed_gzip
except ImportError:
    raise ImportError("The 'indexed_gzip' package is required for loading .nii.gz files fast. Please install it using 'pip install indexed_gzip'")

from .patlak import voxel_patlak
