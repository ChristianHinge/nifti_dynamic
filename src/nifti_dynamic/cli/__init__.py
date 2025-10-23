"""Command-line interface for nifti_dynamic library."""

from .extract_input_function import cli_extract_input_function
from .voxel_patlak import cli_voxel_patlak
from .extract_tacs import cli_extract_tacs
from .resample_pet import cli_resample_pet

__all__ = [
    "cli_extract_input_function",
    "cli_voxel_patlak",
    "cli_extract_tacs",
    "cli_resample_pet",
]
