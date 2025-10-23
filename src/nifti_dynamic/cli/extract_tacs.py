"""CLI command for extracting TACs from segmentation."""

from typing import Annotated, Optional
from pathlib import Path
import typer


def extract_tacs(
    pet: Annotated[Path, typer.Option(exists=True, file_okay=True, dir_okay=False, help="Dynamic PET image path (NIfTI format)")],
    segmentation: Annotated[Path, typer.Option(exists=True, file_okay=True, dir_okay=False, help="Segmentation image with ROI labels (NIfTI format)")],
    output: Annotated[Path, typer.Option(help="Output directory path for TAC CSV files")],
    sidecar: Annotated[Optional[Path], typer.Option(exists=True, file_okay=True, dir_okay=False, help="Sidecar JSON with frame timing (defaults to PET path with .json)")] = None,
    max_roi_size_factor: Annotated[float, typer.Option(help="Memory factor for large ROIs (higher = faster but more RAM)")] = 2.0,
):
    """Extract time-activity curves (TACs) for all ROIs in a segmentation image.

    Extracts TACs from all non-zero labels in the segmentation and saves each
    as a separate CSV file with columns: time, mu, std, n_voxels.
    """
    # Heavy imports only after argument parsing
    import sys
    import nibabel as nib
    import numpy as np
    from nibabel.processing import resample_from_to
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from nifti_dynamic.tacs import extract_multiple_tacs, save_tac
    from nifti_dynamic.utils import load_frame_times, get_sidecar_path

    output.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    dynpet = nib.load(str(pet))
    sidecar_path = get_sidecar_path(pet, sidecar)
    frame_times_start, frame_duration, frame_time_middle = load_frame_times(sidecar_path)

    seg_img = nib.load(str(segmentation))
    seg_img = resample_from_to(seg_img, (dynpet.shape[:3], dynpet.affine), order=0)
    seg_data = seg_img.get_fdata()

    unique_labels = np.unique(seg_data)
    unique_labels = unique_labels[unique_labels != 0]
    n_rois = len(unique_labels)
    print(f"Found {n_rois} ROIs to extract")

    # Extract TACs with rich progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
    ) as progress:
        tacs_mean, tacs_std, tacs_n = extract_multiple_tacs(
            dynpet,
            seg_data,
            max_roi_size_factor=max_roi_size_factor,
            _rich_progress=progress,
            _rich_task=None
        )

    # Save TACs to individual CSV files
    print(f"Saving {len(tacs_mean)} TAC files...")
    for label in tacs_mean.keys():
        tac_filename = output / f"tac_label_{label:03d}.csv"
        save_tac(tac_filename, tacs_mean[label], tacs_std[label], tacs_n[label], time=frame_time_middle)

    print(f"Completed: {output} ({len(tacs_mean)} TACs saved)")


def cli_extract_tacs():
    """CLI entrypoint for extract_tacs."""
    typer.run(extract_tacs)
