"""CLI command for extracting input function from dynamic PET."""

from typing import Annotated, Optional
from pathlib import Path
from enum import Enum
import typer


class SegmentChoice(str, Enum):
    """Aorta segment choices."""
    ASCENDING = "ASCENDING"
    TOP = "TOP"
    DESCENDING = "DESCENDING"
    DESCENDING_BOTTOM = "DESCENDING_BOTTOM"


def extract_input_function(
    pet: Annotated[Path, typer.Option(exists=True, file_okay=True, dir_okay=False, help="Dynamic PET image path (NIfTI format)")],
    totalseg: Annotated[Path, typer.Option(exists=True, file_okay=True, dir_okay=False, help="TotalSegmentator segmentation image path (NIfTI format)")],
    output: Annotated[Path, typer.Option(help="Output directory path")],
    sidecar: Annotated[Optional[Path], typer.Option(exists=True, file_okay=True, dir_okay=False, help="Sidecar JSON with frame timing (defaults to PET path with .json)")] = None,
    segment: Annotated[SegmentChoice, typer.Option(help="Aorta segment to extract VOI from")] = SegmentChoice.DESCENDING_BOTTOM,
    volume: Annotated[float, typer.Option(help="VOI volume in milliliters")] = 1.0,
    cylinder_width: Annotated[int, typer.Option(help="ROI cylinder width in pixels")] = 3,
    aorta_index: Annotated[int, typer.Option(help="Aorta label index in TotalSegmentator output")] = 52,
    skip_visualization: Annotated[bool, typer.Option(help="Skip generating visualization image")] = False,
):
    """Extract input function from dynamic PET using aorta segmentation.

    Processes dynamic PET images with TotalSegmentator aorta masks to extract
    cylindrical VOIs (Volumes of Interest) for input function calculation.
    """
    # Heavy imports only after argument parsing (delays 30+ seconds of imports)
    import sys
    import nibabel as nib
    from nibabel.processing import resample_from_to
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from nifti_dynamic.aorta_rois import AortaSegment, pipeline
    from nifti_dynamic.tacs import extract_tac, save_tac
    from nifti_dynamic.utils import load_frame_times, get_sidecar_path

    output.mkdir(parents=True, exist_ok=True)
    sidecar_path = get_sidecar_path(pet, sidecar)

    # Convert enum to string for processing
    segment_str = segment.value

    # Processing with rich progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Processing", total=7)

        # Load PET
        progress.update(task, description="Loading PET...")
        dynpet = nib.load(str(pet))
        progress.advance(task)

        # Load sidecar
        progress.update(task, description="Loading sidecar...")
        frame_times_start, frame_duration, frame_time_middle = load_frame_times(sidecar_path)
        progress.advance(task)

        # Load segmentation
        progress.update(task, description="Loading segmentation...")
        totalseg_img = nib.load(str(totalseg))
        totalseg_img = resample_from_to(totalseg_img, (dynpet.shape[:3], dynpet.affine), order=0)
        progress.advance(task)

        # Extract aorta
        progress.update(task, description="Extracting aorta...")
        aorta = nib.Nifti1Image(
            (totalseg_img.get_fdata() == aorta_index).astype("uint8"),
            affine=totalseg_img.affine
        )
        aorta_voxels = int(aorta.get_fdata().sum())
        if aorta_voxels == 0:
            print(f"\nError: No aorta voxels found with index {aorta_index}", file=sys.stderr)
            sys.exit(1)
        progress.advance(task)

        # Run pipeline
        progress.update(task, description=f"Running pipeline ({segment_str})...")
        visualization_path = None if skip_visualization else str(output / "visualization.jpg")
        segment_enum = getattr(AortaSegment, segment_str)

        aorta_segments, aorta_vois = pipeline(
            aorta_mask=aorta,
            dpet=dynpet,
            frame_times_start=frame_times_start,
            cylinder_width=cylinder_width,
            volume_ml=volume,
            segment=segment_enum,
            image_path=visualization_path
        )
        progress.advance(task)

        # Extract TAC from the VOI
        progress.update(task, description="Extracting TAC...")
        voi_mask = aorta_vois.get_fdata() == segment_enum.value
        tac_mean, tac_std, n_voxels = extract_tac(dynpet, voi_mask)
        progress.advance(task)

        # Save results
        progress.update(task, description="Saving results...")
        nib.save(aorta_segments, str(output / "aorta_segments.nii.gz"))
        nib.save(aorta_vois, str(output / "aorta_vois.nii.gz"))

        # Save TAC with time information
        tac_filename = f"input_fun_{segment_str}_{volume}ml_{cylinder_width}px.csv"
        tac_path = output / tac_filename
        save_tac(tac_path, tac_mean, tac_std, n_voxels, time=frame_time_middle)
        progress.advance(task)

    print(f"Completed: {output}")


def cli_extract_input_function():
    """CLI entrypoint for extract_input_function."""
    typer.run(extract_input_function)
