"""CLI command for voxel-wise Patlak analysis."""

from typing import Annotated
from pathlib import Path
import typer


def run_voxel_patlak(
    pet: Annotated[Path, typer.Option(exists=True, file_okay=True, dir_okay=False, help="Dynamic PET image path (NIfTI format)")],
    input_function: Annotated[Path, typer.Option(exists=True, file_okay=True, dir_okay=False, help="Input function TAC CSV file (with time, mu, std, n_voxels columns)")],
    output: Annotated[Path, typer.Option(help="Output directory path")],
    axial_chunk_size: Annotated[int, typer.Option(help="Axial chunk size for processing (higher = faster but more RAM)")] = 8,
    gaussian_filter_size: Annotated[int, typer.Option(help="Gaussian smoothing sigma (0 = no smoothing)")] = 0,
    n_frames_linear_regression: Annotated[int, typer.Option(help="Number of frames to use for linear regression")] = 10,
):
    """Run voxel-wise Patlak analysis on dynamic PET data.

    Performs Patlak kinetic modeling on each voxel to generate Ki (slope) and
    intercept parametric maps.
    """
    # Heavy imports only after argument parsing
    import sys
    import csv
    import nibabel as nib
    import numpy as np
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from nifti_dynamic.patlak import voxel_patlak

    output.mkdir(parents=True, exist_ok=True)

    # Processing with rich progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Processing", total=4)

        # Load PET
        progress.update(task, description="Loading PET image...")
        dynpet = nib.load(str(pet))
        n_frames = dynpet.shape[3] if len(dynpet.shape) == 4 else 1
        progress.advance(task)

        # Load input function CSV
        progress.update(task, description="Loading input function...")
        with open(input_function, 'r', newline='') as f:
            reader = csv.reader(f)
            headers = next(reader)

            # Check for required columns
            if 'time' not in headers or 'mu' not in headers:
                print(f"Error: Input function CSV must have 'time' and 'mu' columns", file=sys.stderr)
                sys.exit(1)

            data = {header: list(column) for header, column in zip(headers, zip(*reader))}
            frame_times = np.array(data["time"]).astype(float)
            if_tac = np.array(data["mu"]).astype(float)

        if len(if_tac) != n_frames:
            print(f"Error: Input function has {len(if_tac)} frames but PET has {n_frames} frames", file=sys.stderr)
            sys.exit(1)
        progress.advance(task)

        # Run Patlak analysis
        progress.update(task, description="Running Patlak analysis...")
        ki_map, intercept_map = voxel_patlak(
            dynpet,
            if_tac,
            frame_times,
            gaussian_filter_size=gaussian_filter_size,
            n_frames_linear_regression=n_frames_linear_regression,
            axial_chunk_size=axial_chunk_size,
            _rich_progress=progress,
            _rich_task=task
        )
        progress.advance(task)

        # Save results
        progress.update(task, description="Saving results...")
        nib.save(ki_map, str(output / "patlak_ki.nii.gz"))
        nib.save(intercept_map, str(output / "patlak_intercept.nii.gz"))
        progress.advance(task)

    print(f"Completed: {output}")


def cli_voxel_patlak():
    """CLI entrypoint for voxel_patlak."""
    typer.run(run_voxel_patlak)
