"""CLI command for downsampling dynamic PET images."""

from typing import Annotated
from pathlib import Path
import typer


def resample_pet(
    pet: Annotated[Path, typer.Option(exists=True, file_okay=True, dir_okay=False, help="Dynamic PET image path (NIfTI format)")],
    output: Annotated[Path, typer.Option(help="Output downsampled PET image path")],
    use_gpu: Annotated[bool, typer.Option(help="Use GPU acceleration with CuPy")] = True,
):
    """Downsample dynamic PET image by 2x using fast mean pooling.

    Downsamples huge 4D PET images by 2x2x2 using mean pooling (averaging).
    Uses GPU acceleration for massive speedup if CuPy is installed.
    """
    # Heavy imports only after argument parsing
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from nifti_dynamic.utils import downsample_dynamic_pet_2x

    print(f"Downsampling {pet} by 2x2x2")

    # Downsample with rich progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
    ) as progress:
        downsample_dynamic_pet_2x(
            pet,
            output,
            use_gpu=use_gpu,
            _rich_progress=progress,
            _rich_task=None
        )

    print(f"Completed: {output}")


def cli_resample_pet():
    """CLI entrypoint for resample_pet."""
    typer.run(resample_pet)
