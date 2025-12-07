## @package dataset
# @brief Functions to generate datasets by processing mesh files in parallel 
# and then combine the results in a single CSV file (dataset.csv).

from pathlib import Path
import multiprocessing
from multiprocessing import cpu_count
from tqdm import tqdm
from functools import partial

from .fom import solvensave


##
# @param data_folder (str): path to the data folder.
def reset_results(data_folder: str = "test"):
    """Emtpy the results folder by removing all files inside it."""
    # Ensure the results folder is empty before running the script
    results_folder = Path(data_folder) / "results"
    if results_folder.exists():
        for file in results_folder.iterdir():
            if file.is_file():
                file.unlink()
    else:
        results_folder.mkdir(parents=True)

##
# @param data_folder (str): path to the data folder.
# @param use_multiprocessing (bool): Whether to use multiprocessing for parallel processing.
# @param use_all_cores (bool): Whether to use all CPU cores for multiprocessing.
# @param empty_results_folder (bool): Whether to empty the results folder before generating new datasets.
def generate_datasets(data_folder: str = "test", use_multiprocessing: bool = True, use_all_cores: bool = False, empty_results_folder: bool = True):
    """Generate datasets by processing all mesh files in parallel and saving the results."""
    # Set up the environment
    if empty_results_folder:
        reset_results(data_folder)

    # Get the list of mesh files
    mesh_folder_path = Path(f"{data_folder}/msh")
    meshes = list(mesh_folder_path.iterdir())

    # Process only the meshes that don't have a corresponding results file yet
    results_folder_path = Path(f"{data_folder}/results")
    meshes = [mesh for mesh in meshes if not (results_folder_path / f"{mesh.stem}.h5").exists()]

    if not use_multiprocessing:
        for mesh in tqdm(
            meshes,
            total=len(meshes),
            desc="🚀 Generating solution datasets",
            ncols=100,
            bar_format="{desc} |{bar}| {percentage:3.0f}% [{n}/{total}] ⏱️ {elapsed} ETA {remaining}",
            colour='blue'
        ):
            solvensave(mesh)
    else:
        if use_all_cores:
            num_workers = cpu_count()  # max cores
        else:
            num_workers = max(1, cpu_count() - 1)  # leave one core free

        with multiprocessing.Pool(num_workers) as pool:
                # Fancy progress bar
                solvensave_with_opts = partial(solvensave, data_folder=data_folder)
                for _ in tqdm(
                    pool.imap_unordered(solvensave_with_opts, meshes),
                    total=len(meshes),
                    desc="🚀 Generating solution datasets",
                    ncols=100,
                    bar_format="{desc} |{bar}| {percentage:3.0f}% [{n}/{total}] ⏱️ {elapsed} ETA {remaining}",
                    colour='blue'
                ):
                    pass
        