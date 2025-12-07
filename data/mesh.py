
## @package mesh
# @brief Generate meshes from .geo files using gmsh in parallel.

import multiprocessing
import gmsh
import os
from pathlib import Path
from tqdm import tqdm
from functools import partial

##
# @param file_path (str): Path to the .geo file.
# @param data_folder (str): Path to the data folder.
def generate_mesh_from_geo(file_path: str, data_folder: str = "test") -> None:
    """Generate a mesh from a .geo file using gmsh."""

    # Load the .geo file
    gmsh.open(file_path)

    # Generate 2D or 3D mesh depending on your .geo setup
    gmsh.model.mesh.generate(2)
      
    # Create mesh folder if it doesn't exist
    msh_output_folder = Path(f"{data_folder}/msh")
    msh_output_folder.mkdir(parents=True, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    msh_path = os.path.join(msh_output_folder, base_name + ".msh")

    # Write the mesh to a .msh file 
    gmsh.write(msh_path)


##
# @param i (int): Index of the geometry.
# @param geometry_path (str): Path to the directory containing .geo files.
# @param mesh_path (str): Path to the directory where .msh files will be saved.
def generate_mesh(i: int, data_folder: str = "test") -> None:
    """
    Generate a mesh for a given geometry index.
    """
    # Generate the mesh for each geometry
    geo_path = f"{data_folder}/geo/{i}.geo"
    generate_mesh_from_geo(geo_path, data_folder)


##
# @param data_folder (str): Path to the data folder.
# @param use_multiprocessing (bool): Whether to use multiprocessing for mesh generation.
# @param use_all_cores (bool): Whether to use all CPU cores for multiprocessing.
# @param empty_mesh_folder (bool): Whether to empty the mesh folder before generating new meshes.
# @note We added empty_mesh_folder since the generation of the meshes could take too long, or could be interrupted for some reason. 
# In those cases you might want to keep the existing meshes and start again the procedure, but only generating the missing ones.
# @note While we are using multiprocessing to speed up the mesh generation, gmsh doesn't support gpus, so the speedup will be limited by the CPU capabilities.
def generate_meshes(data_folder: str = "test", use_multiprocessing: bool = True, use_all_cores: bool = False, empty_mesh_folder: bool = True) -> None:
    """
    Generate meshes for all geometries present in the specified directory using multiprocessing.
    Displays a fancy animated progress bar.
    """

    # Initialize gmsh
    gmsh.initialize()

    # Suppress gmsh terminal messages
    gmsh.option.setNumber("General.Terminal", 0)

    # Empty mesh directory if it exists, else create it
    if empty_mesh_folder:
        msh_output_folder = Path(f"{data_folder}/msh")
        if msh_output_folder.exists():
            for file in msh_output_folder.iterdir():
                if file.is_file():
                    file.unlink()
        else:
            msh_output_folder.mkdir(parents=True, exist_ok=True)
    else:
        msh_output_folder = Path(f"{data_folder}/msh")
        if not msh_output_folder.exists():
            msh_output_folder.mkdir(parents=True, exist_ok=True)

    # Get list of .geo files
    geo_folder_path = Path(f"{data_folder}/geo")
    geos = list(geo_folder_path.iterdir())

    # Process only the geometries that don't have a corresponding mesh yet
    geos = [geo for geo in geos if not (msh_output_folder / f"{geo.stem}.msh").exists()]
    r = [int(geo.stem) for geo in geos]

    if not use_multiprocessing:
        for i in tqdm(
            r,
            total=len(r),
            desc="🚀 Generating meshes",
            ncols=100,
            bar_format="{desc} |{bar}| {percentage:3.0f}% [{n}/{total}] ⏱️ {elapsed} ETA {remaining}",
            colour='magenta'
        ):
            generate_mesh(i, f"{data_folder}/geo", f"{data_folder}/msh")
    else:
        if use_all_cores:
            num_workers = multiprocessing.cpu_count()  # max cores
        else:
            num_workers = max(1, multiprocessing.cpu_count() - 1)  # leave one core free
        
        generate_mesh_with_opts = partial(generate_mesh, data_folder=data_folder)

        with multiprocessing.Pool(num_workers) as pool:
            # Fancy progress bar
            for _ in tqdm(
                pool.imap_unordered(generate_mesh_with_opts, r),
                total=len(r),
                desc="🚀 Generating meshes",
                ncols=100,
                bar_format="{desc} |{bar}| {percentage:3.0f}% [{n}/{total}] ⏱️ {elapsed} ETA {remaining}",
                colour='blue'
            ):
                pass

    # Finalize gmsh
    gmsh.finalize()