
## @package mesh_generation
# @brief Generate meshes from .geo files using gmsh in parallel.

import multiprocessing
import gmsh
import os
from pathlib import Path
from tqdm import tqdm
from functools import partial

##
# @param file_path (str): Path to the .geo file.
# @param mesh_path (str): Path to the directory where .msh files will be saved.
def generate_mesh_from_geo(file_path: str, mesh_path: str = "data/msh") -> None:
    """Generate a mesh from a .geo file using gmsh."""

    # Load the .geo file
    gmsh.open(file_path)

    # Generate 2D or 3D mesh depending on your .geo setup
    gmsh.model.mesh.generate(2)
      
    # Create mesh folder if it doesn't exist
    msh_output_folder = Path(mesh_path)
    msh_output_folder.mkdir(parents=True, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    msh_path = os.path.join(msh_output_folder, base_name + ".msh")

    # Write the mesh to a .msh file 
    gmsh.write(msh_path)


##
# @param i (int): Index of the geometry.
# @param geometry_path (str): Path to the directory containing .geo files.
# @param mesh_path (str): Path to the directory where .msh files will be saved.
def generate_mesh(i: int, geometry_path: str = "data/geo", mesh_path: str = "data/msh") -> None:
    """
    Generate a mesh for a given geometry index.
    """
    # Generate the mesh for each geometry
    geo_path = f"{geometry_path}/{i}.geo"
    generate_mesh_from_geo(geo_path, mesh_path)


##
# @param geometry_path (str): Path to the directory containing .geo files.
# @param mesh_path (str): Path to the directory where .msh files will be saved.
def generate_meshes(geometry_path: str = "data/geo", mesh_path: str = "data/msh", use_multiprocessing: bool = True, use_all_cores: bool = False) -> None:
    """
    Generate meshes for all geometries present in the specified directory using multiprocessing.
    Displays a fancy animated progress bar.
    """

    # Initialize gmsh
    gmsh.initialize()

    # Suppress gmsh terminal messages
    gmsh.option.setNumber("General.Terminal", 0)

    # Empty mesh directory if it exists, else create it
    msh_output_folder = Path(mesh_path)
    if msh_output_folder.exists():
        for file in msh_output_folder.iterdir():
            if file.is_file():
                file.unlink()
    else:
        msh_output_folder.mkdir(parents=True, exist_ok=True)

    geo_files = [name for name in os.listdir(geometry_path) if name.endswith('.geo')]
    num_geometries = len(geo_files)
    
    if num_geometries == 0:
        print("No .geo files found!")
        return

    r = range(1, num_geometries + 1)

    if not use_multiprocessing:
        for i in tqdm(
            r,
            total=num_geometries,
            desc="🚀 Generating meshes",
            ncols=100,
            bar_format="{desc} |{bar}| {percentage:3.0f}% [{n}/{total}] ⏱️ {elapsed} ETA {remaining}",
            colour='magenta'
        ):
            generate_mesh(i, geometry_path, mesh_path)
    else:
        if use_all_cores:
            num_workers = multiprocessing.cpu_count()  # max cores
        else:
            num_workers = max(1, multiprocessing.cpu_count() - 1)  # leave one core free
        
        generate_mesh_with_opts = partial(generate_mesh, geometry_path=geometry_path, mesh_path=mesh_path)

        with multiprocessing.Pool(num_workers) as pool:
            # Fancy progress bar
            for _ in tqdm(
                pool.imap_unordered(generate_mesh_with_opts, r),
                total=num_geometries,
                desc="🚀 Generating meshes",
                ncols=100,
                bar_format="{desc} |{bar}| {percentage:3.0f}% [{n}/{total}] ⏱️ {elapsed} ETA {remaining}",
                colour='magenta'
            ):
                pass

    # Finalize gmsh
    gmsh.finalize()