
## @package mesh
# @brief Generate meshes from .geo files using gmsh in parallel.

import gmsh
import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


## 
# @param data_folder (str): Path to the data folder containing the msh subfolder.
def _remove_msh_files(data_folder: str = "test"):
    """
    Remove all .msh files in the subfolder msh of the specified data folder.
    """


    msh_folder = os.path.join(data_folder, "msh")
    for root, _, files in os.walk(msh_folder):
        for file in files:
            if file.endswith('.msh'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")

##
# @param file_path (str): Path to the .geo file.
# @param data_folder (str): Path to the data folder.
def _generate_mesh_from_geo(file_path: str, data_folder: str = "test") -> None:
    """Generate a mesh from a .geo file using gmsh."""

    # Initialize gmsh
    gmsh.initialize()   

    # Suppress gmsh terminal messages
    gmsh.option.setNumber("General.Terminal", 0)

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

    # Finalize gmsh
    gmsh.finalize()


##
# @param i (int): Index of the geometry.
# @param geometry_path (str): Path to the directory containing .geo files.
# @param mesh_path (str): Path to the directory where .msh files will be saved.
def _generate_mesh(i: int, data_folder: str = "test") -> None:
    """
    Generate a mesh for a given geometry index.
    """
    # Generate the mesh for each geometry
    geo_path = f"{data_folder}/geo/{i}.geo"
    generate_mesh_from_geo(geo_path, data_folder)


##
# @param data_folder (str): Path to the data folder.
# @param empty_mesh_folder (bool): Whether to empty the mesh folder before generating new meshes.
# @param max_workers (int): Maximum number of worker processes to use for parallel mesh generation.
def generate_meshes(data_folder: str = "test", empty_mesh_folder: bool = True, max_workers: int = 1) -> None:
    """
    Generate meshes for all geometries present in the specified directory using a variable number of workers.
    """

    msh_output_folder = Path(f"{data_folder}/msh")
    geo_folder_path = Path(f"{data_folder}/geo")

    # Empty mesh directory if it exists, else create it
    if empty_mesh_folder and msh_output_folder.exists():
        for file in msh_output_folder.iterdir():
            if file.is_file():
                file.unlink()
    msh_output_folder.mkdir(parents=True, exist_ok=True)

    # Get list of .geo files
    geos = [geo for geo in geo_folder_path.iterdir() if geo.suffix == ".geo"]
    


    # Process only the geometries that don't have a corresponding mesh yet
    geos = [geo for geo in geos if not (msh_output_folder / f"{geo.stem}.msh").exists()]
    geo_indices = [int(geo.stem) for geo in geos]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_mesh, idx, data_folder): idx for idx in geo_indices}
        for f in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="🚀 Generating meshes",
            ncols=100,
            bar_format="{desc} |{bar}| {percentage:3.0f}% [{n}/{total}] ⏱️ {elapsed} ETA {remaining}",
            colour='blue'
        ):
            _ = f.result()

    

