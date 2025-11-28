## @file main.py
# @brief Main script to generate geometries, meshes, datasets, and plot results.


from geometry_generation import generate_geometries
from mesh_generation import generate_meshes
from dataset_generation import generate_datasets, combine_temp_files

#generate_geometries(overetch_range=(0.0, 0.5), distance_range=(1.5, 2.5), coeff_range=(-0.15, 0.15), directory="geo")
#generate_meshes(geometry_path="data/geo", mesh_path="data/msh", use_multiprocessing=True, use_all_cores=True)
generate_datasets()