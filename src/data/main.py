## @file main.py
# @brief Main script to generate geometries, meshes, datasets, and plot results.


from geometry import generate_geometries
#generate_geometries(overetch_range=(0.0, 0.5), distance_range=(1.5, 2.5), coeff_range=(-0.15, 0.15))



from mesh import generate_meshes
generate_meshes(empty_mesh_folder=False)


from dataset import generate_datasets
generate_datasets(use_multiprocessing=True, empty_results_folder=False)

from clean import remove_msh_files
remove_msh_files()