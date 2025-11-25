## @file main.py
# @brief Main script to generate geometries, meshes, datasets, and plot results.


from geometry_generation import generate_geometries

generate_geometries(overetch_range=(0.0, 0.5), distance_range=(1.5, 2.5), coeff_range=(-0.2, 0.2), directory="geo")