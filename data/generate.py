## @file generate.py
## @brief Main script to generate geometries, meshes, datasets, and plot results.

data_folder = "test"

from .geometry import read_data_file
names, ranges, num_points = read_data_file("test.csv")

from .test import test
test(
    names=names,
    ranges=ranges,
    num_points=num_points,
    geometry_input="geometry.geo",
    parameters_file_name="parameters.csv",
    data_folder=data_folder,
    plot_number=1
)





