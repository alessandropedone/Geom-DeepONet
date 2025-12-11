## @package geometry
# @brief Functions to generate datasets by processing the reference geometry file.


import os
import numpy as np
from itertools import product
from tqdm import tqdm
import csv


##
# @param parameters_head (str): Header line for the parameters CSV file.
# @param parameters_file_name (str): Name of the parameters CSV file.
# @param ignore_data (bool): Whether to ignore existing other data present in data_folder.
# @param data_folder (str): Path to the data folder to reset.
# @note In any case, this function ensures the necessary subfolders and parameters file are set up.
def setup_data(parameters_head : str, 
               parameters_file_name: str, 
               ignore_data: bool = False,
               data_folder: str = "test"):
    """
    Reset the data folder silently.
    """
    # Suppress print statements from reset_environment
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):

            # If the folder is not present create it
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)

            if not ignore_data:
                # remove all the files inside it
                for root, dirs, files in os.walk(data_folder, topdown=False):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            print(f"Error removing {file_path}: {e}")
            else:
                # Clean only geo subfolder
                geo_folder = os.path.join(data_folder, "geo")
                for root, dirs, files in os.walk(geo_folder, topdown=False):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            print(f"Error removing {file_path}: {e}")
    
    # Parameters file path
    parameters_file = os.path.join(data_folder, parameters_file_name)
    
    # Setup parameters file
    with open(parameters_file, "w") as csv_file:
        csv_file.write(parameters_head)
        csv_file.truncate()

    # Create geo subfolder
    geo_folder = os.path.join(data_folder, "geo")
    os.makedirs(geo_folder, exist_ok=True)

    # Create mshfiles subfolder
    mshfiles_folder = os.path.join(data_folder, "msh")
    os.makedirs(mshfiles_folder, exist_ok=True)

    # Create results subfolder
    results_folder = os.path.join(data_folder, "results")
    os.makedirs(results_folder, exist_ok=True)


## 
# @param file_path (str): Path to the data file.
def read_data_file(file_path: str = "test.csv"):
    """Read the data file and return names, ranges, and num_points."""
    names = []
    ranges = []
    num_points = []

    with open(file_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row['name'])
            ranges.append((float(row['min']), float(row['max'])))
            num_points.append(int(row['num_points']))

    return names, ranges, num_points


##
# @param input_path (str): Path to the geometry file (e.g., geometry.geo).
# @param output_path (str): Path to save the modified geometry file (e.g., ./).
# @param name (str): Name for the new geometry file (e.g., test).
# @param quantity(str): Quantity to modify (e.g., 'distance', 'overetch', 'coeff(1)', etc.).
# @param value (float): New quantity value to set (e.g., 2.0).
def modify_quantity(input_path: str, output_path: str, name: str, quantity: str, value: float):
    """Modify the geometry file to change the distance between the plates."""
    # Open the geometry.geo file to read the lines
    with open(str(input_path), "r") as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:

        if str(quantity) + ' =' in line:
            # Split the line by '='
            parts = line.split('=')
            # Update the distance value
            new_line = f"{parts[0]}= {value};\n"
            new_lines.append(new_line)

        else:
            # If the line doesn't match any of the points, keep it unchanged
            new_lines.append(line)
    
    # Define the directory and file name for saving the new geometry
    directory = str(output_path)
    file_name = str(name) + ".geo"
    
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Define the full file path
    file_path = os.path.join(directory, file_name)
    
    # Write the modified lines to the new geometry file
    with open(file_path, "w") as f:
        f.writelines(new_lines)

    #print(f"Geometry updated setting {quantity} to {value}. Saved to {file_path}")



## 
# @param names (list[str]): List of the names, that appear in the geometry file of the quantities.
# @param ranges (list[tuple]): List of ranges for each quantity.
# @param num_points (list[int]): List of number of points to generate within the specified ranges for each quantity.
# @param geometry_input (str): Path to the input geometry file.
# @param data_folder (str): Path to the data folder.
# @param ignore_data (bool): Whether to ignore existing other data present in data_folder. 
# For example you may want to set this equal to TRUE if you have more data in that folder, 
# or already computed solutions, and for some reason you want to generate the geomteries 
# in that folder avoid touching every file except the "geo" subfolder the parameters file.
# @note As output you get a series of geometry files in data_folder/geo and a parameters.csv file in data_folder.
def generate_geometries(names: list[str],
                        ranges: list[tuple],
                        num_points: list[int],
                        geometry_input: str,
                        data_folder: str = "test",
                        parameters_file_name: str = "parameters.csv",
                        ignore_data: bool = False):
    """
    Generate geometries by modifying the list of quantities over specified ranges.
    This function creates a series of geometry files with different parameters.
    """
    # Ensure the provided lists are of the same length
    if not (len(ranges) == len(names) == len(num_points)):
        raise ValueError("The lengths of ranges, names, and num_points must be the same.")
    
    # Create the quantities matrix: list of np.linspace for each range
    quantities = []
    for i in range(len(ranges)):
        quantities.append(np.linspace(ranges[i][0], ranges[i][1], num_points[i]))

    # Setup data_folder directory
    setup_data(parameters_head= f"ID," + ",".join(names) + "\n", 
               parameters_file_name=parameters_file_name,
               ignore_data=ignore_data,
               data_folder=data_folder)

    # Generate all combinations of quantities
    params = list(product(*quantities))
    total = len(params)

    j = 1
    for param in tqdm(
                params,
                total=total,
                desc="🚀 Generating geometries",
                ncols=100,
                bar_format="{desc} |{bar}| {percentage:3.0f}% [{n}/{total}] ⏱️ {elapsed} ETA {remaining}",
                colour='blue'
            ):
        
        # Modify the jth geometry
        geo_folder = os.path.join(data_folder, "geo")
        geo_path = os.path.join(data_folder, "geo", f"{j}.geo")
        modify_quantity(geometry_input, geo_folder, str(j), names[0], param[0])
        for i in range(1, len(names)):
            modify_quantity(geo_path, geo_folder, str(j), names[i], param[i])

        # Save the corresponding geometric parameters
        with open(os.path.join(data_folder, parameters_file_name), "a") as csv_file:
            csv_file.write(f"{j}," + ",".join([str(p) for p in param]) + "\n")

        j += 1
                                    
