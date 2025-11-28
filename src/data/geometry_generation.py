## @package geometry_generation
# @brief Functions to generate datasets by processing the reference geometry file.



import contextlib
import os
import numpy as np
from remove import reset_environment
import sys
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import shutil



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
def reset_data():
    """
    Reset the data folder silently.
    """
    # Suppress print statements from reset_environment
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            reset_environment()
    
    data_folder = "data"
    parameters_file = os.path.join(data_folder, "parameters.csv")
    
    # Ensure folder exists
    os.makedirs(data_folder, exist_ok=True)
    
    # Reset parameters.csv
    with open(parameters_file, "w") as csv_file:
        csv_file.write("ID,Overetch,Distance,Mode1,Mode2,Mode3,Mode4\n")



## 
# @param overetch_range (tuple): Range of overetch values (min, max).
# @param distance_range (tuple): Range of distance values (min, max).
# @param coeff_range (tuple): Range of deformation coefficient values (min, max).
# @param directory (str): Directory to save the generated geometry files.
# @param geometry_input (str): Path to the input geometry file.
def generate_geometries(overetch_range: tuple, distance_range: tuple, coeff_range: tuple, directory: str, geometry_input: str = "geometry.geo"):
    """
    Generate geometries by modifying the distance, overetch, and coefficients of deformation of the plates.
    This function creates a series of geometry files with different parameters.
    """
    overetches = np.linspace(overetch_range[0], overetch_range[1], 5)
    distances = np.linspace(distance_range[0], distance_range[1], 5)
    coeff1 = np.linspace(coeff_range[0], coeff_range[1], 5)
    coeff2 = np.linspace(coeff_range[0], coeff_range[1], 5)
    coeff3 = np.linspace(coeff_range[0], coeff_range[1], 5)
    coeff4 = np.linspace(coeff_range[0], coeff_range[1], 5)

    # Ensure the parameters.csv file is empty before writing
    with open("data/parameters.csv", "w") as csv_file:
        csv_file.write("ID ,Overetch,Distance,Coeff1,Coeff2,Coeff3,Coeff4\n")
        csv_file.truncate()
    
    reset_data()
    total = len(overetches) * len(distances) * len(coeff1) * len(coeff2) * len(coeff3) * len(coeff4)

    j = 1
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None, complete_style="green", finished_style="bright_green", pulse_style="yellow"),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn()
    ) as progress:
        task = progress.add_task("Generating geometries", total=total)
        while not progress.finished:
            for o in overetches:
                for d in distances:
                    for c1 in coeff1:
                        for c2 in coeff2:
                            for c3 in coeff3:
                                for c4 in coeff4:
                                    modify_quantity("geometry.geo", "data/" + directory, str(j), "overetch", o)
                                    modify_quantity(os.path.join("data/" + directory, str(j) + ".geo"), "data/" + directory, str(j), "distance", d)
                                    modify_quantity(os.path.join("data/" + directory, str(j) + ".geo"), "data/" + directory, str(j), "coeff(1)", c1)
                                    modify_quantity(os.path.join("data/" + directory, str(j) + ".geo"), "data/" + directory, str(j), "coeff(2)", c2)
                                    modify_quantity(os.path.join("data/" + directory, str(j) + ".geo"), "data/" + directory, str(j), "coeff(3)", c3)
                                    modify_quantity(os.path.join("data/" + directory, str(j) + ".geo"), "data/" + directory, str(j), "coeff(4)", c4)
                                    with open("data/parameters.csv", "a") as csv_file:
                                        csv_file.write(f"{j},{o},{d},{c1},{c2},{c3},{c4}\n")
                                    j += 1
                                    progress.update(task, advance=1)
                                    
