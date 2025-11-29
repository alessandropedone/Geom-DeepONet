## @package clean
# @brief Functions to clean up the environment.

## @file clean.py
# @brief Functions to clean up the environment.

import os
import contextlib

## 
# @param data_folder (str): Path to the data folder containing the msh subfolder.
def remove_msh_files(data_folder: str = "data"):
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
# @param data_folder (str): Path to the data folder to reset.
def setup_data(data_folder: str = "data", parameters_head : str ="ID,Overetch,Distance,Mode1,Mode2,Mode3,Mode4\n", parameters_file_name: str ="parameters.csv"):
    """
    Reset the data folder silently.
    """
    # Suppress print statements from reset_environment
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            # remove all the files inside it
            for root, dirs, files in os.walk(data_folder, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Error removing {file_path}: {e}")         
    
    parameters_file = os.path.join(data_folder, parameters_file_name)
    
    # Ensure folder exists
    os.makedirs(data_folder, exist_ok=True)
    
    # Reset parameters.csv
    with open(parameters_file, "w") as csv_file:
        csv_file.write(parameters_head)

    # Create geo subfolder
    geo_folder = os.path.join(data_folder, "geo")
    os.makedirs(geo_folder, exist_ok=True)
    # Create mshfiles subfolder
    mshfiles_folder = os.path.join(data_folder, "msh")
    os.makedirs(mshfiles_folder, exist_ok=True)
    # Create results subfolder
    results_folder = os.path.join(data_folder, "results")
    os.makedirs(results_folder, exist_ok=True)