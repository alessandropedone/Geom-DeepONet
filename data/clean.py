## @package clean
# @brief Functions to clean up the environment.

## @file clean.py
# @brief Functions to clean up the environment.

import os
import contextlib

## 
# @param data_folder (str): Path to the data folder containing the msh subfolder.
def remove_msh_files(data_folder: str = "test"):
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
    with open(os.path.join(data_folder, parameters_file_name), "w") as csv_file:
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