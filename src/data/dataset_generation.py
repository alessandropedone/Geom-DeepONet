## @package dataset_generation
# @brief Functions to generate datasets by processing mesh files in parallel 
# and then combine the results in a single CSV file (dataset.csv).

from pathlib import Path
import multiprocessing
from multiprocessing import cpu_count
from tqdm import tqdm
from fom import fom

##
# @param data_folder (str): path to the data folder.
def reset_results(data_folder: str = "data"):
    """Emtpy the results folder by removing all files inside it."""
    # Ensure the results folder is empty before running the script
    results_folder = Path(data_folder) / "results"
    if results_folder.exists():
        for file in results_folder.iterdir():
            if file.is_file():
                file.unlink()
    else:
        results_folder.mkdir(parents=True)

##
# @param data_folder (str): path to the data folder.
def generate_datasets(data_folder: str = "data"):
    """
    Generate datasets by processing all mesh files in parallel.
    This function sets up the environment, processes each mesh file,
    and saves the results in a temporary folder.
    """
    # Set up the environment
    set_up_environment()

    from pathlib import Path
    # Parallel processing
    mesh_folder_path = Path(f"{data_folder}/msh")
    meshes = list(mesh_folder_path.iterdir())

    num_workers = cpu_count()
    with multiprocessing.Pool(num_workers) as pool:
            # Fancy progress bar
            for _ in tqdm(
                pool.imap_unordered(process_mesh, meshes),
                total=len(meshes),
                desc="🚀 Generating datasets",
                ncols=100,
                bar_format="{desc} |{bar}| {percentage:3.0f}% [{n}/{total}] ⏱️ {elapsed} ETA {remaining}",
                colour='magenta'
            ):
                pass
        

##
# @param filename (str): path to the main dataset file.
# @param clean (bool): whether to clean up temporary files after combining.
# @param data_folder (str): path to the data folder.
def combine_temp_files(filename: str, clean: bool = True, data_folder: str = "data"):
    """
    Combine all temporary CSV files into the main dataset file.
    This function reads the main dataset file, updates it with the values from
    the temporary files, and writes the updated dataset back to the main file.
    It also cleans up the temporary files and folder.
    """
    # Define paths
    from pathlib import Path
    temp_folder = Path(f"{data_folder}/temp")
    dataset_file = Path(filename)

    import pandas as pd
    # Read the main dataset once
    dataset_df = pd.read_csv(dataset_file)

    # Collect updates from all temp files
    prefix = dataset_file.stem
    for temp_file in sorted(temp_folder.glob(f"{prefix}_*.csv"), key=lambda x: int(x.stem.split('_')[-1])):
        index = int(temp_file.stem.split('_')[-1]) - 1  # Adjust to 0-based index
        temp_df = pd.read_csv(temp_file)
        
        # Update the corresponding row
        if 0 <= index < len(dataset_df):
            dataset_df.iloc[index] = temp_df.iloc[0]
        print(f"Updated row {index + 1} from {temp_file.name}")

    # Write the dataset once after all updates
    dataset_df.to_csv(dataset_file, index=False)

    # Clean up temp files and folder
    if clean:
        prefix = dataset_file.stem.split('.')[0]
        for temp_file in temp_folder.glob(f"{prefix}_*.csv"):
            temp_file.unlink()
        # Only remove the temp folder if it is empty
        if not any(temp_folder.iterdir()):
            temp_folder.rmdir()


##
# @param data_folder (str): path to the data folder.
def unroll_normal_derivative_potential(data_folder: str = "data"):
    """
    Unroll the normal derivative potential dataset by creating a dataset with 
    one value of the normal derivative of the potetial one coordinate for each row.
    Structure of the dataset:
    - The first column contains the mesh number.
    - The following three columns contain the geometric parameters: 
        - the overecth of the upper plate,
        - the distance between the two plates,
        - the angle of the upper plate.
    - The following column contains the coordinate of the point on the lower edge of the upper plate.
    - The last column contains the normal derivative of the potential at that coordinate.
    This function reads the normal_derivative_potential.csv file, unrolls it,
    and saves the unrolled dataset in a new CSV file named unrolled_normal_derivative_potential.csv.
    """
    import pandas as pd
    # Read the normal derivative potential and coordinates datasets
    normal_df = pd.read_csv(f"{data_folder}/normal_derivative_potential.csv")
    coords_df = pd.read_csv(f"{data_folder}/coordinates.csv")

    # Initialize lists to store unrolled data
    mesh_numbers = []
    overetches = []
    distances = []
    angles = []
    coordinates = []
    normal_derivatives = []

    # Iterate through each row in the dataset
    for index, row in normal_df.iterrows():
        mesh_num = row.iloc[0] 
        overetch = row.iloc[1]
        distance = row.iloc[2]
        angle = row.iloc[3]
        
        # Get coordinate and normal derivative values (starting from column 4)
        coord_values = coords_df.iloc[index, 4:].values
        normal_values = row.iloc[4:].values
        
        # Add data for each coordinate point
        for coord, normal in zip(coord_values, normal_values):
            mesh_numbers.append(mesh_num.astype(int))  # Ensure mesh number is an integer
            overetches.append(overetch.astype(float))  # Ensure overetch is a float
            distances.append(distance.astype(float))
            angles.append(angle.astype(float))
            coordinates.append(coord.astype(float))  # Ensure coordinate is a float
            normal_derivatives.append(normal.astype(float))  # Ensure normal derivative is a float

    # Create the unrolled DataFrame
    unrolled_df = pd.DataFrame({
        'Mesh ID': mesh_numbers,
        'Overetch': overetches,
        'Distance': distances,
        'Angle': angles,
        'Coordinate': coordinates,
        'Normal Derivative': normal_derivatives
    })

    # Save the unrolled dataset
    unrolled_df.to_csv("data/unrolled_normal_derivative_potential.csv", index=False)

    