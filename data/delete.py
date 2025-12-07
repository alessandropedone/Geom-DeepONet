## @package delete
# @brief Functions to delete the data folder.

import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="test", help="Path to the data folder to delete.")

data_folder = parser.parse_args().folder
shutil.rmtree(data_folder)