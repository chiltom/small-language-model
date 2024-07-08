import os
import shutil
import tarfile
from dotenv import load_dotenv

load_dotenv()

# Define the path to the main directory using a raw string to handle backslashes
main_directory = os.getenv("folder_path")

# Walk through the directory structure
for root, dirs, files in os.walk(main_directory):
    for file in files:
        # Check if the file is a .tar file
        if file.endswith('.tar'):
            # Construct the full file path
            tar_path = os.path.join(root, file)
            # Open the .tar file
            with tarfile.open(tar_path) as tar:
                # Extract all the contents of the .tar file to the main directory
                tar.extractall(path=main_directory)

# Now that all files are extracted, we can flatten the directory
for root, dirs, files in os.walk(main_directory):
    for file in files:
        # Construct the full file path
        file_path = os.path.join(root, file)
        # Construct the destination path
        destination_path = os.path.join(main_directory, file)
        # Move the file to the main directory if it's not already there
        if root != main_directory:
            shutil.move(file_path, destination_path)

# Print a success message
print("All .tar files have been extracted, and the directory has been flattened.")
