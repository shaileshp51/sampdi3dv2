#!/usr/bin/bash

# Save the current working directory
cwd=$(pwd)

# Define an array of target folders
folders=("dna" "protein")

for folder in "${folders[@]}"; do
  # Check if the folder exists
  if [ -d "$folder" ]; then
    cd "$folder" || exit 1  # Navigate to the folder, exit if it fails
    # Find and delete files and subfolders excluding "example_*"
    find . -mindepth 1 ! -path "./example_*" -exec rm -rf {} +
    cd "$cwd" || exit 1  # Return to the original directory
  else
    echo "Folder $folder does not exist."
  fi
done
