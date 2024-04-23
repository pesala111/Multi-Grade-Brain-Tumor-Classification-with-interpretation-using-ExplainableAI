#!/bin/bash

# This code is to change the orientation of the NIfTY file from sagittal to axial using FSL

input_dir="/Users/pesala/Downloads/Healthy_skullstripped"

output_dir="/Users/pesala/Documents/Healthy_axial"
mkdir -p "$output_dir"

# Loop through NIfTI files in the input directory
for file_path in "$input_dir"/*.nii.gz; do
    filename=$(basename "$file_path")
    output_path="$output_dir/$filename"

    # Swap dimensions to make axial the primary plane
    # Adjust the dimension arguments as per your specific images' needs
    fslswapdim "$file_path" -z -x y "$output_path"

    echo "Reoriented $filename to axial"
done

echo "All files have been processed."
