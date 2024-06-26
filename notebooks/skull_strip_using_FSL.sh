# This code is for skull stripping using FSL Brain Extraction Tool (BET)

#!/bin/bash

root_dir="/Users/pesala/Downloads/IXI-T1"
output_dir="/Users/pesala/Downloads/Healthy_skullstripped"

for file_path in "$root_dir"/*; do
    if [[ $file_path == *.nii.gz ]]; then
        filename=$(basename "$file_path")
        output_path="$output_dir/${filename}"

        # Adjust the -f parameter as needed
        bet "$file_path" "$output_path" -f 0.5 -g 0

        echo "Processed $filename"
    fi
done

echo "All files have been processed."
