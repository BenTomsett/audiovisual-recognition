#!/bin/bash

# Directory where the original .mp4 files are located
SOURCE_DIR="./videos_cropped"

# Directory where the cropped videos will be saved
DEST_DIR="./videos_cropped_2"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Loop through all .mp4 files in the source directory
for file in "$SOURCE_DIR"/**/*.mp4; do
    # Extract the filename without the extension
    filename=$(basename -- "$file")
    base="${filename%.*}"

    # Define the output file path
    output="$DEST_DIR/${base}_cropped.mp4"

    # Run ffmpeg to crop the video
    ffmpeg -loglevel error -i "$file" -vf "crop=176:96" "$output"
done

