#!/bin/bash

# Directory containing the video files
VIDEO_DIR="./videos_cropped"

# Directory to store the extracted audio files
AUDIO_DIR="./audio"

# Create the audio directory if it doesn't exist
mkdir -p "$AUDIO_DIR"

# Loop through all video files in the video directory
for VIDEO_FILE in "$VIDEO_DIR"/*; do
    # Extract the file name without extension
    BASENAME=$(basename "$VIDEO_FILE" | sed 's/\.[^.]*$//')

    # Output file path
    OUTPUT_FILE="${AUDIO_DIR}/${BASENAME}.wav"

    # Extract audio using FFmpeg
    ffmpeg -i "$VIDEO_FILE" -vn -acodec pcm_s16le -ar 44100 -ac 1 "$OUTPUT_FILE"
done
