#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

DIR=$1

count=$(find "$DIR" -type f -name "*.mp4" | wc -l)

echo "Number of MP4 files in '$DIR': $count"
