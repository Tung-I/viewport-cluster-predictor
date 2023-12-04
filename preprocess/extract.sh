#!/bin/bash

# Define the base paths for videos and frames
VIDEO_BASE_PATH="/home/tungi/datasets/ViewportData/Out/Videos"
FRAME_BASE_PATH="/home/tungi/datasets/ViewportData/Out/Frames"

# Loop from 1 to 27
for i in $(seq -w 1 27); do
    VIDEO_PATH="${VIDEO_BASE_PATH}/Video${i}.mp4"
    FRAME_OUTPUT_DIR="${FRAME_BASE_PATH}/Video${i}"

    # Run the Python script
    python preprocess/extract_frames.py "$VIDEO_PATH" "$FRAME_OUTPUT_DIR"
done




