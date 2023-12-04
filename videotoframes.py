# Take a video file path (mp4 file) and a output directory path as input and extract frames from the video as jpeg images and save them in the output directory
# Usage: python3 videotoframes.py <video_file_path> <output_directory_path>
# Example: python3 videotoframes.py /home/user/Downloads/video.mp4 /home/user/Downloads/frames
# Dependencies: opencv-python

import cv2
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Extract frames from a video file')
    parser.add_argument('video_file_path', help='path to the video file')
    parser.add_argument('output_directory_path', help='path to the output directory')
    args = parser.parse_args()

    video_file_path = args.video_file_path
    output_directory_path = args.output_directory_path

    if not os.path.exists(video_file_path):
        print('Video file path does not exist')
        sys.exit(1)

    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path)

    video = cv2.VideoCapture(video_file_path)
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_count += 1
        frame_path = os.path.join(output_directory_path, 'frame' + str(frame_count) + '.jpg')
        cv2.imwrite(frame_path, frame)
    video.release()
    print('Frames extracted from the video and saved in the output directory')
