import av
import sys
import os

def extract_frames(mp4_path, output_dir):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    input_container = av.open(mp4_path)
    frame_number = 0

    for frame in input_container.decode(video=0):
        # Convert the frame to PIL image
        img = frame.to_image()

        # Save the image
        img.save(os.path.join(output_dir, f"{frame_number}.jpg"))

        frame_number += 1

    print(f"Extracted {frame_number} frames from {mp4_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_frames.py {mp4_path} {output_dir}")
        sys.exit(1)

    mp4_path = sys.argv[1]
    output_dir = sys.argv[2]

    # if output_dir does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    extract_frames(mp4_path, output_dir)
