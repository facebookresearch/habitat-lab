import cv2
import argparse
import numpy as np
import os

def scale_and_pad(video_path, output_path, output_width, output_height, pad_color, interpolation, start_time=None, duration=None, extend_top=0, extend_bottom=0):
    # Check if the input file is an image (e.g., PNG)
    file_ext = os.path.splitext(video_path)[1].lower()
    is_image = file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

    if is_image:
        # Load the image
        frame = cv2.imread(video_path)
        if frame is None:
            print(f"Error opening image file: {video_path}")
            return
        
        original_width, original_height = frame.shape[1], frame.shape[0]
        fps = 1  # Image treated as a one-frame video
        total_frames = 1
    else:
        # Open the input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return

        # Get original video details
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps

        # Convert start time and duration to frames for videos
        if start_time:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        if duration:
            end_time = min(start_time + duration, video_duration) if start_time else duration
            end_frame = int(end_time * fps)
        else:
            end_frame = total_frames

    # Convert the pad_color string into a tuple of integers
    pad_color = tuple(map(int, pad_color.split(',')))

    scale_w = output_width / original_width
    scale_h = output_height / (original_height + extend_top + extend_bottom)  # Include the extension in height calculation
    scale = min(scale_w, scale_h)  # Use the smaller scaling factor to maintain aspect ratio

    new_width = int(original_width * scale)
    new_height = int((original_height + extend_top + extend_bottom) * scale)

    # Compute the padding needed after scaling
    pad_left = (output_width - new_width) // 2
    pad_right = output_width - new_width - pad_left
    pad_top = (output_height - new_height) // 2
    pad_bottom = output_height - new_height - pad_top

    # Define interpolation method
    interpolation_methods = {
        'nearest': cv2.INTER_NEAREST,
        'cubic': cv2.INTER_CUBIC
    }
    interpolation_method = interpolation_methods.get(interpolation, cv2.INTER_LINEAR)

    # Create VideoWriter object to write the output video (using mp4v codec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    if not out.isOpened():
        print(f"Error opening VideoWriter for output file: {output_path}")
        if not is_image:
            cap.release()
        return

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    while cap.isOpened() and current_frame < end_frame:
        if is_image:
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=interpolation_method)
        else:
            ret, frame = cap.read()
            if not ret:
                break

            # Extend the top and bottom of the current frame using pad_color
            if extend_top > 0 or extend_bottom > 0:
                extended_frame = np.full((original_height + extend_top + extend_bottom, original_width, 3), pad_color, dtype=frame.dtype)
                extended_frame[extend_top:extend_top + original_height, :, :] = frame
                frame = extended_frame

            # Resize the frame with the selected interpolation method
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=interpolation_method)

        # Create a padded frame with the exact output size
        padded_frame = cv2.copyMakeBorder(
            resized_frame,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=pad_color
        )

        # Write the padded frame to the output video
        out.write(padded_frame)
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if is_image:
            break  # Exit after one frame for images

    # Release resources
    if not is_image:
        cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video has been processed and saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Scale and pad a video or image to a specified resolution.")
    parser.add_argument('video_path', type=str, help="Path to the input video or image file.")
    parser.add_argument('output_path', type=str, help="Path to save the output video.")
    parser.add_argument('--output_width', type=int, default=1920, help="Width of the output video (default: 1920)")
    parser.add_argument('--output_height', type=int, default=1080, help="Height of the output video (default: 1080)")
    parser.add_argument('--pad_color', type=str, default="0,0,0", help="Padding color in BGR format (default: black)")
    parser.add_argument('--upscaling_interpolation', type=str, choices=['nearest', 'cubic'], default='cubic', help="Interpolation method for upscaling (default: cubic)")
    parser.add_argument('--start_time', type=float, help="Start time of the video in seconds (optional)")
    parser.add_argument('--duration', type=float, help="Duration of the video to process in seconds (optional)")
    parser.add_argument('--extend_top', type=int, default=0, help="Extra padding to add to the top before processing (default: 0)")
    parser.add_argument('--extend_bottom', type=int, default=0, help="Extra padding to add to the bottom before processing (default: 0)")

    args = parser.parse_args()

    scale_and_pad(args.video_path, args.output_path, args.output_width, args.output_height, args.pad_color, args.upscaling_interpolation, args.start_time, args.duration, args.extend_top, args.extend_bottom)

if __name__ == "__main__":
    main()
