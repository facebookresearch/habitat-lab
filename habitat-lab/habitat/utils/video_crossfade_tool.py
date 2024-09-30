import cv2
import numpy as np
import sys

# Constants
CROSS_FADE_DURATION = 30  # Crossfade duration in frames
PROGRESS_INTERVAL = 100  # Number of frames between progress updates

def blend_frames(frame1, frame2, alpha):
    """Blend two frames with the given alpha."""
    return cv2.addWeighted(frame1, alpha, frame2, 1 - alpha, 0)

def process_videos(video_path1, video_path2, keyframes, output_path):
    # Open the video files
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Unable to open one or both video files.")
        return

    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create VideoWriter to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Iterate over frames and apply cross-fades at keyframes
    keyframe_idx = 0
    keyframe = keyframes[keyframe_idx]
    frame_count = 0
    cross_fading = False
    fade_in = False

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # Print progress every `PROGRESS_INTERVAL` frames
        if frame_count % PROGRESS_INTERVAL == 0:
            percent_complete = (frame_count / total_frames) * 100
            print(f"Processing frame {frame_count}/{total_frames} ({percent_complete:.2f}% complete)")

        if frame_count >= keyframe:
            # Start cross-fade
            cross_fading = True
            fade_in = not fade_in  # Alternate between fade in and fade out
            keyframe_idx += 1
            if keyframe_idx < len(keyframes):
                keyframe = keyframes[keyframe_idx]
            else:
                keyframe = total_frames + 1  # No more keyframes
        
        if cross_fading:
            # Calculate alpha based on the frame position within the crossfade duration
            fade_position = frame_count % CROSS_FADE_DURATION
            alpha = fade_position / CROSS_FADE_DURATION

            if fade_in:
                # Fade into video 2
                blended_frame = blend_frames(frame1, frame2, alpha)
            else:
                # Fade back into video 1
                blended_frame = blend_frames(frame2, frame1, alpha)
            
            if fade_position == CROSS_FADE_DURATION - 1:
                cross_fading = False  # End cross-fading after duration

        else:
            blended_frame = frame1 if fade_in else frame2

        out.write(blended_frame)
        frame_count += 1

    # Release everything when the job is finished
    cap1.release()
    cap2.release()
    out.release()
    print(f"Video processing complete. Output saved to {output_path}.")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python crossfade_tool_cv2.py <video1> <video2> <keyframes> <output>")
        sys.exit(1)

    video_path1 = sys.argv[1]
    video_path2 = sys.argv[2]
    keyframes = list(map(int, sys.argv[3].split(",")))
    output_path = sys.argv[4]

    process_videos(video_path1, video_path2, keyframes, output_path)
