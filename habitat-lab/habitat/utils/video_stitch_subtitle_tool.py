import cv2
import json
import numpy as np
import sys
import math

TEXT_WIDTH_FRACTION = 0.84
BOTTOM_MARGIN = 30
INT_INF = 10**9  # or a larger number

global_overlays = []


def add_clip_overlays_to_global_list(clip, output_frame_num):
    overlays = clip.get('overlays', [])
    
    for idx, overlay in enumerate(overlays):
        start_offset = overlay.get('start_frame_offset', 0)  # Default to 0
        global_start_frame = output_frame_num + start_offset

        # Set duration based on the next overlay or clip's end
        if idx < len(overlays) - 1:
            next_overlay_start = output_frame_num + overlays[idx + 1].get('start_frame_offset', 0)
        else:
            next_overlay_start = None  # No subsequent overlay in this clip
        
        # Handle duration and hold_until_next
        if overlay.get('hold_until_next', False):
            assert 'duration' not in overlay
            if next_overlay_start is not None:
                duration = next_overlay_start - global_start_frame
            else:
                duration = INT_INF  # set to inf for now; we'll fix up later when we get another overlay
        else:
            duration = overlay['duration']

        # fix up previous overlay and check for overlap
        if global_overlays:
            last_overlay = global_overlays[-1]
            if last_overlay['duration'] == INT_INF:
                last_overlay['duration'] = global_start_frame - last_overlay['global_start_frame']
                assert last_overlay['duration'] > 0
            else:
                last_overlay_end = last_overlay['global_start_frame'] + last_overlay['duration']
                assert global_start_frame >= last_overlay_end, f"Overlay starting at frame {global_start_frame} overlaps with previous overlay ending at frame {last_overlay_end}"
        
        # Add the overlay to the global list
        global_overlays.append({
            'text': overlay['text'],
            'global_start_frame': global_start_frame,
            'duration': duration
        })

def render_overlays_on_frame(frame, current_frame):
    for overlay in global_overlays:
        overlay_start = overlay['global_start_frame']
        overlay_end = overlay_start + overlay['duration']
        
        # Check if the overlay should be visible at the current frame
        if overlay_start <= current_frame < overlay_end:
            frame = add_overlay_text(frame, overlay['text'], frame.shape[1])
            # only ever render one overlay
            break
    
    return frame



def wrap_text(text, font, font_scale, max_width):
    words = text.split(' ')
    wrapped_lines = []
    current_line = words[0]

    for word in words[1:]:
        # Check the width of the current line with the new word
        (line_width, _), _ = cv2.getTextSize(current_line + ' ' + word, font, font_scale, 2)
        if line_width <= max_width:
            current_line += ' ' + word
        else:
            wrapped_lines.append(current_line)
            current_line = word

    # Add the last line
    wrapped_lines.append(current_line)
    
    return wrapped_lines


def add_overlay_text(frame, overlay_text, video_width):
    # Define the maximum text width
    max_text_width = int(video_width * TEXT_WIDTH_FRACTION)

    # Prepare the text font
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.8
    font_thickness = 3
    font_color = (255, 255, 255)  # White text
    background_color = (0, 0, 0)  # Black background
    overlay_alpha = 0.6  # Transparency for the background

    # Split the text into lines based on max width
    wrapped_text = wrap_text(overlay_text, font, font_scale, max_text_width)

    # Calculate the starting position (bottom of the frame with margin)
    y_offset = frame.shape[0] - BOTTOM_MARGIN
    padding = 10  # Padding for the black background

    # Calculate the total height of the text block
    total_text_height = 0
    for line in wrapped_text:
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        total_text_height += text_height + padding

    # Draw the semi-transparent black bar
    overlay = frame.copy()
    y_top = y_offset - total_text_height - padding
    cv2.rectangle(overlay, (0, y_top), (video_width, frame.shape[0]), background_color, -1)
    frame = cv2.addWeighted(overlay, overlay_alpha, frame, 1 - overlay_alpha, 0)

    # Draw the wrapped text on top of the black bar
    for line in reversed(wrapped_text):
        # Calculate the text size and position
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        x_pos = (frame.shape[1] - text_width) // 2  # Center the text
        y_pos = y_offset
        
        # Put the text on the frame
        cv2.putText(frame, line, (x_pos, y_pos), font, font_scale, font_color, font_thickness)

        # Move to the next line
        y_offset -= text_height + padding  # Move up for the next line
    return frame




def add_frame_number(frame, source_frame_num, output_frame_num, clip_filepath):
    # Make a copy of the frame to prevent overwriting the same frame multiple times
    frame_copy = frame.copy()
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (255, 255, 255)  # White text
    font_thickness = 2
    position_source = (10, frame_copy.shape[0] - 50)  # Adjusted to fit filepath, source frame at bottom
    position_output = (10, frame_copy.shape[0] - 30)  # Output frame number above source frame
    position_filepath = (10, frame_copy.shape[0] - 10)  # Filepath text at the bottom
    
    # Add text for source frame number
    cv2.putText(frame_copy, f"Source: {source_frame_num}", position_source, font, font_scale, font_color, font_thickness)

    # Add text for output frame number
    cv2.putText(frame_copy, f"Output: {output_frame_num}", position_output, font, font_scale, font_color, font_thickness)

    # Add text for the current clip filepath
    cv2.putText(frame_copy, f"Clip: {clip_filepath}", position_filepath, font, font_scale, font_color, font_thickness)

    return frame_copy




def load_json(json_filepath):
    with open(json_filepath, 'r') as f:
        return json.load(f)

def fade_in_from_black(frame, fade_duration, frame_index):
    alpha = min(frame_index / fade_duration, 1.0)
    black_frame = np.zeros_like(frame)
    return cv2.addWeighted(frame, alpha, black_frame, 1 - alpha, 0)

def crossfade_frames(prev_frame, next_frame, fade_duration, frame_index):
    alpha = min(frame_index / fade_duration, 1.0)
    return cv2.addWeighted(next_frame, alpha, prev_frame, 1 - alpha, 0)
 

def process_clip(clip, output_writer, prev_clip=None, show_frame_numbers=False, output_frame_start=0):
    cap = cv2.VideoCapture(clip['source'])
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = clip.get('start_frame', 0)
    duration = clip.get('duration', None)
    if duration is None:
        duration = total_frames - start_frame
    crossfade_in = clip.get('crossfade_in', 0)

    # Add this clip's overlays to the global list with their global start frames
    add_clip_overlays_to_global_list(clip, output_frame_start)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    print(f"Processing clip: {clip['source']} from frame {start_frame} for {duration} frames")

    last_frame = None
    output_frame_num = output_frame_start

    for i in range(duration):
        ret, frame = cap.read()
        if not ret:
            frame = last_frame  # Reuse last frame if the video ends early
        else:
            last_frame = frame

        if output_writer is None:
            continue

        # Crossfade or fade-in
        if prev_clip is None and i < crossfade_in:
            frame = fade_in_from_black(frame, crossfade_in, i)
        elif prev_clip is not None and i < crossfade_in:
            frame = crossfade_frames(prev_clip, frame, crossfade_in, i)

        # Add frame numbers if required
        if show_frame_numbers:
            frame = add_frame_number(frame, start_frame + i, output_frame_num, clip['source'])

        # Render any active overlays from the global list
        # temp
        # frame = render_overlays_on_frame(frame, output_frame_num)

        output_writer.write(frame)
        output_frame_num += 1

    cap.release()
    return last_frame, output_frame_num

class CropWriter:
    def __init__(self, output_filepath, fourcc, fps, source_width, source_height, trim_vals):
        self.source_width = source_width
        self.source_height = source_height

        trim_left = trim_vals[0]
        trim_right = trim_vals[1]
        trim_top = trim_vals[2]
        trim_bottom = trim_vals[3]

        # Validate that trims don't exceed source dimensions
        assert trim_left + trim_right < source_width, "Total horizontal trim exceeds video width"
        assert trim_top + trim_bottom < source_height, "Total vertical trim exceeds video height"

        # Set trims
        self.trim_left = trim_left
        self.trim_right = trim_right
        self.trim_top = trim_top
        self.trim_bottom = trim_bottom

        # Calculate the final dimensions after trimming
        final_width = source_width - (trim_left + trim_right)
        final_height = source_height - (trim_top + trim_bottom)

        # Initialize the actual cv2.VideoWriter with the new dimensions
        self.writer = cv2.VideoWriter(output_filepath, fourcc, fps, (final_width, final_height))

    def write(self, frame):
        # Apply trimming
        frame = frame[self.trim_top:self.source_height - self.trim_bottom,
                      self.trim_left:self.source_width - self.trim_right]
        
        # Write the trimmed frame to the output
        self.writer.write(frame)

    def release(self):
        self.writer.release()



def stitch_clips(clips, output_filepath, fade_out_duration, trim_vals, show_frame_numbers=False):
    prev_clip_frame = None
    output_frame_num = 0  # Track the output frame number

    # Get fps and other metadata from the first clip
    first_clip = clips[0]
    cap = cv2.VideoCapture(first_clip['source'])
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create the CropWriter object, which handles video writing and optional trimming
    crop_writer = CropWriter(output_filepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, source_width, source_height, trim_vals)

    for idx, clip in enumerate(clips):
        prev_clip_frame, output_frame_num = process_clip(
            clip, crop_writer, prev_clip=prev_clip_frame, 
            show_frame_numbers=show_frame_numbers, output_frame_start=output_frame_num)

    # Fade out at the end
    print(f"Applying fade out over {fade_out_duration} frames.")
    for i in range(fade_out_duration):
        faded_frame = fade_in_from_black(prev_clip_frame, fade_out_duration, fade_out_duration - i)

        if show_frame_numbers:
            faded_frame = add_frame_number(faded_frame, "N/A", output_frame_num, "Final Fade")

        faded_frame = render_overlays_on_frame(faded_frame, output_frame_num)

        crop_writer.write(faded_frame)
        output_frame_num += 1

    crop_writer.release()



import subprocess

def play_video(output_video_path):
    try:
        # Play the video using totem
        subprocess.run(['totem', output_video_path])
    except FileNotFoundError:
        print("Totem is not installed. Please install totem.")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Video stitching script")
    parser.add_argument("json_filepath", help="Path to the JSON configuration file")
    parser.add_argument("output_filepath", help="Path to save the output video")
    parser.add_argument("--show-frame-numbers", action="store_true", help="Show frame numbers in the output video")
    args = parser.parse_args()

    # Load JSON config
    config = load_json(args.json_filepath)

    # temp
    # config['clips'] = config['clips'][:3]

    trim_vals = [config.get(name, 0) for name in ["trim_left", "trim_right", "trim_top", "trim_bottom"]]

    # Stitch clips together
    stitch_clips(config['clips'], args.output_filepath, config['fade_out_duration'], trim_vals,
                 show_frame_numbers=args.show_frame_numbers)

    print(f"Video saved to {args.output_filepath}")

    # Play the output video
    play_video(args.output_filepath)



if __name__ == "__main__":
    main()
