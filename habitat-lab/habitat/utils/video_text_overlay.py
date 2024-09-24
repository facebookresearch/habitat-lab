import cv2
import json
import numpy as np
import argparse

# Global constants for padding and ramp effect
X_PADDING = 40  # Padding from the left or right edge of the frame
Y_PADDING = 40  # Padding from the top edge of the frame
LETTER_PAUSE_FRAMES = 1  # Number of frames to duplicate between each letter
RAMP_FRAMES = 10  # Number of frames for the ramp in/out effect

def wrap_text(text, box_width, font, font_scale, font_thickness):
    """
    Wrap text to fit within the text box width and handle newlines.
    
    Args:
    - text: The text to wrap, including newline characters.
    - box_width: The width of the text box (half the frame width).
    - font: The font type for the text.
    - font_scale: The scale of the font.
    - font_thickness: The thickness of the font.
    
    Returns:
    - A list of strings, each representing a line of wrapped text.
    """
    lines = text.split('\n')  # Split by newlines
    wrapped_lines = []
    for line in lines:
        words = line.split(' ')
        current_line = words[0]

        for word in words[1:]:
            # Measure the current line with the next word added
            line_size = cv2.getTextSize(current_line + ' ' + word, font, font_scale, font_thickness)[0]
            if line_size[0] <= box_width - 2 * X_PADDING:
                current_line += ' ' + word
            else:
                wrapped_lines.append(current_line)
                current_line = word

        wrapped_lines.append(current_line)  # Append the last line of the current paragraph

    return wrapped_lines

def ramp_darkness(frame, current_frame, total_frames, ramp_in=True):
    """
    Gradually darken or lighten the frame for a ramp-in or ramp-out effect.
    
    Args:
    - frame: The original frame to be darkened.
    - current_frame: The current frame number in the ramp.
    - total_frames: The total number of frames for the ramp.
    - ramp_in: If True, darkens the frame (ramp in), otherwise lightens (ramp out).
    
    Returns:
    - The darkened or lightened frame.
    """
    # Calculate the darkening factor (from 1.0 to 0.5 for ramp in, or 0.5 to 1.0 for ramp out)
    if ramp_in:
        factor = 1.0 - (current_frame / total_frames) * 0.5
    else:
        factor = 0.5 + (current_frame / total_frames) * 0.5
    
    return (frame * factor).astype(np.uint8)


def process_video_frames(cap, config, out):
    overlay_idx = 0
    current_frame = 0
    total_overlays = len(config['overlays'])
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Flag to track if the copying message has been printed
    copying_frames_printed = False

    # Loop through video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Function to add frame number in the bottom right corner
        def add_frame_number(frame, frame_number):
            text = f"{frame_number}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            text_color = (255, 255, 255)

            # Get the size of the text
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_width, text_height = text_size

            # Calculate the bottom-right position
            x = frame_width - text_width - 10  # 10 px padding from the right edge
            y = frame_height - 10  # 10 px padding from the bottom edge

            # Add the text to the frame
            cv2.putText(frame, text, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            return frame

        # Check if we need to add an overlay on the current frame
        if overlay_idx < total_overlays and current_frame == config['overlays'][overlay_idx]['frame']:
            # Reset copying message flag
            copying_frames_printed = False

            overlay_info = config['overlays'][overlay_idx]
            text = overlay_info['text']  # This may contain \n
            pause_frames = overlay_info['pause_frames']
            side = overlay_info['side']  # "left" or "right"

            print(f"Adding overlay {overlay_idx + 1}/{total_overlays}...")

            # Define the font and text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.3
            font_thickness = 2
            text_color = (255, 255, 255)
            line_height = cv2.getTextSize("Test", font, font_scale, font_thickness)[0][1] + 10  # Get height of a single line

            # Determine the text box's start x position based on the side
            box_width = frame_width // 2  # Half of the frame width
            if side == 'left':
                start_x = X_PADDING
            elif side == 'right':
                start_x = box_width + X_PADDING  # Start from the right half

            # Wrap the text and handle \n and box width
            wrapped_lines = wrap_text(text, box_width, font, font_scale, font_thickness)

            typed_lines = []  # Keep track of fully typed lines
            total_typing_frames = len(''.join(wrapped_lines)) * LETTER_PAUSE_FRAMES
            ramp_in_frames_count = min(RAMP_FRAMES, total_typing_frames)

            # Type out each line, one by one, with ramp-in darkening effect
            typed_char_count = 0
            for line in wrapped_lines:
                typed_line = ''

                # Type out the line letter by letter
                for letter in line:
                    typed_line += letter
                    typed_char_count += 1

                    # Prepare the ramp-in darkening effect (based on character count)
                    if typed_char_count <= ramp_in_frames_count:
                        ramp_frame_idx = typed_char_count
                        temp_frame = ramp_darkness(frame, ramp_frame_idx, ramp_in_frames_count, ramp_in=True)
                    else:
                        temp_frame = (frame * 0.5).astype(np.uint8)  # Darken the frame normally after ramp-in

                    # Render all previously typed lines
                    for i, typed_text in enumerate(typed_lines):
                        y_position = Y_PADDING + (i + 1) * line_height
                        cv2.putText(temp_frame, typed_text, (start_x, y_position), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

                    # Render the current line as it's being typed
                    y_position = Y_PADDING + (len(typed_lines) + 1) * line_height
                    cv2.putText(temp_frame, typed_line, (start_x, y_position), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

                    # Add the frame number to the bottom right
                    temp_frame = add_frame_number(temp_frame, current_frame)

                    # Write the frame multiple times to simulate typing delay
                    for _ in range(LETTER_PAUSE_FRAMES):
                        out.write(temp_frame)  # Write the frame with the current typed text

                # Once the line is fully typed, add it to the list of typed lines
                typed_lines.append(typed_line)

            # After all lines are typed, pause for the specified number of frames
            for pause_frame in range(pause_frames - RAMP_FRAMES):
                darkened_frame = (frame * 0.5).astype(np.uint8)  # Darken the frame

                # Render all the fully typed lines
                for i, line in enumerate(typed_lines):
                    y_position = Y_PADDING + (i + 1) * line_height
                    cv2.putText(darkened_frame, line, (start_x, y_position), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

                # Add the frame number to the bottom right
                darkened_frame = add_frame_number(darkened_frame, current_frame)

                out.write(darkened_frame)  # Write the paused frame

            # Ramp out darkening at the end
            for ramp_frame in range(RAMP_FRAMES):
                ramped_frame = ramp_darkness(frame, ramp_frame, RAMP_FRAMES, ramp_in=False)
                
                # Add the frame number to the bottom right
                ramped_frame = add_frame_number(ramped_frame, current_frame)
                
                out.write(ramped_frame)  # Write each ramp-out frame

            overlay_idx += 1
        else:
            # Only print the copying message once per block of frames
            if not copying_frames_printed and overlay_idx < total_overlays:
                next_overlay_frame = config['overlays'][overlay_idx]['frame']
                copied_frames = next_overlay_frame - current_frame
                print(f"Copying {copied_frames} frames...")
                copying_frames_printed = True

            # Add the frame number to the bottom right
            frame = add_frame_number(frame, current_frame)

            # Write the normal frame
            out.write(frame)

        current_frame += 1




def add_text_overlays_to_video(input_video_path, json_config_path, output_video_path):
    # Load video and JSON configuration
    cap = cv2.VideoCapture(input_video_path)
    with open(json_config_path, 'r') as f:
        config = json.load(f)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Process the video frames
    process_video_frames(cap, config, out)

    cap.release()
    out.release()
    print(f"Output video saved to {output_video_path}")

import subprocess


def play_video(output_video_path):
    try:
        # Play the video using totem
        subprocess.run(['totem', output_video_path])
    except FileNotFoundError:
        print("Totem is not installed. Please install totem.")


def main():
    parser = argparse.ArgumentParser(description="Add text overlays to video and pause the frames with darkened background.")
    
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("config_json", help="Path to the JSON configuration file for text overlays")
    parser.add_argument("output_video", help="Path to save the output video file")
    
    args = parser.parse_args()

    # Call the function to add text overlays to the video
    add_text_overlays_to_video(args.input_video, args.config_json, args.output_video)

    play_video(args.output_video)



if __name__ == "__main__":
    main()
