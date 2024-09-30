import json
import numpy as np
import argparse
from dataclasses import dataclass
from typing import Tuple

ENABLE_FRAME_NUMBER = False
FRAME_NUMBER_FONT_SCALE = 0.7

import cv2
import re

# Global constants for padding and ramp effect
X_PADDING = 40  # Padding from the left or right edge of the frame
Y_PADDING = 40  # Padding from the top edge of the frame
RAMP_FRAMES = 10  # Number of frames for the ramp in/out effect


class VideoProcessor:

    @dataclass
    class TextLine:
        text: str
        typing_delay: int
        font_scale: float
        color: Tuple[int, int, int]
        position_y: int
        
    def __init__(self):

        self._frame_number = None
        self._side = None
        self._cap = None
        self._out = None
        self._config = None
        self._copying_frames_printed = None
        self._show_frame_numbers = None
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_thickness = 2
        self._frame_number_font_scale = 0.7


    def wrap_text(self, text, box_width, font_scale):
        """
        Wrap text to fit within the text box width and handle newlines.

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
                line_size = cv2.getTextSize(current_line + ' ' + word, self._font, font_scale, self._font_thickness)[0]
                if line_size[0] <= box_width - 2 * X_PADDING:
                    current_line += ' ' + word
                else:
                    wrapped_lines.append(current_line)
                    current_line = word

            wrapped_lines.append(current_line)  # Append the last line of the current paragraph

        return wrapped_lines


    def ramp_darkness_full(self, frame):
        return self.ramp_darkness(frame, 1, 1, ramp_in=True)

    def ramp_darkness(self, frame, current_frame, total_frames, ramp_in=True):
        """
        Gradually darken or lighten one half of the frame for a ramp-in or ramp-out effect.
        
        Returns:
        - The partially darkened or lightened frame.
        """
        # Calculate the darkening factor (from 1.0 to 0.5 for ramp in, or 0.5 to 1.0 for ramp out)
        if ramp_in:
            factor = 1.0 - (current_frame / total_frames) * 0.5
        else:
            factor = 0.5 + (current_frame / total_frames) * 0.5

        # Create a copy of the frame to modify
        darkened_frame = frame.copy()
        
        frame_height, frame_width = frame.shape[:2]
        half_width = frame_width // 2
        
        # Darken only the left or right half of the frame
        if self._side == "left":
            darkened_frame[:, :half_width] = (frame[:, :half_width] * factor).astype(np.uint8)
        elif self._side == "right":
            darkened_frame[:, half_width:] = (frame[:, half_width:] * factor).astype(np.uint8)
        else:
            assert False
        
        return darkened_frame


    def check_add_frame_number(self, frame):
        """
        Add the frame number in the bottom-right corner of the frame.
        
        Returns:
        - The frame with the frame number added.
        """

        if not ENABLE_FRAME_NUMBER:
            return frame

        text = f"{self._frame_number}"  # Just print the frame number
        text_color = (255, 255, 255)

        # Get the size of the text
        text_size = cv2.getTextSize(text, self._font, self._frame_number_font_scale, self._font_thickness)[0]
        text_width, text_height = text_size

        # Calculate the bottom-right position
        x = self._frame_width - text_width - 10  # 10 px padding from the right edge
        y = self._frame_height - 10  # 10 px padding from the bottom edge

        # Add the text to the frame
        cv2.putText(frame, text, (x, y), self._font, self._frame_number_font_scale, text_color, self._font_thickness, cv2.LINE_AA)
        return frame


    def write_frame(self, frame):
        """
        Helper function to write a frame to the output, including adding the frame number.
        """
        frame = self.check_add_frame_number(frame)
        self._out.write(frame)


    def process_video_frames(self, frames_to_skip=0, show_frame_numbers=False):
        self._overlay_idx = 0
        self._total_overlays = len(self._config['overlays'])
        self._show_frame_numbers = show_frame_numbers

        # Skip initial frames if needed
        self._current_frame_idx = self.skip_initial_frames(frames_to_skip)

        # Main frame processing loop
        while self._cap.isOpened():
            ret, self._current_source_frame = self._cap.read()
            if not ret:
                break

            # Process the current frame and overlay text if needed
            if self._overlay_idx < self._total_overlays and self._current_frame_idx == self._config['overlays'][self._overlay_idx]['frame']:
                self.process_overlay_frame()
                self._overlay_idx += 1
                self._copying_frames_printed = False
            else:
                self.handle_copying_frames()

            self._current_frame_idx += 1



    def skip_initial_frames(self, frames_to_skip):
        """
        Skip a specified number of frames in the video.
        Returns:
        - The updated current frame number.
        """
        current_frame = 0
        while frames_to_skip > 0 and self._cap.isOpened():
            ret, _ = self._cap.read()
            if not ret:
                break
            frames_to_skip -= 1
            current_frame += 1
        return current_frame


    def get_current_source_frame(self):
        return self._current_source_frame
    

    def process_overlay_frame(self):
        """
        Add text overlays to the frame with ramp-in and ramp-out effects, handling dynamic tokens.
        """

        overlay_info = self._config['overlays'][self._overlay_idx]

        self._side = overlay_info['side']
        self._font_thickness = 2

        sections = overlay_info['sections']

        # Print the overlay progress (1-based index for display)
        print(f"Adding overlay {self._overlay_idx + 1}/{self._total_overlays}...")

        self.add_fade_in_out_frames(RAMP_FRAMES, fade_in=True)

        typed_lines = []  # Track lines that have been fully typed

        lines = []
        self._next_line_position_y = Y_PADDING
        for section in sections:
            self.add_lines(section, lines)

        self.handle_typing_effect(lines)

        # Pause on the frame after all text is rendered
        pause_frames = section.get('pause_frames', 0)
        self.pause_on_frame(lines, pause_frames)

        self.add_fade_in_out_frames(RAMP_FRAMES, fade_in=False)

    def add_lines(self, section, lines):

        font_scale = section.get('font_scale', 1.3)
        line_height = cv2.getTextSize("Test", cv2.FONT_HERSHEY_SIMPLEX, font_scale, self._font_thickness)[0][1] + 10
        color = (255, 255, 255)  # Default color white
        typing_delay = section.get('typing_delay', 0)

        # todo: handle margins
        wrapped_lines = self.wrap_text(section['text'], self._frame_width // 2, font_scale)
        for (_, wrapped_line) in enumerate(wrapped_lines):
            self._next_line_position_y += line_height
            lines.append(VideoProcessor.TextLine(
                text=wrapped_line, 
                typing_delay=typing_delay,
                font_scale=font_scale,
                color=color,
                position_y=self._next_line_position_y))


    def handle_typing_effect(self, lines):
        for (i, line) in enumerate(lines):
            prev_lines = lines[:-1]
            if line.typing_delay > 0:
                for i in range(len(line.text)):
                    temp_frame = self.ramp_darkness_full(self.get_current_source_frame())
                    self.render_lines_on_frame(temp_frame, prev_lines)
                    self.render_partial_line(temp_frame, line, i)
                    for j in range(line.typing_delay):
                        self.write_frame(temp_frame)


    def render_lines_on_frame(self, frame, lines):
        """
        Render the fully typed lines on the frame with the given font scale and color.
        """

        start_x = X_PADDING if self._side == 'left' else self._frame_width // 2 + X_PADDING
        for i, line in enumerate(lines):
            cv2.putText(frame, line.text, (start_x, line.position_y), self._font, line.font_scale, line.color, self._font_thickness, cv2.LINE_AA)


    def render_partial_line(self, frame, line, letter_idx):
        """
        Render a single line of text as it is being typed.
        """
        start_x = X_PADDING if self._side == 'left' else self._frame_width // 2 + X_PADDING
        y_position = line.position_y
        partial_text = line.text[:letter_idx + 1]
        cv2.putText(frame, partial_text, (start_x, y_position), self._font, line.font_scale, line.color, self._font_thickness, cv2.LINE_AA)


    def add_fade_in_out_frames(self, frame_count, fade_in=True):
        for frame_idx in range(frame_count):
            adjusted_frame = self.ramp_darkness(self.get_current_source_frame(), frame_idx, frame_count, ramp_in=fade_in)
            self.write_frame(adjusted_frame)

    def pause_on_frame(self, lines, pause_frames):
        """
        Pause on the frame for the specified number of pause_frames after typing is done.
        """
        for pause_frame in range(pause_frames):
            temp_frame = self.ramp_darkness_full(self.get_current_source_frame())
            temp_frame = self.ramp_darkness_full(self.get_current_source_frame())
            self.render_lines_on_frame(temp_frame, lines)
            self.write_frame(temp_frame)


    def handle_copying_frames(self):
        """
        Handle copying frames when there is no overlay to apply.
        """
        # Calculate the next overlay frame and print the copying progress once per block
        next_overlay_frame = self._config['overlays'][self._overlay_idx]['frame'] if self._overlay_idx < len(self._config['overlays']) else None
        # todo: rename self._current_frame_idx to self._current_frame_idx_idx
        copied_frames = next_overlay_frame - self._current_frame_idx if next_overlay_frame is not None else None
        
        if copied_frames and not self._copying_frames_printed:
            print(f"Copying {copied_frames} frames...")  # Print this message once per block of copying
            self._copying_frames_printed = True

        self.write_frame(self.get_current_source_frame())


    def add_text_overlays_to_video(self, input_video_path, json_config_path, output_video_path, frames_to_skip=0, show_frame_numbers=False):
        # Load video and JSON configuration
        self._cap = cv2.VideoCapture(input_video_path)
        with open(json_config_path, 'r') as f:
            self._config = json.load(f)

        # Get video properties
        self._frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)

        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._out = cv2.VideoWriter(output_video_path, fourcc, fps, (self._frame_width, self._frame_height))

        # Process the video frames with the new options
        self.process_video_frames(frames_to_skip=frames_to_skip, show_frame_numbers=show_frame_numbers)

        self._cap.release()
        self._out.release()
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
    
    parser.add_argument("--skip-frames", type=int, default=0, help="Number of frames to skip at the start of the video (default: 0)")
    parser.add_argument("--show-frame-numbers", action="store_true", help="Display frame numbers in the output video (default: False)")

    args = parser.parse_args()

    video_processor = VideoProcessor()

    # Call the function to add text overlays to the video, with frames_to_skip and show_frame_numbers
    video_processor.add_text_overlays_to_video(args.input_video, args.config_json, args.output_video, frames_to_skip=args.skip_frames, show_frame_numbers=args.show_frame_numbers)

    play_video(args.output_video)





if __name__ == "__main__":
    main()
