import cv2
import argparse

def censor_video(source_video_path, output_video_path, rect_coords, rect_color):
    # Open the video file
    cap = cv2.VideoCapture(source_video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Rectangle coordinates (x, y, width, height)
    x, y, w, h = rect_coords
    color = rect_color  # (B, G, R)

    # Process each frame
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw the rectangle on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=-1)  # Filled rectangle

        # Write the frame to the output video
        out.write(frame)

        frame_num += 1
        print(f'Processing frame {frame_num}/{total_frames}', end='\r')

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Censor a region in a video by overlaying a colored rectangle on every frame.")
    
    parser.add_argument('source_video', type=str, help="Path to the input video.")
    parser.add_argument('output_video', type=str, help="Path to the output censored video.")
    
    parser.add_argument('--rect', type=int, nargs=4, metavar=('x', 'y', 'width', 'height'),
                        required=True, help="Rectangle coordinates and size (x, y, width, height).")
    
    parser.add_argument('--color', type=int, nargs=3, metavar=('R', 'G', 'B'),
                        required=True, help="Rectangle color in RGB format (R, G, B).")

    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments from the CLI
    args = parse_arguments()

    # Convert RGB to BGR for OpenCV
    rect_color_bgr = (args.color[2], args.color[1], args.color[0])

    # Censor the video using the provided arguments
    censor_video(args.source_video, args.output_video, args.rect, rect_color_bgr)
