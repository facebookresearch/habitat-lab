import os

import cv2
import imageio
import numpy as np
import tqdm


def images_to_video(images, output_dir, video_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    writer = imageio.get_writer(os.path.join(output_dir, video_name), fps=10)
    for im in tqdm.tqdm(images):
        writer.append_data(im)
    writer.close()


def write_text(img, text):
    font = cv2.FONT_HERSHEY_COMPLEX
    bottom_left = (10, 500)
    font_scale = 1.0
    font_color = (255, 255, 255)
    lineType = 2
    x, y = bottom_left
    for line in reversed(text.split("\n")):
        cv2.putText(img, line, (x, y), font, font_scale, font_color, lineType)
        _, h = cv2.getTextSize(line, font, font_scale, lineType)[0]

        y -= int(1.3 * h)


def make_video(
    video_text, rgb_frames, depth, labels, output_dir="data/videos/test"
):
    assert len(rgb_frames) > 0
    size = rgb_frames[0].shape[0]

    big_frame = np.empty((size, 3 * size, 3), dtype=np.uint8)

    frames = []
    episode_len = len(rgb_frames)
    lut = np.random.randint(0, 256, size=(256, 3), dtype=np.uint8)
    for step_id in tqdm.trange(episode_len, leave=False):
        rgb = rgb_frames[step_id]
        d = depth[step_id]
        label = np.array(labels[step_id], dtype=np.uint8)
        h, w = label.shape
        scaled = np.zeros([size, size], dtype=np.uint8)  # output array - 6x6

        # Loop, filling A with tiled values of a at each index
        for i in range(scaled.shape[0]):  # lines in a
            for j in range(scaled.shape[1]):
                scaled[i, j] = label[
                    i * h // scaled.shape[0], j * w // scaled.shape[1]
                ]

        big_frame[:, 0:size] = rgb  # sim.img
        big_frame[:, size : 2 * size] = np.dstack([d] * 3)
        big_frame[:, 2 * size : 3 * size] = lut[scaled]
        write_text(big_frame, video_text)
        frames.append(big_frame.copy())

    images_to_video(frames, output_dir, video_text)
