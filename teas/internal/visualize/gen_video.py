import os
import shlex
import shutil
import subprocess

import cv2
import numpy as np
import tqdm


def combine_video(images, text, output_dir):
    frames_dir = '{}/frames'.format(output_dir)

    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    for i, f in enumerate(images):
        cv2.imwrite('{}/frames/f_{:0>5}.png'.format(output_dir, i), f)

    subprocess.check_call(
        shlex.split(
            'ffmpeg -framerate 10 -i {}/f_%05d.png -r 30 -pix_fmt yuv420p'
            ' -threads 0 -q:v 3 {}/{}.mp4'.format(
                frames_dir, output_dir,
                text.replace(" ", "_").replace("\n", "_"))
        )
    )

    shutil.rmtree(frames_dir, ignore_errors=True)


def write_text(img, text):
    font = cv2.FONT_HERSHEY_COMPLEX
    bottom_left = (10, 500)
    font_scale = 1.0
    font_color = (255, 255, 255)
    lineType = 2
    x, y = bottom_left
    for line in reversed(text.split('\n')):
        cv2.putText(img, line, (x, y), font, font_scale, font_color, lineType)
        _, h = cv2.getTextSize(line, font, font_scale, lineType)[0]

        y -= int(1.3 * h)


def make_video(text, rgb_frames, depth, labels,
               output_dir="data/videos/test"):
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
                    i * h // scaled.shape[0], j * w // scaled.shape[1]]

        big_frame[:, 0: size] = rgb  # sim.img
        big_frame[:, size: 2 * size] = np.dstack([d] * 3)
        big_frame[:, 2 * size: 3 * size] = lut[scaled]
        write_text(big_frame, text)
        frames.append(big_frame.copy())

    combine_video(frames, text, output_dir)
