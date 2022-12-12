from PIL import Image
import numpy as np
import io
import imageio
import h5py
from tqdm import tqdm
from pygifsicle import optimize


def img_from_bytes(data: bytes, height=None, width=None) -> np.ndarray:
    """Convert image from png bytes"""
    image = Image.open(io.BytesIO(data), mode="r", formats=["png"])
    # TODO: decide if default image format should switch over to webp
    # Issue: not quite as good at handling depth
    # image = Image.open(data, mode='r', formats=['webp'])
    if height and width:
        image = image.resize([width, height])
    return np.asarray(image)


def pil_to_bytes(img: Image) -> bytes:
    """Convert image to bytes using PIL"""
    data = io.BytesIO()
    img.save(data, format="png")
    return data.getvalue()


def img_to_bytes(img: np.ndarray) -> bytes:
    # return bytes(Image.fromarray(data)).tobytes()
    img = Image.fromarray(img)
    return pil_to_bytes(img)


def torch_to_bytes(img: np.ndarray) -> bytes:
    """convert from channels-first image (torch) to bytes)"""
    assert len(img.shape) == 3
    img = np.rollaxis(img, 0, 3)
    return img_to_bytes(img)


def png_to_gif(
    group: h5py.Group, key: str, name: str, save=True, height=None, width=None
):
    """
    Write key out as a gif
    """
    gif = []
    print("Writing gif to file:", name)
    img_stream = group[key]
    # for i,aimg in enumerate(tqdm(group[key], ncols=50)):
    for ki, k in tqdm(
        sorted([(int(j), j) for j in img_stream.keys()], key=lambda pair: pair[0]),
        ncols=50,
    ):
        bindata = img_stream[k][()]
        img = img_from_bytes(bindata, height, width)
        gif.append(img)
    if save:
        imageio.mimsave(name, gif)
    else:
        return gif


def pngs_to_gifs(filename: str, key: str):
    h5 = h5py.File(filename, "r")
    for group_name, group in h5.items():
        png_to_gif(group, key, group_name + ".gif")


def schema_to_gifs(filename: str):
    keys = [
        "top_rgb",
        "right_rgb",
        "left_rgb",
        "wrist_rgb",
    ]
    h5 = h5py.File(filename, "r")
    x = 1
    for group_name, grp in h5.items():
        print(f"Processing {group_name}, {x}/{len(h5.keys())}")
        x += 1
        gifs = []
        gif_name = group_name + ".gif"
        for key in keys:
            if key in grp.keys():
                gifs.append(
                    png_to_gif(grp, key, name="", height=120, width=155, save=False)
                )
        # TODO logic for concatenating the gifs and saving with group's name
        concatenated_gif = None
        for gif in gifs:
            if gif:
                if concatenated_gif is not None:
                    concatenated_gif = np.hstack((concatenated_gif, gif))
                else:
                    concatenated_gif = gif
        imageio.mimsave(gif_name, concatenated_gif)
        optimize(gif_name)


def png_to_mp4(group: h5py.Group, key: str, name: str, fps=10):
    """
    Write key out as a mpt
    """
    gif = []
    print("Writing gif to file:", name)
    img_stream = group[key]
    writer = None

    # for i,aimg in enumerate(tqdm(group[key], ncols=50)):
    for ki, k in tqdm(
        sorted([(int(j), j) for j in img_stream.keys()], key=lambda pair: pair[0]),
        ncols=50,
    ):

        bindata = img_stream[k][()]
        _img = img_from_bytes(bindata)
        w, h = _img.shape[:2]
        img = np.zeros_like(_img)
        img[:, :, 0] = _img[:, :, 2]
        img[:, :, 1] = _img[:, :, 1]
        img[:, :, 2] = _img[:, :, 0]

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
            writer = cv2.VideoWriter(name, fourcc, fps, (h, w))
        writer.write(img)
    writer.release()


def pngs_to_mp4(filename: str, key: str, vid_name: str, fps: int):
    h5 = h5py.File(filename, "r")
    for group_name, group in h5.items():
        png_to_mp4(group, key, str(vid_name) + "_" + group_name + ".mp4", fps=fps)
