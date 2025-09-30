import cv2
import numpy as np
import torch
import shutil
import os
import math
import copy
import habitat.utils.geometry_utils as geo_utils
import imageio
import zipfile
import quaternion
import matplotlib
matplotlib.use('Agg')

import torch
import torchvision.transforms as T
import torchvision.models as models

from scipy import stats
from habitat.utils.geometry_utils import quaternion_from_coeff, quaternion_to_list
from habitat.utils.visualizations import maps
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis

from vint_based import load_distance_model
from vip import load_vip
from vint_train.models.vint.vint import ViNT


def zip_directory(src_dir: str, dst_zip: str) -> None:
    """Zip entire directory with *no* compression (ZIP_STORED)."""
    with zipfile.ZipFile(dst_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(src_dir):
            for f in files:
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, start=src_dir)
                zf.write(abs_path, arcname=rel_path)


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def quaternion_to_yaw(quat):
    """
    Extract yaw from a quaternion, assuming (x, y, z, w) order.
    """
    x, y, z, w = geo_utils.quaternion_to_list(quat)
    
    # Using standard formula: 
    #   yaw = atan2(2*(w*y + x*z), 1 - 2*(y^2 + z^2))
    sin_yaw = 2.0 * (w * y + x * z)
    cos_yaw = 1.0 - 2.0 * (y**2 + z**2)
    yaw = np.arctan2(sin_yaw, cos_yaw)
    return yaw


def angle_difference(a, b):
    """
    Returns the angle difference (a - b) in the range (-pi, pi].
    """
    diff = (a - b + np.pi) % (2.0 * np.pi) - np.pi
    return diff


def get_desired_yaw(goal_pos, current_pos):
    dx = goal_pos[0] - current_pos[0]
    dy = goal_pos[1] - current_pos[1]

    return np.arctan2(dy, dx)


def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )


def average_min_distance(goal_feat: torch.Tensor, current_feat: torch.Tensor, k: int = 1) -> torch.Tensor:
    """
    Computes the average minimum distance between each feature in current_feat and the closest feature in goal_feat.

    Args:
        goal_feat (torch.Tensor): Tensor of shape (BS, NR_FEATURES, NR_CHANNELS)
        current_feat (torch.Tensor): Tensor of shape (BS, NR_FEATURES, NR_CHANNELS)
        k (int): Number of closest matches to consider (default: 1, meaning min distance)

    Returns:
        torch.Tensor: The average of the minimum distances for each feature.
    """
    # Compute pairwise Euclidean distances between current_feat and goal_feat
    diff = current_feat.unsqueeze(2) - goal_feat.unsqueeze(1)  # Shape: (BS, NR_FEATURES, NR_FEATURES, NR_CHANNELS)
    dist = torch.norm(diff, dim=-1)  # Shape: (BS, NR_FEATURES, NR_FEATURES)
    
    # Find the k closest goal features for each current feature
    min_distances, _ = torch.topk(dist, k, dim=-1, largest=False)
    
    # Compute the mean over the selected min distances
    avg_min_dist = min_distances.mean()
    
    return avg_min_dist


def rotate_agent_state(agent_state, angle_degrees, axis=np.array([0.0, 1.0, 0.0])):
    """
    Rotate the agent's orientation by `angle_rad` around `axis` (world up by default).
    Returns a shallow copy with updated rotation.
    """
    # build an incremental rotation Δq from angle-axis
    dq = quat_from_angle_axis(np.radians(angle_degrees), axis)  # unit quaternion
    # compose with current orientation (left-multiply applies Δq in world frame)
    q_new = dq * agent_state.rotation

    rotated = type(agent_state)()
    rotated.position = agent_state.position.copy()  # keep position unless you also want to orbit
    rotated.rotation = q_new
    return rotated


def get_default_transform(normalize=True):
    transforms = [T.Resize(256), T.CenterCrop(224), T.ToTensor()]
    if normalize:
        transforms.append(T.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225]))
    return T.Compose(transforms)


def load_embedding(rep: str):
    """
    Dynamically load a model and its transform based on the representation name.
    Model IDs must match rep exactly for decoders (e.g. 'one_scene_decoder_20').
    """
    # VIP-based models
    if rep.startswith('vip-'):
        model = load_vip(modelid=rep)
        transform = get_default_transform()
    # VIP original model
    elif rep == 'vip':
        model = load_vip(modelid='resnet50')
        transform = get_default_transform(normalize=False)
    # ViNT original model
    elif rep == "vint":
        model = ViNT(obs_encoder='efficientnet-b0', mha_num_attention_layers=4)
        model.load_state_dict(torch.load("/cluster/home/lmilikic/.dist_models/vint/vint.pt"))
        model = model.to('cuda')
        transform = get_default_transform()
    # Distance decoders, Vint distance, and one-scene quasi MSE
    elif rep.startswith(('one_scene_decoder', 'dist_decoder', 'vint_dist', 'one_scene_quasi', 'quasi', 'dist_vld')):
        model = load_distance_model(modelid=rep)
        transform = get_default_transform()
    # Standard ResNet
    elif rep == 'resnet':
        model = models.resnet50(pretrained=True, progress=False)
        transform = get_default_transform()
    # DINO-V2 ViT-B/14
    elif rep == 'dino':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        transform = get_default_transform()
    else:
        raise ValueError(f"Unsupported representation: '{rep}'")
    return model, transform


def compute_correlation(distances, gt, method: str = 'kendall'):
    if method == 'spearman':
        corr = stats.spearmanr(distances, gt)
        symbol = '$S_\\rho$'
    elif method == 'kendall':
        corr = stats.kendalltau(distances, gt)
        symbol = '$\\tau$'
    else:
        raise ValueError(f"Unsupported correlation: {method}")
    return corr.statistic, corr.pvalue, symbol


def images_to_video(
    images: list,
    output_path: str,
    fps: int = 10,
    quality: float = 5.0,
    verbose: bool = True,
    **kwargs,
):
    """
    Saves a list of RGB images to a video file using imageio/FFMPEG.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        quality=quality,
        **kwargs,
    )
    iterator = images if not verbose else tqdm(images, desc='Writing frames')
    for frame in iterator:
        writer.append_data(frame)
    writer.close()


def animate_episode(
    frames: list,
    maps: list,
    geo_distances: list,
    pred_distances: list,
    goal_frame: np.ndarray,
    out_video: str,
    max_len: int,
    fps: int,
    corr_method: str = 'kendall',
    pause_sec: float = 0.0
):
    """
    Renders an episode sequence (trajectory, top-down map, distances, goal) into a video file.
    """
    # Truncate sequences
    frames = frames[-(max_len+1):]
    maps   = maps[-(max_len+1):]
    geo    = np.array(geo_distances[-(max_len+1):],  dtype=float)
    pred   = np.array(pred_distances[-(max_len+1):], dtype=float)
    baseline = np.linspace(pred[0], 0.0, len(frames))

    # Correlations
    τ_geo_base, _, sym = compute_correlation(geo,  baseline, corr_method)
    τ_pred_base, _, _   = compute_correlation(pred, baseline, corr_method)
    τ_pred_geo,  _, _   = compute_correlation(pred, geo,      corr_method)

    # Calculate pause frames
    pause_frames = int(fps * pause_sec)
    total_steps = len(frames) + pause_frames

    # Prepare rendering
    rendered = []
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    ax_curve, ax_video, ax_map, ax_goal = axes
    max_y = max(geo.max(), pred.max(), baseline.max())

    def render_step(i):
        idx = min(i, len(frames)-1)
        x = np.arange(idx + 1)

        # Distances curves
        ax_curve.clear()
        ax_curve.plot(x, geo[:idx+1], label=f"Geo ({sym}: {τ_geo_base:.2f})", linewidth=2)
        ax_curve.plot(x, pred[:idx+1], '-.',
                      label=f"Pred ({sym}: {τ_pred_base:.2f}, vsGeo: {τ_pred_geo:.2f})", linewidth=2)
        ax_curve.plot(x, baseline[:idx+1], '--', label="Baseline", linewidth=2)
        ax_curve.set(xlim=(0, len(frames)-1), ylim=(0, max_y),
                     xlabel="Frame", ylabel="Distance",
                     title="Geo & Pred vs Baseline")
        ax_curve.legend(loc="upper right", fontsize="x-small")

        # RGB trajectory
        ax_video.clear()
        ax_video.imshow(frames[idx])
        ax_video.axis("off")
        ax_video.set_title("Trajectory")

        # Top-down map
        ax_map.clear()
        ax_map.imshow(maps[idx])
        ax_map.axis("off")
        ax_map.set_title("Top‑down Map")

        # Goal image
        ax_goal.clear()
        ax_goal.imshow(goal_frame)
        ax_goal.axis("off")
        ax_goal.set_title("Goal Image")

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        rendered.append(buf.reshape(height, width, 3))

    # Render all frames
    for i in range(total_steps):
        render_step(i)
    plt.close(fig)

    # Save video
    images_to_video(rendered, out_video, fps=fps)
    print(f"Saved animation to {out_video}")


class ImageDescriber:
    def __init__(self, device=None, prompt_template="the {label} is"):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.prompt_template = prompt_template
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def describe_image(self, image, label, max_length=150):
        prompt = self.prompt_template.format(label=label)
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        
        output = self.model.generate(**inputs, max_length=max_length)
        
        # Full generated tokens including prompt and new text
        generated_text = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Remove prompt fragment from generated_text safely
        prompt = prompt.lower()
        if prompt in generated_text:
            generated_text = generated_text.split(prompt, 1)[-1].strip()
        
        return generated_text

def depth_to_world_points(
    depth: np.ndarray,
    sensor_state,
    hfov_deg: float,
    width: int,
    height: int,
    pixels = None,
):
    """
    Parameters
    ----------
    depth : (H, W) float32 array
        Depth map in metres for the current frame.
    sensor_state : habitat_sim.SensorState
        From ``agent_state.sensor_states['depth_sensor']``.
    hfov_deg : float
        Horizontal field-of-view in **degrees** (e.g. ``depth_sensor.hfov``).
    width, height : int
        Resolution of the sensor (``depth.shape[1]``, ``depth.shape[0]``).
    pixels : (N, 2) int array, optional
        Specific (row, col) pairs you want to back-project.

    Returns
    -------
    world_xyz : (N, 3) float32 array
        3-D points in **Habitat world coordinates**.
    """

    if pixels is None:
        H, W = depth.shape
        pixels = _default_pixel_set(H, W)

    ys, xs = pixels[:, 0], pixels[:, 1]

    ds = depth[ys, xs]                    # (N,)
    # guard against invalid depth (0 means "no hit" in Habitat)
    valid = ds > 0.0
    ys, xs, ds = ys[valid], xs[valid], ds[valid]

    # --- camera intrinsics --------------------------------------------------
    hfov = math.radians(hfov_deg)
    vfov = 2 * math.atan(math.tan(hfov / 2) * (height / width))
    fx = 0.5 * width  / math.tan(hfov / 2)
    fy = 0.5 * height / math.tan(vfov / 2)
    cx = (width  - 1) * 0.5
    cy = (height - 1) * 0.5

    # --- back-project to camera frame --------------------------------------
    x_cam = (xs - cx) * ds / fx
    y_cam = (ys - cy) * ds / fy
    z_cam = -ds
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)  # (N, 3)

    # --- camera → world -----------------------------------------------------
    R = quaternion.as_rotation_matrix(sensor_state.rotation)  # (3, 3)
    t = np.asarray(sensor_state.position)                     # (3,)
    pts_world = pts_cam @ R.T + t                             # (N, 3)

    return pts_world.astype(np.float32)


def _default_pixel_set(h: int, w: int) -> np.ndarray:
    """
    Build the 15-pixel anchor pattern:

        • 7-ray strip in the upper third     (row = h/3)
        • 7-ray strip in the lower third     (row = 2h/3)
        • 1 pixel exactly at the image centre
    """
    Δ = w // 4
    c = w // 2

    cols = np.array([max(c - 2*Δ, 0), c - Δ, c, c + Δ, min(c + 2*Δ, w - 1)],
                    dtype=np.int32)

    rows_upper = np.full_like(cols, h // 3)
    rows_lower = np.full_like(cols, 2 * h // 3)

    strip_upper = np.stack([rows_upper, cols], axis=1)
    strip_lower = np.stack([rows_lower, cols], axis=1)
    centre_pt   = np.array([[h // 2, c]], dtype=np.int32)

    return np.concatenate([strip_upper, strip_lower, centre_pt], axis=0)

def make_mini_plot(pred, conf, size=256):
    # Normalize by the first value (avoid zero division)
    pred = np.array(pred, dtype=np.float32)
    conf = np.array(conf, dtype=np.float32)
    if pred[0] == 0: pred[0] = 1
    pred_norm = pred / pred[0]

    fig = Figure(figsize=(1.3, 1.3), dpi=size//1.3)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(pred_norm, label='VLD', lw=1)
    ax.plot(conf, label='Conf.', lw=1)
    # Remove axis ticks and spines for minimalism
    # ax.set_xticks([])
    # ax.set_yticks([])
    # for spine in ax.spines.values():
    #     spine.set_visible(False)

    # Set tick font sizes
    ax.tick_params(axis='both', labelsize=6)

    # Add a legend
    ax.legend(fontsize=6, loc='upper right', frameon=True)
    fig.tight_layout(pad=0.1)
    canvas.draw()
    # Get image as np array (RGB)
    plot_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plot_img = cv2.resize(plot_img, (size, size))
    return plot_img

# ---------- intrinsics from FOV ----------
def K_from_fov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    hfov = math.radians(hfov_deg)
    vfov = 2.0 * math.atan(math.tan(hfov/2.0) * (height/width))
    fx = 0.5 * width  / math.tan(hfov/2.0)
    fy = 0.5 * height / math.tan(vfov/2.0)
    cx = (width  - 1) * 0.5
    cy = (height - 1) * 0.5
    return np.array([[fx, 0,  cx],
                     [0,  fy, cy],
                     [0,   0,  1]], dtype=np.float32)

# ---------- build world_T_cam with +Z forward ----------
def se3_world_T_cam_plusZ(position, rotation_quat) -> np.ndarray:
    """Return 4x4 world_T_cam where the *camera* frame is +Z forward (OpenCV).
    Habitat's native camera is -Z forward; we fix that with a constant flip."""
    R_wc_hab = quaternion.as_rotation_matrix(rotation_quat)  # world <- cam(Habitat basis)
    F = np.diag([1, 1, -1])                                  # cam(+Z) -> cam(Habitat)
    # world <- cam(+Z)
    R_wc = R_wc_hab @ F
    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = R_wc
    T[:3, 3] = np.asarray(position, dtype=np.float32)
    return T

# ---------- symmetric log encoding ----------
def sym_log(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.sign(x) * np.log1p(alpha * np.abs(x))

# ---------- relative transform (goal -> current, all in +Z camera basis) ----------
def relative_cam_T(goal_world_T_cam: np.ndarray, curr_world_T_cam: np.ndarray):
    curr_cam_T_world = np.linalg.inv(curr_world_T_cam)
    curr_cam_T_goal_cam = curr_cam_T_world @ goal_world_T_cam
    R_rel = curr_cam_T_goal_cam[:3,:3]        # (3,3)
    t_rel = curr_cam_T_goal_cam[:3, 3]        # (3,)
    return R_rel, t_rel

def projection_success_ratio(
    depth_goal: np.ndarray, K_goal: np.ndarray, world_T_cam_goal: np.ndarray,
    depth_curr: np.ndarray, K_curr: np.ndarray, world_T_cam_curr: np.ndarray,
    depth_thresh: float = 0.10, sample_every: int = 1
) -> float:
    """All inputs assume +Z-forward camera frames and metric depths in meters."""
    Hg, Wg = depth_goal.shape
    # Habitat uses 0 for invalid; set to -1 for our logic
    dg = depth_goal.copy()
    dg[dg <= 0] = -1.0

    valid = dg > 0
    if sample_every > 1:
        # downsample to speed up if you want
        mask = np.zeros_like(valid)
        mask[::sample_every, ::sample_every] = True
        valid = valid & mask

    total = int(valid.sum())
    if total == 0:
        return 0.0

    vg, ug = np.where(valid)               # (N,)
    zg = dg[vg, ug]                        # (N,)

    pix = np.stack([ug, vg, np.ones_like(ug)], 0)   # (3,N)
    Kg_inv = np.linalg.inv(K_goal)
    pts_cam_goal = Kg_inv @ pix * zg                 # (3,N)
    pts_h_goal = np.vstack([pts_cam_goal, np.ones((1, pts_cam_goal.shape[1]))])

    # goal-cam -> world -> current-cam
    curr_cam_T_goal_cam = np.linalg.inv(world_T_cam_curr) @ world_T_cam_goal
    pts_h_curr = curr_cam_T_goal_cam @ pts_h_goal
    pts_curr = pts_h_curr[:3, :]                       # (3,N)

    Z = pts_curr[2, :]
    in_front = Z > 0                                   # +Z forward

    # project to current image
    pix_curr_h = K_curr @ pts_curr
    u = pix_curr_h[0, :] / (pix_curr_h[2, :] + 1e-8)
    v = pix_curr_h[1, :] / (pix_curr_h[2, :] + 1e-8)
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    Hc, Wc = depth_curr.shape
    in_bounds = (ui >= 0) & (ui < Wc) & (vi >= 0) & (vi < Hc)
    ok = in_front & in_bounds
    if not np.any(ok):
        return 0.0

    Z_curr_img = depth_curr[vi[ok], ui[ok]]
    valid_curr = Z_curr_img > 0
    dep_ok = np.abs(Z[ok] - Z_curr_img) < depth_thresh
    success = valid_curr & dep_ok
    return float(success.sum()) / float(total)

def project_goal_to_current_details(
    depth_goal: np.ndarray, K_goal: np.ndarray, world_T_cam_goal: np.ndarray,
    depth_curr: np.ndarray, K_curr: np.ndarray, world_T_cam_curr: np.ndarray,
    depth_thresh: float = 0.10, sample_every: int = 1,
):
    """
    Same semantics as `projection_success_ratio`:
      • invalid depth == 0 is ignored on goal and current
      • round-to-nearest pixel sampling in the current frame
      • hard pass/fail using absolute depth threshold
      • ratio = (#successes) / (#valid goal pixels)

    Returns:
      u, v           : float arrays of projected pixel coords (length N)
      ok_mask        : bool mask (length N): in-front & in-bounds
      success_mask   : bool mask (length N): ok & current-depth-valid & |Δz| < thresh
      ratio          : float
      Z_proj         : float array (length N): projected depths in current cam (Z>0 if ok)
      Z_img          : float array (length N): sampled current image depths at (u,v),
                       -1 where not ok or invalid.
    """
    # 1) select valid goal pixels
    Hg, Wg = depth_goal.shape
    dg = depth_goal.copy()
    dg[dg <= 0] = -1.0  # Habitat invalid -> -1
    valid = dg > 0

    if sample_every > 1:
        mask = np.zeros_like(valid)
        mask[::sample_every, ::sample_every] = True
        valid &= mask

    total = int(valid.sum())
    if total == 0:
        return (np.array([]), np.array([]),
                np.array([], dtype=bool), np.array([], dtype=bool),
                0.0, np.array([]), np.array([]))

    vg, ug = np.where(valid)           # (N,)
    zg = dg[vg, ug]                    # (N,)
    pix = np.stack([ug, vg, np.ones_like(ug)], axis=0)  # (3,N)

    # 2) unproject into goal cam (+Z forward), then to current cam
    Kg_inv = np.linalg.inv(K_goal)
    pts_cam_goal = Kg_inv @ pix * zg                      # (3,N)
    pts_h_goal = np.vstack([pts_cam_goal, np.ones((1, pts_cam_goal.shape[1]))])
    curr_cam_T_goal_cam = np.linalg.inv(world_T_cam_curr) @ world_T_cam_goal
    pts_h_curr = curr_cam_T_goal_cam @ pts_h_goal
    pts_curr = pts_h_curr[:3, :]                          # (3,N)

    Z_proj = pts_curr[2, :]
    in_front = Z_proj > 0

    # 3) project to current image and round to nearest pixel
    pix_curr_h = K_curr @ pts_curr
    u = pix_curr_h[0, :] / (pix_curr_h[2, :] + 1e-8)
    v = pix_curr_h[1, :] / (pix_curr_h[2, :] + 1e-8)
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    Hc, Wc = depth_curr.shape
    in_bounds = (ui >= 0) & (ui < Wc) & (vi >= 0) & (vi < Hc)
    ok = in_front & in_bounds
    if not np.any(ok):
        return (u, v, ok, np.zeros_like(ok, bool), 0.0, Z_proj, np.full_like(Z_proj, -1.0))

    # 4) sample current depth (nearest) and evaluate hard threshold
    Z_img_all = np.full_like(Z_proj, -1.0, dtype=np.float32)
    Z_img_ok = depth_curr[vi[ok], ui[ok]]
    Z_img_all[ok] = Z_img_ok

    valid_curr = Z_img_ok > 0
    dep_ok = np.abs(Z_proj[ok] - Z_img_ok) < depth_thresh
    success_small = valid_curr & dep_ok

    success_mask = np.zeros_like(ok, dtype=bool)
    success_mask[np.where(ok)[0]] = success_small

    ratio = float(success_small.sum()) / float(total)

    return u, v, ok, success_mask, ratio, Z_proj, Z_img_all

def draw_projection_overlay(
    rgb_curr: np.ndarray,
    u: np.ndarray, v: np.ndarray,
    ok_mask: np.ndarray, success_mask: np.ndarray,
    point_size: int = 2, alpha: float = 0.85, max_points: int = 2000
) -> np.ndarray:
    """Draw green = success, red = fail. Returns RGB image with overlay."""
    H, W, _ = rgb_curr.shape
    overlay = rgb_curr.copy()

    idx_ok = np.where(ok_mask)[0]
    if idx_ok.size == 0:
        return rgb_curr

    # split successes/fails among valid
    succ_idx = idx_ok[success_mask[idx_ok]]
    fail_idx = idx_ok[~success_mask[idx_ok]]

    # subsample so we don’t draw 100k dots
    def _sub(ix):
        if ix.size > max_points:
            sel = np.random.choice(ix.size, max_points, replace=False)
            return ix[sel]
        return ix

    succ_idx = _sub(succ_idx)
    fail_idx = _sub(fail_idx)

    def _draw(ix, color):
        if ix.size == 0: return
        uu = np.clip(np.round(u[ix]).astype(int), 0, W-1)
        vv = np.clip(np.round(v[ix]).astype(int), 0, H-1)
        for x, y in zip(uu, vv):
            cv2.circle(overlay, (x, y), point_size, color, -1, lineType=cv2.LINE_AA)

    _draw(succ_idx, (0,255,0))   # green
    _draw(fail_idx, (0,0,255))   # red

    if alpha >= 1.0:
        return overlay
    return cv2.addWeighted(overlay, alpha, rgb_curr, 1.0 - alpha, 0.0)

def add_projection_legend(
    img: np.ndarray,
    ratio: float = None,
    succ: int = None,
    fail: int = None,
    pos: str = "tr",        # "tl" | "tr" | "bl" | "br"
    box_w: int = 16,
    box_h: int = 16,
    pad: int = 10,
    line_h: int = 20,
    alpha_bg: float = 0.55
) -> np.ndarray:
    """
    Draw a small legend: green=match, red=fail, plus optional ratio/counts.
    Colors are given in RGB (we keep using (0,255,0) and (0,0,255) to stay
    consistent with your overlay, which you later channel-swap for imshow).
    """
    h, w = img.shape[:2]
    lines = [
        ("", None),  # spacer on top
        ("match", (0, 255, 0)),
        ("fail",  (0,   0, 255)),
    ]
    if ratio is not None:
        lines.append((f"ratio: {ratio:.3f}", None))
    if succ is not None and fail is not None:
        lines.append((f"succ: {succ}  fail: {fail}", None))

    # compute legend block size
    max_text_w = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1
    for text, color in lines:
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        max_text_w = max(max_text_w, tw)
    width  = pad*3 + box_w + max_text_w
    height = pad + line_h * (len(lines)) + pad

    # anchor
    if pos == "tr":
        x0, y0 = w - width - pad, pad
    elif pos == "bl":
        x0, y0 = pad, h - height - pad
    elif pos == "br":
        x0, y0 = w - width - pad, h - height - pad
    else:  # "tl"
        x0, y0 = pad, pad

    # translucent bg
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, alpha_bg, img, 1 - alpha_bg, 0)

    # draw lines
    y = y0 + pad + line_h  # skip spacer line
    for text, color in lines[1:]:
        # color chip
        if color is not None:
            cv2.rectangle(img, (x0 + pad, y - box_h + 4),
                          (x0 + pad + box_w, y + 4), color, -1)
            tx = x0 + pad*2 + box_w
        else:
            tx = x0 + pad
        cv2.putText(img, text, (tx, y),
                    font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        y += line_h

    return img
