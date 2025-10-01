import os
import copy
import numpy as np
import gymnasium as gym
from PIL import Image

from metadrive.envs.top_down_env import TopDownMetaDrive


cfg = {
    "num_scenarios": 1,
    "start_seed": 123,
    "random_lane_width": True,
    "random_lane_num": False,
    "use_render": False,
    "traffic_density": 0.0,
    "traffic_mode": "hybrid",
    "manual_control": False,
    "controller": "keyboard",
    "vehicle_config": {
        "show_navi_mark": True,
        "show_line_to_dest": False,
        "show_line_to_navi_mark": True,
        # obs centering: 'ego_center', 'nav_center', 'start_bottom'
        "obs_center_mode": "start_bottom",
    },
    "distance": 40,
    "resolution_size": 224,
}


def make_env():
    def _init():
        env = TopDownMetaDrive(cfg)

        # If observation is a dict with 'image', we will extract it below. Keep env as-is.
        return env

    return gym.wrappers.TimeLimit(_init(), max_episode_steps=1000) if False else _init()


def save_first_frame(out_path: str, cfg_in: dict):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    env = TopDownMetaDrive(cfg_in)

    # Some envs return (obs, info) on reset
    try:
        res = env.reset()
    except TypeError:
        # try old signature
        res = env.reset(None)

    if isinstance(res, tuple) and len(res) == 2:
        obs, info = res
    else:
        obs = res

    # observation may be a dict with 'image' and 'state'
    if isinstance(obs, dict):
        image = obs.get('image', None)
    else:
        image = obs

    if image is None:
        raise RuntimeError('No image observation found in env.reset()')

    # image expected to be float32 in [0,1] or uint8
    img = np.asarray(image)
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    # If channel-first, transpose to HWC
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[0] != img.shape[2]:
        # convert CHW -> HWC
        img = np.transpose(img, (1, 2, 0))

    # Save PNG
    Image.fromarray(img).save(out_path)
    print(f"Saved agent-view image to: {out_path}")
    print(f"Image shape: {img.shape}, dtype: {img.dtype}, min/max: {img.min()}/{img.max()}")

    # Diagnostics: non-white pixel count and bounding box
    try:
        if img.ndim == 3:
            mask = (img.sum(axis=2) < 255 * 3)
        else:
            mask = (img < 255)
        nonwhite = mask.sum()
        print(f"Non-white pixel count: {int(nonwhite)} / {img.shape[0]*img.shape[1]}")
        if nonwhite > 0:
            ys, xs = np.where(mask)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            print(f"Non-white bbox (xmin,ymin,xmax,ymax): {bbox}")
            # Save a highlight image where non-white pixels are red for visibility
            highlight = np.ones_like(img) * 255
            try:
                highlight[ys, xs] = [255, 0, 0]
            except Exception:
                highlight[ys, xs] = 0
            hl_path = out_path.replace('.png', '_highlight.png')
            Image.fromarray(highlight).save(hl_path)
            print(f"Saved highlight image to: {hl_path}")
        else:
            print("Image appears blank (all white).")
    except Exception as e:
        print(f"Diagnostic failed: {e}")

    env.close()


if __name__ == '__main__':
    modes = [
        "ego_center",
        "nav_center",
        "start_bottom",
    ]
    for mode in modes:
        cfg_copy = copy.deepcopy(cfg)
        # ensure vehicle_config exists
        if "vehicle_config" not in cfg_copy or cfg_copy["vehicle_config"] is None:
            cfg_copy["vehicle_config"] = {}
        cfg_copy["vehicle_config"]["obs_center_mode"] = mode
        out = os.path.join(os.getcwd(), 'debug_bev_out', f'agent_view_{mode}.png')
        try:
            save_first_frame(out, cfg_copy)
        except Exception as e:
            print(f"Failed to save for mode {mode}: {e}")
