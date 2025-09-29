"""
manual_bev_record.py

Start a TopDownMetaDrive environment with manual keyboard control and record the
BEV second channel (road_lines) per frame as raw numpy arrays (not images).

Usage:
    python envs/manual_bev_record.py

Controls:
- Use the in-game keyboard controls provided by MetaDrive (set `manual_control=True` and `controller='keyboard'`).
- Exit by closing the env window or with Ctrl+C in the terminal. The script will save all collected frames on exit.

Output:
- A NumPy file saved to `recordings/bev_channel2.npy` containing an array of shape
  (T, H, W) where T is number of recorded frames.

Post-processing (example included below) shows how to load that .npy and make
an animation (matplotlib or imageio).

Notes:
- `use_render` must be True to allow keyboard control in most setups. If you
  run headless, you can still set `manual_control=False` and drive programmatically.
- The script attempts to call env.step(None) for manual control; if the env
  requires an action it will fall back to sampling.
"""

import os
import time
import numpy as np

from metadrive.envs.top_down_env import TopDownMetaDrive


def get_bev_hwc(obs):
    """Return HWC (H,W,C) float array with values in [0,1].
    Accepts either the raw observation array or a dict with key "image".
    """
    x = obs["image"] if isinstance(obs, dict) and "image" in obs else obs
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"Expect 3D BEV, got {x.shape}")
    # If channels-first (C,H,W) -> transpose to HWC
    if x.shape[0] in (2, 3, 4, 5) and x.shape[-1] not in (2, 3, 4, 5):
        x = np.transpose(x, (1, 2, 0))
    if x.max() > 1.0:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def make_output_dir(base="recordings"):
    os.makedirs(base, exist_ok=True)
    # timestamped filename
    ts = time.strftime("%Y%m%d_%H%M%S")
    return base, os.path.join(base, f"bev_channel2_{ts}.npy")


def main():
    cfg = {
        "map": "OO",
        "random_lane_width": True,
        "random_lane_num": False,
        # For manual control you usually want rendering on
        "use_render": True,
        "traffic_density": 0.0,
        "traffic_mode": "hybrid",
        "manual_control": True,
        "controller": "keyboard",
        "vehicle_config": {
            "show_navi_mark": True,
            "show_line_to_dest": False,
            "show_line_to_navi_mark": True,
        },
        # BEV parameters (TopDownMetaDrive will read these)
        "distance": 15,
        "resolution_size": 150,
    }

    env = TopDownMetaDrive(cfg)

    frames = []  # store HxW arrays (second channel)
    out_dir, out_file = make_output_dir()
    print("Recording BEV channel-2 ->", out_file)

    try:
        obs, info = env.reset()
        print("Env reset done. Use keyboard to control the vehicle. Close window or press Ctrl+C to finish recording.")
        step = 0
        while True:
            # When manual_control is True and controller='keyboard', MetaDrive may
            # accept None to indicate user input. If env.step(None) fails, we
            # fallback to sampling an action (so the loop continues).
            action = None
            try:
                result = env.step(action)
            except Exception:
                # fallback if None is not accepted
                try:
                    action = env.action_space.sample()
                    result = env.step(action)
                except Exception as e:
                    print("env.step failed:", e)
                    break

            # Support both Gym (obs, reward, done, info) and Gymnasium (obs, reward, terminated, truncated, info)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = bool(terminated or truncated)
            elif len(result) == 4:
                obs, reward, done, info = result
            else:
                # unexpected signature
                obs = result[0]
                done = False

            # Convert to HWC float [0,1]
            try:
                bev = get_bev_hwc(obs)
            except Exception as e:
                print("Failed to convert observation to BEV HWC:", e)
                bev = None

            if bev is not None:
                # second channel is index 1 (road_lines)
                if bev.shape[-1] < 2:
                    print("Observation has <2 channels; skipping frame")
                else:
                    ch2 = bev[..., 1]
                    frames.append(ch2.copy())

            step += 1
            if done:
                print("Episode done after steps:", step)
                break

            # small sleep to avoid busy-looping; adjust as needed
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("Recording interrupted by user (Ctrl+C)")
    finally:
        env.close()
        if len(frames) == 0:
            print("No frames recorded; nothing to save.")
            return
        arr = np.stack(frames, axis=0)  # T, H, W
        np.save(out_file, arr)
        print(f"Saved {arr.shape[0]} frames to: {out_file}")
        print("To make an animation from the saved .npy, see the example below.")


# ------------ Post-processing example (not executed automatically) ------------
# Example: turn saved .npy into an MP4 using matplotlib.animation or imageio.
#
# import numpy as np
# import imageio
# data = np.load('recordings/bev_channel2_YYYYMMDD_hhmmss.npy')  # shape (T, H, W), values in [0,1]
# # Convert each frame to grayscale PNG-like arrays (H, W) -> (H, W, 3)
# frames = (data * 255).astype(np.uint8)
# writer = imageio.get_writer('recordings/bev_channel2_movie.mp4', fps=15)
# for f in frames:
#     rgb = np.stack([f, f, f], axis=-1)
#     writer.append_data(rgb)
# writer.close()
# -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()
