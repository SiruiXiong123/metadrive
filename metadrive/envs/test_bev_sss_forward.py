import os
import time
import numpy as np
import matplotlib.pyplot as plt

try:
    import imageio
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False

from metadrive.envs.top_down_env import TopDownMetaDrive


def get_bev_hwc(obs):
    x = obs["image"] if isinstance(obs, dict) and "image" in obs else obs
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"Expect 3D BEV, got {x.shape}")
    if x.shape[0] in (2, 3, 4, 5) and x.shape[-1] not in (2, 3, 4, 5):
        x = np.transpose(x, (1, 2, 0))
    if x.max() > 1.0:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def save_bev_multi_frame(bev_hwc, save_path):
    # save a simple figure with each channel shown in a column
    channel_names = [
    "Road Network",
    "Checkpoints",
    "Ego Position",
    ]   
    C = bev_hwc.shape[-1]
    titles = channel_names[:C] + [f"Channel {i}" for i in range(len(channel_names), C)]
    plt.figure(figsize=(3 * C, 3), dpi=120)
    for i in range(C):
        plt.subplot(1, C, i + 1)
        plt.imshow(bev_hwc[..., i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(__file__), "bev_sequence_sss")
    os.makedirs(out_dir, exist_ok=True)

    # create environment on straight SSS map and disable rendering to speed up
    config = dict(
        map="OO",
        num_scenarios=1,
        start_seed=0,
        use_render=False,
        # ensure image observation is enabled for top-down env
        # TopDownMetaDrive generally returns top-down BEV as `image` in obs
    )

    env = TopDownMetaDrive(config)

    try:
        obs, _ = env.reset()
        bev = get_bev_hwc(obs)
        print("Initial BEV shape:", bev.shape)

        frames_for_gif = []
        max_steps = 200
        # forward action: (steering, throttle) -> keep steering 0, throttle 1.0
        action = [0, 1]

        # save initial frame
        save_bev_multi_frame(bev, os.path.join(out_dir, f"frame_{0:04d}.png"))
        if _HAS_IMAGEIO:
            # compose a RGB preview for the gif: map three channels to RGB if possible
            rgb = None
            try:
                if bev.shape[-1] >= 3:
                    rgb = np.stack([bev[..., 0], bev[..., 1], bev[..., 2]], axis=2)
                else:
                    rgb = np.repeat(bev[..., 0:1], 3, axis=2)
                frames_for_gif.append((rgb * 255).astype(np.uint8))
            except Exception:
                pass

        for i in range(1, max_steps + 1):
            o, r, terminated, truncated, info = env.step(action)
            bev = get_bev_hwc(o)
            save_bev_multi_frame(bev, os.path.join(out_dir, f"frame_{i:04d}.png"))

            if _HAS_IMAGEIO:
                try:
                    if bev.shape[-1] >= 3:
                        rgb = np.stack([bev[..., 0], bev[..., 1], bev[..., 2]], axis=2)
                    else:
                        rgb = np.repeat(bev[..., 0:1], 3, axis=2)
                    frames_for_gif.append((rgb * 255).astype(np.uint8))
                except Exception:
                    pass

            print(f"Saved frame {i:04d}", end="\r")
            if terminated or truncated:
                print("\nEpisode finished at step", i)
                break

        # try to save gif
        if _HAS_IMAGEIO and len(frames_for_gif) > 0:
            gif_path = os.path.join(out_dir, "bev_seq.gif")
            try:
                imageio.mimsave(gif_path, frames_for_gif, fps=10)
                print("Saved GIF to", gif_path)
            except Exception as e:
                print("Failed to save GIF:", e)

        print("All frames saved to:", out_dir)

    finally:
        env.close()
