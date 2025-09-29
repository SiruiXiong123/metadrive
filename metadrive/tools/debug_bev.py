import os
import numpy as np
import matplotlib.pyplot as plt
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


def save_debug_images(bev_hwc, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    H, W, C = bev_hwc.shape
    for i in range(C):
        arr = bev_hwc[..., i]
        # save as grayscale PNG
        path = os.path.join(out_dir, f"bev_ch{i}.png")
        plt.imsave(path, arr, cmap='gray')
    # also save combined
    plt.figure(figsize=(3*C, 4), dpi=150)
    for i in range(C):
        plt.subplot(1, C, i+1)
        plt.imshow(bev_hwc[..., i], cmap='gray')
        plt.axis('off')
    plt.savefig(os.path.join(out_dir, 'bev_combined.png'), bbox_inches='tight')
    plt.close()


cfg = {
    "map": "SS",
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
    },
    "distance": 25,
    "resolution_size": 160,
}

if __name__ == '__main__':
    env = TopDownMetaDrive(cfg)
    out = 'debug_bev_out'
    try:
        obs, _ = env.reset()
        bev = get_bev_hwc(obs)
        print('BEV shape:', bev.shape, 'dtype:', bev.dtype, 'min/max:', bev.min(), bev.max())
        H, W, C = bev.shape
        for i in range(C):
            ch = bev[..., i]
            print(f'channel {i}: min={ch.min()}, max={ch.max()}, mean={ch.mean():.4f}')
            # print last 5 rows summary
            last_rows = ch[-5:,:,:] if ch.ndim==3 else ch[-5:,:]
            # compute per-row sums
            if ch.ndim==2:
                row_sums = ch.sum(axis=1)
                print('  last 5 row sums:', row_sums[-5:])
            else:
                print('  shape for last rows:', last_rows.shape)
        save_debug_images(bev, out)
        print('Saved debug images to', out)
    finally:
        env.close()
