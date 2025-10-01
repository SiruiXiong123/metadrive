import os
import numpy as np
import matplotlib.pyplot as plt
from metadrive.envs.top_down_env import TopDownMetaDrive
from metadrive.constants import DEFAULT_AGENT


def get_bev_hwc(obs):
    """Ensure HWC format and pixel range [0,1]"""
    x = obs["image"] if isinstance(obs, dict) and "image" in obs else obs
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"Expect 3D BEV, got {x.shape}")
    # If CHW -> to HWC
    if x.shape[0] in (2, 3, 4, 5) and x.shape[-1] not in (2, 3, 4, 5):
        x = np.transpose(x, (1, 2, 0))
    if x.max() > 1.0:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


if __name__ == '__main__':
    cfg = {
        "map": "SS",
        "use_render": False,
        "num_scenarios": 1,
        "start_seed": 123,
        "distance": 50,
        "resolution_size": 224,
        "traffic_density": 0.0,
        "vehicle_config": {"show_navi_mark": True, "show_line_to_navi_mark": True},
    }

    env = TopDownMetaDrive(cfg)
    try:
        obs, _ = env.reset()
        # enable debug color on the per-agent observation object
        obs_obj = env.observations[DEFAULT_AGENT]
        # render one step to ensure surfaces ready
        obs, _ = env.reset()
        img = obs if not isinstance(obs, dict) else obs.get('image', obs)
        bev = get_bev_hwc(img)
        print('BEV shape HWC:', bev.shape, 'dtype:', bev.dtype, 'min/max:', bev.min(), bev.max())
        # show
        plt.figure(figsize=(6, 6))
        plt.imshow(bev)
        plt.axis('off')
        plt.title('Color BEV (debug_color=True)')
        os.makedirs('figures', exist_ok=True)
        plt.savefig('figures/彩色_bev_frame.png', dpi=200, bbox_inches='tight')
        plt.show()
        print('[OK] Saved figures/彩色_bev_frame.png')
    finally:
        env.close()


