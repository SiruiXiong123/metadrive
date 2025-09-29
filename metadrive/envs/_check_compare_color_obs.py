import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from metadrive.envs.top_down_env import TopDownMetaDrive
from metadrive.constants import DEFAULT_AGENT

# Load saved image
img_path = 'figures/彩色_bev_frame.png'
try:
    img = np.array(Image.open(img_path).convert('RGB'))
    print('Loaded saved image', img.shape, img.dtype)
except Exception as e:
    print('Failed to load saved image:', e)
    img = None

# Create env and get observation (debug_color True)
cfg = {
    'map': 'SS',
    'use_render': False,
    'num_scenarios': 1,
    'start_seed': 123,
    'distance': 25,
    'resolution_size': 180,
    'traffic_density': 0.0,
    'vehicle_config': {'show_navi_mark': True, 'show_line_to_navi_mark': True},
}

env = TopDownMetaDrive(cfg)
try:
    obs, _ = env.reset()
    obs_obj = env.observations[DEFAULT_AGENT]
    if hasattr(obs_obj, 'debug_color'):
        obs_obj.debug_color = True
    # reset again to ensure color mode applied
    obs, _ = env.reset()
    img_obs = obs if not isinstance(obs, dict) else obs.get('image', obs)
    img_obs = np.asarray(img_obs)
    print('Observation image shape/dtype:', img_obs.shape, img_obs.dtype, 'min/max:', img_obs.min(), img_obs.max())

    # If img_obs is float in [0,1], convert to uint8
    if img_obs.dtype == np.float32 or img_obs.max() <= 1.0:
        img_obs_u8 = (np.clip(img_obs, 0, 1) * 255).astype(np.uint8)
    else:
        img_obs_u8 = img_obs.astype(np.uint8)

    # transpose if needed (CHW -> HWC)
    if img_obs_u8.shape[0] in (2,3,4,5) and img_obs_u8.shape[-1] not in (2,3,4,5):
        img_obs_u8 = np.transpose(img_obs_u8, (1,2,0))

    # Compare unique colors presence
    def has_color(arr, rgb, tol=3):
        # check if any pixel within tol of rgb
        r,g,b = rgb
        diff = np.abs(arr.astype(int) - np.array([r,g,b])[None,None,:])
        mask = np.all(diff <= tol, axis=2)
        return mask.any()

    targets = {
        'ego_green': (0,255,0),
        'nav_light_blue': (135,206,250),
        'lane_orange': (255,175,35),
        'drivable_red': (255,0,0)
    }

    if img is not None:
        print('\nSaved image color presence:')
        for k,c in targets.items():
            print(k, has_color(img, c))

    print('\nObserved image color presence:')
    for k,c in targets.items():
        print(k, has_color(img_obs_u8, c))

finally:
    env.close()
