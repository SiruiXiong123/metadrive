import os
import sys
from PIL import Image
import numpy as np

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def main():
    try:
        from metadrive.envs.top_down_env import TopDownMetaDriveEnvV2
    except Exception as e:
        print("Import env failed:", e)
        return

    try:
        env = TopDownMetaDriveEnvV2()
    except Exception as e:
        print("Create env failed:", e)
        return

    try:
        res = env.reset()
        obs = res[0] if isinstance(res, tuple) else res
    except Exception as e:
        print("Env reset failed:", e)
        env.close()
        return

    obs = np.array(obs)
    print("obs shape:", obs.shape, "dtype:", obs.dtype, "min/max:", obs.min(), obs.max())

    # obs expected shape (H, W, C)
    out_dir = os.path.join(ROOT, "smoke_out")
    os.makedirs(out_dir, exist_ok=True)

    for i in range(min(obs.shape[2], 2)):
        ch = obs[..., i]
        # normalize to 0-255
        ch = ch.astype(np.float32)
        if ch.max() <= 1.0:
            ch = (ch * 255.0)
        ch = np.clip(ch, 0, 255).astype(np.uint8)
        im = Image.fromarray(ch)
        path = os.path.join(out_dir, f"channel_{i}.png")
        im.save(path)
        print("Saved", path)

    env.close()

if __name__ == '__main__':
    main()
