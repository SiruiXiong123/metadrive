import numpy as np
import sys

print("Python:", sys.executable)

try:
    from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
    print("Imported TopDownMultiChannel")
except Exception as e:
    print("Import failed:", e)
    raise

try:
    inst = TopDownMultiChannel(vehicle_config={}, onscreen=False, clip_rgb=True)
    print("Instantiated TopDownMultiChannel")
except Exception as e:
    print("Instantiation failed:", e)
    raise

# Show key attributes
print("norm_pixel:", getattr(inst, 'norm_pixel', None))
print("debug_color:", getattr(inst, 'debug_color', None))
print("num_stacks:", getattr(inst, 'num_stacks', None))
print("resolution:", getattr(inst, 'resolution', None))

# Observation space inspection
try:
    sp = inst.observation_space
    print("observation_space:", sp)
    try:
        print("space dtype:", sp.dtype)
    except Exception:
        print("space dtype not available")
    try:
        print("space low min/max:", np.min(sp.low), np.max(sp.low))
        print("space high min/max:", np.min(sp.high), np.max(sp.high))
    except Exception as e:
        print("Couldn't inspect low/high:", e)
except Exception as e:
    print("Failed to get observation_space:", e)

# Test _transform behavior with a white RGB image
try:
    shape = None
    if hasattr(sp, 'shape'):
        shape = sp.shape
    else:
        # fallback to instance resolution
        res = getattr(inst, 'resolution', None)
        if res is not None:
            shape = (res[1], res[0], 3)  # obs_shape likely (H,W)

    if shape is None:
        print("Can't determine sample shape to test _transform")
    else:
        # make a white image in surfarray style (W,H,3) or (H,W,3) depending on internal use
        # _transform expects (W,H,3) as used earlier in file, but implementation uses axis as last dim.
        # We'll create an array with shape (shape[0], shape[1], 3)
        h, w = shape[0], shape[1]
        sample = np.ones((h, w, 3), dtype=np.uint8) * 255
        transformed = inst._transform(sample)
        print("_transform result dtype:", transformed.dtype)
        print("_transform min/max:", float(np.min(transformed)), float(np.max(transformed)))
except Exception as e:
    print("_transform test failed:", e)

print("Done.")
