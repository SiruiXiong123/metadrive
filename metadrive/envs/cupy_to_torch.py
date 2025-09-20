# cupy_to_torch.py
import torch

def cupy_to_torch(x, device=None, dtype=None):
    try:
        import cupy as cp
    except ImportError:
        cp = None

    if cp is not None and isinstance(x, cp.ndarray):
        t = torch.utils.dlpack.from_dlpack(x.toDlpack())  # 无拷贝
        if dtype is not None:
            t = t.to(dtype)
        return t if device is None else t.to(device, non_blocking=True)

    import numpy as np
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
        if dtype is not None:
            t = t.to(dtype)
        return t if device is None else t.to(device, non_blocking=True)

    if isinstance(x, torch.Tensor):
        return x if device is None else x.to(device, non_blocking=True)

    return x
