import os
import glob

try:
    import imageio
except Exception:
    raise RuntimeError("imageio is required. Install with: python -m pip install imageio")

out_dir = os.path.join(os.path.dirname(__file__), "bev_sequence_sss")
pattern = os.path.join(out_dir, "frame_*.png")
files = sorted(glob.glob(pattern))
if not files:
    raise SystemExit(f"No frames found in {out_dir}")

frames = []
for f in files:
    frames.append(imageio.imread(f))

gif_path = os.path.join(out_dir, "bev_seq_generated.gif")
imageio.mimsave(gif_path, frames, fps=10)
print("Saved GIF to:", gif_path)
