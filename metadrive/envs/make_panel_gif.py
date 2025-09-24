import os
import glob

try:
    import imageio
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False

from PIL import Image

frames_dir = os.path.join(os.path.dirname(__file__), "bev_sequence_sss")
pattern = os.path.join(frames_dir, "frame_*.png")
files = sorted(glob.glob(pattern))

if not files:
    print("No frame PNGs found at:", pattern)
    raise SystemExit(1)

print(f"Found {len(files)} frames, first: {files[0]}, last: {files[-1]}")

gif_path = os.path.join(frames_dir, "bev_seq_panel.gif")

if _HAS_IMAGEIO:
    try:
        imgs = [imageio.imread(f) for f in files]
        imageio.mimsave(gif_path, imgs, fps=10)
        print("Saved GIF to", gif_path)
        raise SystemExit(0)
    except Exception as e:
        print("imageio failed:", e)

# fallback to PIL
try:
    pil_imgs = [Image.open(f).convert("RGBA") for f in files]
    pil_imgs[0].save(gif_path, save_all=True, append_images=pil_imgs[1:], duration=100, loop=0)
    print("Saved GIF (PIL) to", gif_path)
except Exception as e:
    print("Failed to save GIF with PIL:", e)
    raise SystemExit(2)
