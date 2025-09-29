"""
make_bev_gif.py

Find the newest recordings/bev_channel2_*.npy and convert it to a GIF.
Usage:
    python envs/make_bev_gif.py --input recordings/bev_channel2_YYYYMMDD_HHMMSS.npy --out recordings/out.gif --fps 15
If --input is not provided, the script picks the most recent matching file.
"""

import argparse
import glob
import os
import numpy as np
import imageio


def find_latest(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='.npy file to read (T,H,W)', default=None)
    parser.add_argument('--out', '-o', help='output gif path', default=None)
    parser.add_argument('--fps', type=int, default=15)
    args = parser.parse_args()

    if args.input is None:
        inp = find_latest(os.path.join('recordings', 'bev_channel2_*.npy'))
        if inp is None:
            raise FileNotFoundError('No recordings/bev_channel2_*.npy found')
    else:
        inp = args.input

    data = np.load(inp)
    if data.ndim != 3:
        raise ValueError('Expected (T,H,W) array')
    print(f'Loaded {inp}, shape={data.shape}, min={data.min()}, max={data.max()}')

    # Convert to uint8 0..255
    frames = (data * 255.0).clip(0, 255).astype('uint8')
    # Convert to RGB frames list
    imgs = [np.stack([f, f, f], axis=-1) for f in frames]

    out = args.out
    if out is None:
        base = os.path.splitext(os.path.basename(inp))[0]
        out = os.path.join('recordings', base + '.gif')

    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f'Writing GIF to {out} at {args.fps} fps ({len(imgs)} frames)')

    # imageio.mimsave accepts duration per frame as 1/fps
    imageio.mimsave(out, imgs, fps=args.fps)
    print('Done')


if __name__ == '__main__':
    main()
