"""
彩色RBG_npz_to_mp4.py

将 recordings 目录下的彩色RBG npz 文件（由 `彩色RBG奖励函数打印.py` 生成）转换为 MP4 视频。

用法示例（默认会选择最近的 npz）：
    python envs/彩色RBG_npz_to_mp4.py
或指定文件：
    python envs/彩色RBG_npz_to_mp4.py recordings/彩色RBG_reward_20250929_143459.npz

输出： recordings/<same_basename>.mp4

优先使用 imageio (ffmpeg). 如不可用，会尝试使用 OpenCV 的 VideoWriter。
"""

import os
import sys
import glob
import argparse
import numpy as np


def write_with_imageio(frames, out_path, fps=15):
    try:
        import imageio
        # imageio's ffmpeg plugin will be used
        writer = imageio.get_writer(out_path, fps=fps)
        for f in frames:
            writer.append_data(f)
        writer.close()
        return True, None
    except Exception as e:
        return False, e


def write_with_cv2(frames, out_path, fps=15):
    try:
        import cv2
        h, w = frames[0].shape[:2]
        # FourCC for mp4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
        for f in frames:
            # cv2 expects BGR
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()
        return True, None
    except Exception as e:
        return False, e


def find_latest_npz(pattern='recordings/彩色RBG_reward_*.npz'):
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    if 'frames' in data:
        frames = data['frames']
    else:
        # try to find the largest array as frames
        frames = None
        for k in data.files:
            arr = data[k]
            if isinstance(arr, np.ndarray) and arr.ndim == 4 and arr.shape[3] in (3,4):
                frames = arr
                break
        if frames is None:
            raise ValueError('No frames array found in npz')
    # Ensure uint8 and shape T,H,W,3
    if frames.dtype != np.uint8:
        # assume float in [0,1]
        frames = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
    if frames.shape[-1] != 3:
        frames = frames[..., :3]
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npz', nargs='?', help='Path to npz file (default: latest recordings/彩色RBG_reward_*.npz)')
    parser.add_argument('--fps', type=int, default=15, help='Output FPS')
    args = parser.parse_args()

    npz_path = args.npz or find_latest_npz()
    if npz_path is None or not os.path.exists(npz_path):
        print('No npz file found. Please specify path or ensure recordings/彩色RBG_reward_*.npz exists')
        sys.exit(1)

    print('Loading', npz_path)
    frames = load_npz(npz_path)
    print('Loaded frames:', frames.shape, frames.dtype)

    out_path = os.path.splitext(npz_path)[0] + '.mp4'

    print('Attempting to write MP4 with imageio...')
    ok, err = write_with_imageio(frames, out_path, fps=args.fps)
    if ok:
        print('Saved mp4 to', out_path)
        return
    else:
        print('imageio write failed:', err)

    print('Falling back to OpenCV...')
    ok, err = write_with_cv2(frames, out_path, fps=args.fps)
    if ok:
        print('Saved mp4 to', out_path)
        return
    else:
        print('OpenCV write failed:', err)

    print('Failed to write mp4. Please install imageio[ffmpeg] or opencv-python')

if __name__ == '__main__':
    main()
