import os
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse


def create_multichannel_gif(obs_dir: str, output_gif: str, fps: int = 10, num_channels: int = None):
    """
    Convert .npz observation files to animated GIF showing all channels side by side.
    
    Args:
        obs_dir: Directory containing .npz files
        output_gif: Path to save the output GIF
        fps: Frames per second for the GIF
        num_channels: Number of channels to display (if None, auto-detect from first file)
    """
    # 列出所有 .npz 文件并按步数排序
    npz_files = sorted(glob.glob(os.path.join(obs_dir, "*.npz")))
    
    if not npz_files:
        print(f"[ERROR] No .npz files found in {obs_dir}")
        return
    
    print(f"[INFO] Found {len(npz_files)} .npz files")
    
    # 从第一个文件探测通道数和图像大小
    try:
        first_obs = np.load(npz_files[0])['observation']
        print(f"[INFO] First observation shape: {first_obs.shape}")
        
        if len(first_obs.shape) == 3:
            height, width, channels = first_obs.shape
            if num_channels is None:
                num_channels = channels
        else:
            print(f"[ERROR] Unexpected observation shape: {first_obs.shape}")
            return
    except Exception as e:
        print(f"[ERROR] Failed to load first observation: {e}")
        return
    
    print(f"[INFO] Image size: {height}x{width}, Channels: {num_channels}")
    
    frames = []
    
    for i, npz_file in enumerate(npz_files):
        try:
            obs = np.load(npz_file)['observation']
            
            # 验证形状
            if obs.shape != first_obs.shape:
                print(f"[WARN] Shape mismatch at {npz_file}: {obs.shape} vs {first_obs.shape}, skipping")
                continue
            
            # 创建一个大图像来容纳所有通道并排显示
            # 计算布局：尽可能接近正方形
            cols = int(np.ceil(np.sqrt(num_channels)))
            rows = int(np.ceil(num_channels / cols))
            
            # 单个通道图像
            channel_size = 100  # 缩小到 100x100 用于显示
            
            # 计算总图像大小（包括间距和标签）
            label_height = 20
            gap = 5
            total_width = cols * (channel_size + gap) + gap
            total_height = rows * (channel_size + label_height + gap) + gap
            
            # 创建背景
            composite = Image.new('L', (total_width, total_height), color=0)
            
            # 将每个通道绘制到对应位置
            for ch_idx in range(num_channels):
                row = ch_idx // cols
                col = ch_idx % cols
                
                x = col * (channel_size + gap) + gap
                y = row * (channel_size + label_height + gap) + gap
                
                # 获取该通道的数据
                channel_data = obs[..., ch_idx]
                
                # 归一化到 0-255
                if channel_data.dtype == np.float32 or channel_data.dtype == np.float64:
                    if channel_data.max() <= 1.0:
                        channel_data = (channel_data * 255).astype(np.uint8)
                    else:
                        channel_data = channel_data.astype(np.uint8)
                else:
                    channel_data = channel_data.astype(np.uint8)
                
                # 缩放到 channel_size
                channel_img = Image.fromarray(channel_data, mode='L')
                channel_img = channel_img.resize((channel_size, channel_size), Image.BILINEAR)
                
                # 粘贴到复合图像
                composite.paste(channel_img, (x, y))
                
                # 添加通道标签（简单方式：直接用画笔）
                # 如果有问题可以跳过标签
                try:
                    draw = ImageDraw.Draw(composite)
                    label_text = f"Ch{ch_idx}"
                    draw.text((x, y - label_height + 2), label_text, fill=255)
                except:
                    pass
            
            frames.append(composite)
            
            if (i + 1) % 50 == 0:
                print(f"[INFO] Loaded {i + 1} frames...")
        
        except Exception as e:
            print(f"[WARN] Failed to load {npz_file}: {e}")
            continue
    
    if not frames:
        print("[ERROR] No valid frames loaded")
        return
    
    print(f"[INFO] Total {len(frames)} frames loaded")
    
    # 保存为 GIF
    duration = int(1000 / fps)  # 毫秒
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    
    print(f"[SUCCESS] GIF saved to {output_gif}")
    print(f"[INFO] GIF details: {len(frames)} frames, {fps} fps, {cols}x{rows} layout")


def main():
    parser = argparse.ArgumentParser(description="Convert .npz multi-channel observations to animated GIF")
    parser.add_argument("--obs_dir", type=str,
                        default=r"C:\Users\37945\OneDrive\Desktop\obs\obs",
                        help="Directory containing .npz files")
    parser.add_argument("--output", type=str,
                        default=r"C:\Users\37945\OneDrive\Desktop\obs\observation_multichannel.gif",
                        help="Output GIF file path")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for GIF")
    parser.add_argument("--channels", type=int, default=None, help="Number of channels to display (auto-detect if not specified)")
    
    args = parser.parse_args()
    
    create_multichannel_gif(args.obs_dir, args.output, fps=args.fps, num_channels=args.channels)


if __name__ == '__main__':
    main()
