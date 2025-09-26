import os
import time
import numpy as np

try:
    import pygame
except Exception:
    pygame = None

from metadrive.envs.top_down_env import TopDownMetaDrive


def get_bev_rgb_from_obs(obs):
    x = obs["image"] if isinstance(obs, dict) and "image" in obs else obs
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"Expect 3D BEV, got {x.shape}")
    # if channels first -> transpose
    if x.shape[0] in (2, 3, 4, 5) and x.shape[-1] not in (2, 3, 4, 5):
        x = np.transpose(x, (1, 2, 0))
    # normalize
    if x.max() > 1.0:
        x = x / 255.0
    x = np.clip(x, 0.0, 1.0)
    # build RGB preview: use first 3 channels if available, else repeat
    if x.shape[-1] >= 3:
        rgb = np.stack([x[..., 0], x[..., 1], x[..., 2]], axis=2)
    else:
        rgb = np.repeat(x[..., 0:1], 3, axis=2)
    return (rgb * 255).astype(np.uint8)


def run_interactive(config=None):
    if pygame is None:
        raise RuntimeError("pygame is required for interactive mode. Please install pygame.")

    cfg = dict(
        map="OO",
        num_scenarios=1,
        start_seed=0,
        use_render=False,
        manual_control=True,
    )
    if config:
        cfg.update(config)

    env = TopDownMetaDrive(cfg)

    try:
        obs, _ = env.reset()
        bev_rgb = get_bev_rgb_from_obs(obs)

        pygame.init()
        h, w = bev_rgb.shape[0], bev_rgb.shape[1]
        screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("MetaDrive BEV Interactive")

        clock = pygame.time.Clock()
        running = True
        step = 0

        # default no-op action based on action_space shape
        action = [0, 0]

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    elif event.key == pygame.K_s:
                        # save snapshot
                        out_dir = os.path.join(os.path.dirname(__file__), "interactive_snapshots")
                        os.makedirs(out_dir, exist_ok=True)
                        path = os.path.join(out_dir, f"bev_snap_{int(time.time())}.png")
                        pygame.image.save(screen, path)
                        print("Saved snapshot:", path)

            # read pressed keys for continuous control
            keys = pygame.key.get_pressed()
            # steering: left/right arrows or A/D
            steer = 0.0
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                steer = -1.0
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                steer = 1.0
            # throttle: up/down or W/S
            throttle = 0.0
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                throttle = 1.0
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                throttle = -1.0

            action = [steer, throttle]

            o, r, terminated, truncated, info = env.step(action)
            bev_rgb = get_bev_rgb_from_obs(o)

            # blit to pygame screen
            surf = pygame.surfarray.make_surface(np.transpose(bev_rgb, (1, 0, 2)))
            screen.blit(surf, (0, 0))

            # optionally render some text overlay
            font = pygame.font.SysFont(None, 20)
            text = f"Step: {step}  Reward: {r:.3f}  Terminated: {terminated}"
            txt_s = font.render(text, True, (255, 255, 255))
            screen.blit(txt_s, (5, 5))

            pygame.display.flip()
            clock.tick(30)
            step += 1

            if terminated or truncated:
                print("Episode finished, resetting environment")
                obs, _ = env.reset()
                step = 0

    finally:
        env.close()
        pygame.quit()


if __name__ == '__main__':
    run_interactive()
