from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.lange_change_policy import LaneChangePolicy

env = MetaDriveEnv(dict(
    map="C",
    log_level=50,
    discrete_action=True,
    use_multi_discrete=True,
    agent_policy=LaneChangePolicy,
    traffic_density=0,
))

obs, _ = env.reset(seed=0)

# 发送“向右变道”高层命令 (steering_cmd=2)
action = [2, 3]  # [lane-change command, throttle level]

for step in range(100):
    obs, reward, terminated, truncated, info = env.step(action)

    steering_value = env.engine.get_policy(env.agent.id).action_info["action"][0]
    print(f"step={step}, steering={steering_value:.3f}")

    env.render(
        mode="topdown",
        window=True,
        screen_size=(600, 600),
        camera_position=(60, -10)
    )

env.close()
