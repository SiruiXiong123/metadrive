"""Debug script to call nav._get_info_for_checkpoint and env.reward_function without modifying library code.
Run this in your conda env. It prints navi_info and reward results to the main terminal.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.metadrive_env import MetaDriveEnv
import numpy as np

# Minimal config to create env quickly
cfg = dict(
    num_scenarios=1,
    start_seed=1000,
    use_render=False,
    image_observation=False,
    random_lane_width=False,
    random_lane_num=False,
)

print('[debug] creating env')
env = MetaDriveEnv(cfg)
print('[debug] resetting env')
obs, _ = env.reset()
print('[debug] reset done')

# get the first vehicle (robust to agent id)
vehicle = next(iter(env.agents.values()))
print('[debug] vehicle id:', vehicle.id)
nav = vehicle.navigation
print('[debug] nav type:', type(nav))

# current and next reference lanes
ref_lane_cur = nav.current_ref_lanes[0]
ref_lane_next = nav.next_ref_lanes[0] if nav.next_ref_lanes is not None else ref_lane_cur

print('[debug] ref_lane_cur type:', type(ref_lane_cur))

# call the internal method and print results
try:
    navi_info, lanes_heading, cp = nav._get_info_for_checkpoint(lanes_id=0, ref_lane=ref_lane_cur, ego_vehicle=vehicle)
    print('[debug] _get_info_for_checkpoint returned:')
    print('  navi_info:', navi_info)
    print('  lanes_heading:', lanes_heading)
    print('  checkpoint (world):', cp)
except Exception as e:
    import traceback
    print('[debug] exception when calling _get_info_for_checkpoint:')
    traceback.print_exc()

# call get_checkpoints() as an alternative
try:
    ckpt, _ = vehicle.navigation.get_checkpoints()
    print('[debug] get_checkpoints returned ckpt:', ckpt)
except Exception as e:
    print('[debug] get_checkpoints exception:', e)

# call env.reward_function directly and print returned values
try:
    print('[debug] calling env.reward_function(vehicle.id)')
    reward, step_info = env.reward_function(vehicle.id)
    print('[debug] reward_function returned reward=', reward)
    print('[debug] step_info keys:', list(step_info.keys()))
except Exception as e:
    import traceback
    print('[debug] exception when calling env.reward_function:')
    traceback.print_exc()

print('[debug] closing env')
env.close()
print('[debug] done')
