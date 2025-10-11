# @title Draw the generated maps in top-down view
from metadrive.engine.engine_utils import close_engine
close_engine()
# NOTE: usually you don't need the above lines. It is only for avoiding a potential bug when running on colab


import random

import matplotlib.pyplot as plt
from metadrive import MetaDriveEnv
from metadrive.utils.draw_top_down_map import draw_top_down_map

env = MetaDriveEnv(config=dict(
    num_scenarios=100,
    map=2,
    start_seed=random.randint(0, 1000)
))

fig, axs = plt.subplots(3, 3, figsize=(10, 10), dpi=100)
for i in range(3):
    for j in range(3):
        env.reset()
        m = draw_top_down_map(env.current_map)
        ax = axs[i][j]
        ax.imshow(m, cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
fig.suptitle("Bird's-eye view of generated maps")
plt.show()

env.close()
