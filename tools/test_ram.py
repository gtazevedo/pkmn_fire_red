import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from src.env import PokemonEnv
from src.config import ActionSpace

env = PokemonEnv(env_id=99)
obs, _ = env.reset()

def get_coords(env):
    ram = env.env.get_ram()
    x = ram[0x036E4C] | (ram[0x036E4D] << 8) if len(ram) > 0x036E4D else -1
    y = ram[0x036E4E] | (ram[0x036E4F] << 8) if len(ram) > 0x036E4F else -1
    return x, y

print("Initial Coords:", get_coords(env))

act_down = ActionSpace.NAMES.index("DOWN")
for _ in range(16):
    env.step(act_down)

print("After walking DOWN 16 steps:", get_coords(env))

env.close()
