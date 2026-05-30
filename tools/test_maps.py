import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from src.env import PokemonEnv
from src.reader import RamReader
from src.config import ActionSpace

env = PokemonEnv(env_id=99)
obs, _ = env.reset()

act_b = ActionSpace.NAMES.index("B")
obs, r, d, t, info = env.step(act_b)
print("INFO KEYS:", list(info.keys()))

env.close()
