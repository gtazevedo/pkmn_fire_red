import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from src.env import PokemonEnv

env = PokemonEnv(env_id=99)
obs = env.reset()
print(f"Initial Map: bank={env.env.data.lookup_value('map_bank')} id={env.env.data.lookup_value('map_id')}")
print(f"Initial X,Y: {env.env.data.lookup_value('x')}, {env.env.data.lookup_value('y')}")
env.close()
