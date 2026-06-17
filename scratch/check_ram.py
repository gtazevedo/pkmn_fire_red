import retro
import numpy as np

env = retro.make("PokemonFireRed")
ram = env.get_ram()
print(f"RAM size: {len(ram)}")
