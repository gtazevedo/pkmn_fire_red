import stable_retro as retro
import os

state1 = os.path.abspath("scratch/events_121.state.gz")
state2 = os.path.abspath("scratch/pallet_exterior.state.gz")

env = retro.make("PokemonFireRed", state=state1)
obs = env.reset()
_, _, _, info = env.step(0)
print(f"events_121.state -> gMain_callback2: {info.get('gMain_callback2', None)}")

env2 = retro.make("PokemonFireRed", state=state2)
obs2 = env2.reset()
_, _, _, info2 = env2.step(0)
print(f"pallet_exterior.state -> gMain_callback2: {info2.get('gMain_callback2', None)}")
