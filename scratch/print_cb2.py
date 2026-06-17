import stable_retro as retro
import os

env = retro.make("PokemonFireRed")
env.reset()

for state in ["pallet_exterior.state", "events_121.state", "events_112.state", "events_120.state", "events_118.state"]:
    state_path = f"progress_states/{state}"
    if not os.path.exists(state_path):
        continue
    with open(state_path, "rb") as f:
        env.em.set_state(f.read())
    
    result = env.step([0]*12)
    info = result[-1]
    cb2 = info.get("gMain_callback2", 0)
    print(f"State {state}: gMain_callback2 = {hex(cb2)}")
