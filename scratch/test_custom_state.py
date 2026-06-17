import stable_retro as retro
import os

env = retro.make("PokemonFireRed")
env.reset()

def test_state(state_name):
    state_path = f"progress_states/{state_name}"
    with open(state_path, "rb") as f:
        env.em.set_state(f.read())
        
    info = env.step([0]*12)[-1]
    cb2 = info.get('gMain_callback2', 0)
    b_struct = info.get('gBattleStruct_ptr', 0)
    print(f"{state_name} -> CB2: {hex(cb2)}, BattleStruct: {hex(b_struct)}")

test_state("events_112.state")
test_state("events_113.state")
test_state("events_118.state")
test_state("events_120.state")
test_state("events_121.state")
test_state("pallet_exterior.state")
