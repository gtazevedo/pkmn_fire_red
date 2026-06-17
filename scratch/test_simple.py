import stable_retro as retro
env = retro.make("PokemonFireRed") # uses default state
obs = env.reset()
result = env.step([0]*12)
info = result[-1]
print(f"Start state -> CB2: {info.get('gMain_callback2')} (hex: {hex(info.get('gMain_callback2', 0))})")
print(f"Start state -> gBattleStruct_ptr: {info.get('gBattleStruct_ptr')} (hex: {hex(info.get('gBattleStruct_ptr', 0))})")
