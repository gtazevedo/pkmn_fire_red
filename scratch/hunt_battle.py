import stable_retro as retro
import cv2
import os

env = retro.make("PokemonFireRed")
env.reset()

state_path = "progress_states/pallet_exterior.state"
with open(state_path, "rb") as f:
    env.em.set_state(f.read())

print("Hunting for a battle by running straight UP into Route 1...")
last_cb2 = 0
for i in range(5000):
    # Walk UP and DOWN in grass in Route 1
    action = [0]*12
    if i < 200:
        action[4] = 1 # Walk UP to Route 1
    else:
        if (i // 16) % 2 == 0:
            action[4] = 1 # UP
        else:
            action[5] = 1 # DOWN
        
    result = env.step(action)
    info = result[-1]
    
    cb2 = info.get("gMain_callback2", 0)
    
    if cb2 != last_cb2:
        print(f"Callback changed! gMain_callback2: {hex(cb2)} at step {i}")
        last_cb2 = cb2
        
        # We know 0x80565b5 is standard overworld
        if cb2 not in [0, 134571445, 134571853, 134572041, 0x805671d]:
            cv2.imwrite(f"/mnt/c/Users/guilh/.gemini/antigravity/brain/7277d84d-02ea-4c17-85c5-47df9833bc22/scratch/cb2_0x{cb2:x}_step{i}.png", cv2.cvtColor(result[0], cv2.COLOR_RGB2BGR))
