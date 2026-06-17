import stable_retro as retro
import cv2
import os
import glob

env = retro.make("PokemonFireRed")

states = glob.glob("progress_states/*.state")
for state in states:
    env.reset()
    state_name = os.path.basename(state)
    with open(state, "rb") as f:
        env.em.set_state(f.read())
    obs = env.step([0]*12)[0]
    out_path = f"/mnt/c/Users/guilh/.gemini/antigravity/brain/7277d84d-02ea-4c17-85c5-47df9833bc22/scratch/{state_name}.png"
    cv2.imwrite(out_path, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    print(f"Captured {state_name}")
