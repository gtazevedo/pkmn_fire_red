import stable_retro as retro
import cv2
import os

env = retro.make("PokemonFireRed")
env.reset()

state_path = "progress_states/events_121.state"
with open(state_path, "rb") as f:
    env.em.set_state(f.read())

obs = env.step([0]*12)[0]
img_path = "/mnt/c/Users/guilh/.gemini/antigravity/brain/7277d84d-02ea-4c17-85c5-47df9833bc22/scratch/state_121_real.png"
obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
cv2.imwrite(img_path, obs_bgr)
print(f"Snapshot saved to {img_path}")
