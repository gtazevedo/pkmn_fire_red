import stable_retro as retro
import cv2
import os

env = retro.make("PokemonFireRed")
env.reset()

state_path = "progress_states/events_118.state"
with open(state_path, "rb") as f:
    env.em.set_state(f.read())

result = env.step([0]*12)
obs = result[0]
cv2.imwrite("/mnt/c/Users/guilh/.gemini/antigravity/brain/7277d84d-02ea-4c17-85c5-47df9833bc22/scratch/events_118.png", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
