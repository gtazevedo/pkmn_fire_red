import cv2
import sys
import os

sys.path.append("/home/guilh/projects/pkmn_fire_red")
from src.vision import VisionEngine, GameState

img_path = "/mnt/c/Users/guilh/.gemini/antigravity/brain/7277d84d-02ea-4c17-85c5-47df9833bc22/battle_start_env4.png"
if not os.path.exists(img_path):
    print("Image not found!")
    sys.exit(1)

# cv2 loads as BGR, VisionEngine expects RGB
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

vision = VisionEngine()
state = vision.detect_state(img_rgb)
print(f"Detected State: {state}")
