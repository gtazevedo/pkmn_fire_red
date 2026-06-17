import cv2
import numpy as np
import os

files = [
    "/mnt/c/Users/guilh/.gemini/antigravity/brain/7277d84d-02ea-4c17-85c5-47df9833bc22/battle_start_env0.png",
    "/mnt/c/Users/guilh/.gemini/antigravity/brain/7277d84d-02ea-4c17-85c5-47df9833bc22/scratch/pallet_exterior.state.png"
]

for f in files:
    if not os.path.exists(f): continue
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    menu_box = img[112:160, 120:240]
    mean_val = np.mean(menu_box)
    var_val = np.var(menu_box)
    print(f"{os.path.basename(f)}: Mean={mean_val:.1f}, Var={var_val:.1f}")
