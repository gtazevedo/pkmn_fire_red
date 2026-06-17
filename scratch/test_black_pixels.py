import cv2
import numpy as np
import os

files = [
    "/mnt/c/Users/guilh/.gemini/antigravity/brain/7277d84d-02ea-4c17-85c5-47df9833bc22/battle_start_env0.png",
    "/mnt/c/Users/guilh/.gemini/antigravity/brain/7277d84d-02ea-4c17-85c5-47df9833bc22/scratch/pallet_exterior.state.png",
    "/mnt/c/Users/guilh/.gemini/antigravity/brain/7277d84d-02ea-4c17-85c5-47df9833bc22/scratch/events_121.state.png"
]

for f in files:
    if not os.path.exists(f): continue
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    
    enemy_box = img[16:45, 12:100]
    player_box = img[88:115, 136:228]
    
    enemy_black = np.sum(enemy_box < 50)
    player_black = np.sum(player_box < 50)
    
    enemy_mean = np.mean(enemy_box)
    player_mean = np.mean(player_box)
    
    print(f"{os.path.basename(f)}: EnemyBlack={enemy_black}, PlayerBlack={player_black}, EnemyMean={enemy_mean:.1f}, PlayerMean={player_mean:.1f}")
