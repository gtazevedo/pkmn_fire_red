import cv2
import numpy as np
from src.vision import VisionEngine

img = cv2.imread('/mnt/c/Users/guilh/.gemini/antigravity/brain/7277d84d-02ea-4c17-85c5-47df9833bc22/battle_start_env4.png')
# Convert BGR to RGB since vision engine expects RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

vision = VisionEngine()
is_battle = vision._is_battle(rgb, gray)

enemy_box = gray[16:45, 12:100]
player_box = gray[88:115, 136:228]
print('enemy_bright:', np.mean(enemy_box))
print('player_bright:', np.mean(player_box))
print('enemy_text:', np.sum(enemy_box < 50))
print('player_text:', np.sum(player_box < 50))
print('is_battle:', is_battle)
