import cv2
import numpy as np
from enum import Enum, auto

class GameState(Enum):
    OVERWORLD = auto()
    BATTLE = auto()
    DIALOGUE = auto()
    WHITEOUT = auto()

class VisionEngine:
    def __init__(self):
        # Fire Red GBA resolution is 240x160
        self.width = 240
        self.height = 160

    def detect_state(self, frame_rgb: np.ndarray) -> GameState:
        """
        Main pipeline to classify the current frame into a GameState.
        Takes a 160x240x3 RGB numpy array.
        """
        # 1. Whiteout / Black screen check
        # If the average brightness is extremely low
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        if np.mean(gray) < 15:
            return GameState.WHITEOUT
            
        # 2. Battle Detection
        if self._is_battle(frame_rgb, gray):
            return GameState.BATTLE
            
        # 3. Dialogue Detection
        if self._is_dialogue(frame_rgb, gray):
            return GameState.DIALOGUE
            
        # 4. Default Overworld
        return GameState.OVERWORLD

    def _is_battle(self, frame_rgb: np.ndarray, gray: np.ndarray) -> bool:
        """
        Detects if the screen is in the battle engine.
        We look for the distinct Enemy HP Box and Player HP Box.
        In Fire Red, these are light-colored boxes with dark borders at specific locations.
        """
        # Top-left enemy box region approx: y=10:50, x=10:110
        # Bottom-right player box region approx: y=80:120, x=130:230
        
        # A simple and very robust heuristic is checking the dominant colors
        # in the health bar regions. Health bars are purely green, yellow, or red.
        # But for now, we will use a brightness check on the distinct white/light areas
        # of the HP boxes.
        
        enemy_box = gray[16:45, 12:100]
        player_box = gray[88:115, 136:228]
        
        # In battle, these regions contain the bright white background of the HP UI.
        # We also check for black pixels (< 50) to ensure there is actually text (HP numbers/letters)
        # and it's not just a blank white wall or floor.
        enemy_bright = np.mean(enemy_box) > 200
        player_bright = np.mean(player_box) > 200
        enemy_text = np.sum(enemy_box < 50) > 10
        player_text = np.sum(player_box < 50) > 10
        
        if enemy_bright and player_bright and enemy_text and player_text:
            return True
            
        return False

    def _is_dialogue(self, frame_rgb: np.ndarray, gray: np.ndarray) -> bool:
        """
        Detects if a standard text box is present at the bottom of the screen.
        """
        # Dialogue box is usually at y=112:152, x=8:232
        dialogue_box = gray[112:152, 8:232]
        
        # A dialogue box has a white interior and a dark border.
        # If the interior is very bright and uniform, it's a dialogue.
        if np.mean(dialogue_box) > 220:
            return True
            
        return False
