"""
text_advisor.py
===============
Local text understanding for the Pokemon FireRed RL agent.
Zero API cost — runs entirely on your machine.

Pipeline
--------
  GBA screen (160×144 RGB)
      ↓
  DialogDetector  — pixel heuristics to detect if a text box is visible
      ↓  (only runs OCR when a text box is actually open — saves CPU)
  ScreenOCR       — EasyOCR reads the GBA pixel font
      ↓
  SituationClassifier — rule-based: maps text → structured game situation
      ↓
  TextAdvisor     — combines situation into:
                      • extra RAM features (fed to the policy network)
                      • shaped reward signal (teaches correct button presses)

Installation
------------
  pip install easyocr

  EasyOCR downloads its model (~100MB) on first run.
  After that it runs fully offline.

  If easyocr is too slow on your CPU, use pytesseract instead:
  pip install pytesseract  (also needs: apt install tesseract-ocr)
  Set USE_TESSERACT = True below.

Integration with pokemon_rl.py
-------------------------------
  1. from text_advisor import TextAdvisor, SituationEmbedding
  2. In PokemonEnv.__init__:      self._advisor = TextAdvisor()
  3. In PokemonEnv._build_obs:    expand ram vector with advisor.get_embedding()
  4. In PokemonEnv.step:          reward += advisor.update(screen_rgb, action_idx)
  5. Update observation_space ram shape: (12,) → (12 + SituationEmbedding.SIZE,)

The advisor runs OCR at most once every N steps (configurable) and caches
the result between calls, so it adds minimal overhead to the training loop.
"""

import re
import time
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("pokemon_rl")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

USE_TESSERACT       = True    # EasyOCR takes 2-8s/call on CPU = unusable. Tesseract ~0.08s.
OCR_EVERY_N_STEPS   = 100     # 50 calls/episode * 0.08s = ~4s overhead. Acceptable.
DIALOG_BOX_BONUS    = 0.05    # reward for pressing A/B correctly in a dialog
BATTLE_MENU_BONUS   = 0.10    # reward for pressing A when Fight menu is open
MOVE_SELECT_BONUS   = 0.10    # reward for pressing A when selecting a move
HEAL_DETECT_BONUS   = 0.20    # reward for reaching the Pokemon Center
RUN_BONUS           = 0.05    # reward for pressing B to run from weak battles

# GBA screen region where the dialog box appears (bottom ~25% of screen)
# GBA native res: 240×160. After our 84×84 resize these coords scale too.
# We crop the ORIGINAL 240×160 frame before resize for OCR — better quality.
DIALOG_BOX_Y_START  = 110     # pixels from top in native 160px height
DIALOG_BOX_Y_END    = 160
DIALOG_BOX_X_START  = 0
DIALOG_BOX_X_END    = 240

# Pixel color of the dialog box border in FireRed (dark blue/black frame)
# Used by DialogDetector to avoid running OCR every step
DIALOG_BORDER_COLOR = (24, 56, 88)   # approximate RGB
DIALOG_BORDER_TOL   = 40             # tolerance


# ---------------------------------------------------------------------------
# Game situations (what is currently happening on screen)
# ---------------------------------------------------------------------------

class Situation(IntEnum):
    UNKNOWN       = 0
    EXPLORING     = 1   # overworld, no text box
    DIALOG        = 2   # NPC dialogue / story text
    BATTLE_MENU   = 3   # top-level battle menu: FIGHT / BAG / POKéMON / RUN
    MOVE_SELECT   = 4   # move selection screen
    ITEM_MENU     = 5   # bag / item menu
    LEVEL_UP      = 6   # "xxx grew to level N!" screen
    HEAL_CONFIRM  = 7   # "Your Pokemon have been healed" at Pokemon Center
    YES_NO        = 8   # yes/no dialog prompt
    NAME_ENTRY    = 9   # naming a Pokemon / player


# Recommended action for each situation
SITUATION_ACTION: dict[Situation, str] = {
    Situation.UNKNOWN:      "B",     # safe default: close any menu
    Situation.EXPLORING:    "",      # no text hint — let the policy decide
    Situation.DIALOG:       "A",     # advance text
    Situation.BATTLE_MENU:  "A",     # open Fight submenu
    Situation.MOVE_SELECT:  "A",     # select first move (usually damaging)
    Situation.ITEM_MENU:    "B",     # close bag (agent shouldn't use items yet)
    Situation.LEVEL_UP:     "A",     # dismiss level-up screen
    Situation.HEAL_CONFIRM: "A",     # confirm healing
    Situation.YES_NO:       "B",     # default: No (avoids accidental nicknames etc.)
    Situation.NAME_ENTRY:   "B",     # skip naming
}

# Reward given when agent presses the recommended button for its situation
SITUATION_BONUS: dict[Situation, float] = {
    Situation.UNKNOWN:      0.0,
    Situation.EXPLORING:    0.0,
    Situation.DIALOG:       DIALOG_BOX_BONUS,
    Situation.BATTLE_MENU:  BATTLE_MENU_BONUS,
    Situation.MOVE_SELECT:  MOVE_SELECT_BONUS,
    Situation.ITEM_MENU:    0.05,
    Situation.LEVEL_UP:     0.05,
    Situation.HEAL_CONFIRM: HEAL_DETECT_BONUS,
    Situation.YES_NO:       0.05,
    Situation.NAME_ENTRY:   0.05,
}


# ---------------------------------------------------------------------------
# Situation embedding (fed into the policy network as extra RAM features)
# ---------------------------------------------------------------------------

@dataclass
class SituationEmbedding:
    """
    Fixed-size float vector encoding the current text situation.
    Appended to the existing RAM observation vector.
    """

    SIZE = 5   # number of extra features added to the RAM vector

    situation:         Situation = Situation.UNKNOWN
    recommended_action_idx: int = 1   # default: B (index 1 in ActionSpace)
    confidence:        float = 0.0
    steps_in_situation: int = 0
    last_ocr_age:      float = 1.0    # 0=fresh, 1=stale (normalised)

    def to_array(self, max_steps_in_sit: int = 200) -> np.ndarray:
        return np.array([
            self.situation / len(Situation),          # normalised situation id
            self.recommended_action_idx / 5.0,        # normalised action (0-5)
            self.confidence,                           # 0.0-1.0
            min(self.steps_in_situation, max_steps_in_sit) / max_steps_in_sit,
            min(self.last_ocr_age, 1.0),
        ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Dialog box detector (fast pixel heuristic — skips OCR when no text box)
# ---------------------------------------------------------------------------

class DialogDetector:
    """
    Detects if a dialog/text box is currently visible on screen.
    Uses a simple pixel color heuristic on the bottom region of the
    native GBA frame — much faster than running OCR every step.
    """

    # The dialog box in FireRed has a consistent dark border.
    # We check a few pixel positions that are always part of that border.
    # Coordinates are in native 240×160 GBA resolution.
    _BORDER_SAMPLES = [
        (8,  112), (8,  155), (232, 112), (232, 155),   # corners
        (120, 112), (120, 155),                           # midpoints
    ]

    @classmethod
    def is_dialog_visible(cls, frame_rgb: np.ndarray) -> bool:
        """
        frame_rgb: native GBA frame, shape (160, 240, 3), dtype uint8.
        Returns True if a dialog box border is detected.
        """
        if frame_rgb.shape[:2] != (160, 240):
            # Already resized — skip detection (assume possible dialog)
            return True

        for x, y in cls._BORDER_SAMPLES:
            pixel = frame_rgb[y, x].astype(int)
            target = np.array(DIALOG_BORDER_COLOR, dtype=int)
            if np.max(np.abs(pixel - target)) > DIALOG_BORDER_TOL:
                return False
        return True

    @classmethod
    def crop_dialog_region(cls, frame_rgb: np.ndarray) -> np.ndarray:
        """Extract the dialog text region for OCR."""
        return frame_rgb[
            DIALOG_BOX_Y_START:DIALOG_BOX_Y_END,
            DIALOG_BOX_X_START:DIALOG_BOX_X_END,
        ]


# ---------------------------------------------------------------------------
# OCR backend (EasyOCR or Tesseract)
# ---------------------------------------------------------------------------

class ScreenOCR:
    """Thin wrapper around EasyOCR or Tesseract for GBA screen text."""

    def __init__(self):
        self._reader = None
        self._backend = "tesseract" if USE_TESSERACT else "easyocr"
        self._init_backend()

    def _init_backend(self) -> None:
        if USE_TESSERACT:
            try:
                import pytesseract
                self._pytesseract = pytesseract
                log.info("[TextAdvisor] Using Tesseract OCR backend")
            except ImportError:
                log.warning("[TextAdvisor] pytesseract not installed — OCR disabled")
                self._backend = "none"
        else:
            try:
                import easyocr
                # gpu=False is safer for multi-process envs; set True if you have CUDA
                self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                log.info("[TextAdvisor] Using EasyOCR backend ✓")
            except ImportError:
                log.warning("[TextAdvisor] easyocr not installed — OCR disabled. "
                            "Run: pip install easyocr")
                self._backend = "none"
            except Exception as e:
                # Catches BadZipFile, FileNotFoundError from corrupt/partial downloads.
                # Training continues safely; OCR features are simply disabled.
                log.warning(
                    f"[TextAdvisor] EasyOCR init failed ({type(e).__name__}: {e}). "
                    "OCR disabled — delete ~/.EasyOCR/model/ and restart to retry."
                )
                self._backend = "none"

    def read(self, image_rgb: np.ndarray) -> str:
        """
        Run OCR on image_rgb. Returns concatenated text, lowercase.
        Returns empty string if OCR is unavailable or fails.
        """
        if self._backend == "none":
            return ""

        # Upscale for better OCR accuracy on small pixel-font text
        h, w = image_rgb.shape[:2]
        upscaled = cv2.resize(image_rgb, (w * 3, h * 3), interpolation=cv2.INTER_NEAREST)
        gray     = cv2.cvtColor(upscaled, cv2.COLOR_RGB2GRAY)

        try:
            if self._backend == "easyocr":
                results = self._reader.readtext(gray, detail=0, paragraph=True)
                return " ".join(results).lower().strip()

            elif self._backend == "tesseract":
                text = self._pytesseract.image_to_string(
                    gray,
                    config="--psm 6 -c tessedit_char_whitelist="
                           "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 !?.',/-"
                )
                return text.lower().strip()

        except Exception as e:
            log.debug(f"[TextAdvisor] OCR error: {e}")
            return ""

        return ""


# ---------------------------------------------------------------------------
# Situation classifier (rule-based — Pokemon text is structured and finite)
# ---------------------------------------------------------------------------

class SituationClassifier:
    """
    Maps OCR text → Situation enum.

    Uses keyword matching. This covers ~95% of Pokemon FireRed text situations.
    No LLM needed — the game's menus are completely predictable.
    """

    # Keywords that identify each situation (checked in priority order)
    _RULES: list[tuple[Situation, list[str]]] = [
        (Situation.HEAL_CONFIRM, [
            "healed", "pokemon have been healed", "restored to full",
        ]),
        (Situation.LEVEL_UP, [
            "grew to level", "grew to lv", "level up",
        ]),
        (Situation.MOVE_SELECT, [
            "tackle", "growl", "scratch", "tail whip", "ember", "water gun",
            "vine whip", "pound", "leer", "thunder shock", "pp ",
            # generic: if we see 4 short words in a 2x2 grid it's moves
        ]),
        (Situation.BATTLE_MENU, [
            "fight", "bag", "pokemon", "run",
        ]),
        (Situation.ITEM_MENU, [
            "items", "key items", "poke balls", "tms & hms",
        ]),
        (Situation.YES_NO, [
            "yes", "no",   # only when both appear together
        ]),
        (Situation.NAME_ENTRY, [
            "nickname", "give a nickname", "name your",
        ]),
        (Situation.DIALOG, [
            # Catch-all for NPC text — any sentence-like text in dialog region
            # We use length heuristic: >10 chars of text = likely dialog
        ]),
    ]

    @classmethod
    def classify(cls, text: str) -> tuple[Situation, float]:
        """
        Returns (Situation, confidence).
        Confidence is 1.0 for exact keyword matches, 0.5 for heuristics.
        """
        if not text:
            return Situation.UNKNOWN, 0.0

        t = text.lower()

        for situation, keywords in cls._RULES:
            if situation == Situation.YES_NO:
                # Both "yes" AND "no" must appear
                if "yes" in t and "no" in t:
                    return situation, 0.9
                continue

            if situation == Situation.DIALOG:
                # Fallback: any substantial text in the dialog region
                if len(text) > 10:
                    return situation, 0.5
                continue

            for kw in keywords:
                if kw in t:
                    return situation, 1.0

        return Situation.UNKNOWN, 0.0


# ---------------------------------------------------------------------------
# Action name → index mapping (matches ActionSpace in pokemon_rl.py)
# ---------------------------------------------------------------------------

ACTION_NAME_TO_IDX = {"A": 0, "B": 1, "UP": 2, "DOWN": 3, "LEFT": 4, "RIGHT": 5}
ACTION_IDX_TO_NAME = {v: k for k, v in ACTION_NAME_TO_IDX.items()}


# ---------------------------------------------------------------------------
# TextAdvisor — main interface
# ---------------------------------------------------------------------------

class TextAdvisor:
    """
    Main interface. Plugs into PokemonEnv.

    Typical usage in pokemon_rl.py
    --------------------------------
    # __init__:
        self._advisor = TextAdvisor()

    # step() — pass the RAW (pre-resize) screen:
        reward += self._advisor.update(raw_obs, action_idx)

    # _build_obs() — append embedding to RAM vector:
        extra = self._advisor.get_embedding().to_array()
        ram   = np.concatenate([ram, extra])

    # Remember to update observation_space:
        "ram": spaces.Box(0.0, 1.0, (12 + SituationEmbedding.SIZE,), dtype=np.float32)
    """

    def __init__(self):
        self._ocr        = ScreenOCR()
        self._detector   = DialogDetector()
        self._classifier = SituationClassifier()

        self._embedding         = SituationEmbedding()
        self._last_text         = ""
        self._last_ocr_step     = -OCR_EVERY_N_STEPS   # force first run
        self._current_step      = 0
        self._steps_in_situation = 0
        self._last_situation    = Situation.UNKNOWN

        # Stats
        self.ocr_calls          = 0
        self.situation_counts   = {s: 0 for s in Situation}
        self.hints_matched      = 0
        self.total_bonus_given  = 0.0

    # ── main interface ────────────────────────────────────────────────────

    def update(self, raw_frame_rgb: np.ndarray, action_idx: int) -> float:
        """
        Call every env step with the raw GBA frame (before resize).
        Returns a shaped reward bonus (0.0 if no hint applies this step).
        """
        self._current_step += 1

        # Run OCR only every N steps, and only if dialog might be visible
        should_ocr = (self._current_step - self._last_ocr_step) >= OCR_EVERY_N_STEPS

        if should_ocr:
            self._run_ocr(raw_frame_rgb)

        # Update step counter for current situation
        situation = self._embedding.situation
        if situation == self._last_situation:
            self._steps_in_situation += 1
        else:
            self._steps_in_situation = 0
            self._last_situation = situation

        self._embedding.steps_in_situation = self._steps_in_situation
        self._embedding.last_ocr_age = min(
            (self._current_step - self._last_ocr_step) / OCR_EVERY_N_STEPS, 1.0
        )

        # Compute reward bonus
        return self._compute_bonus(situation, action_idx)

    def get_embedding(self) -> SituationEmbedding:
        """Returns the current situation embedding for the obs vector."""
        return self._embedding

    def reset(self) -> None:
        """Call on episode reset."""
        self._embedding          = SituationEmbedding()
        self._last_text          = ""
        self._last_ocr_step      = -OCR_EVERY_N_STEPS
        self._current_step       = 0
        self._steps_in_situation = 0
        self._last_situation     = Situation.UNKNOWN

    def stats_dict(self) -> dict:
        """Returns stats for TensorBoard logging."""
        return {
            "advisor/ocr_calls":        self.ocr_calls,
            "advisor/hints_matched":    self.hints_matched,
            "advisor/total_bonus":      self.total_bonus_given,
            "advisor/situation":        int(self._embedding.situation),
        }

    # ── internal ──────────────────────────────────────────────────────────

    def _run_ocr(self, frame_rgb: np.ndarray) -> None:
        """
        Run OCR on the dialog region every N steps unconditionally.

        The pixel-color DialogDetector heuristic was removed because the
        hardcoded border color (24,56,88) never matched real FireRed frames,
        causing is_dialog_visible() to always return False and blocking all
        OCR calls — which is why txt was always 0.00.

        Tesseract at ~0.08s/call with OCR_EVERY_N_STEPS=100 adds only ~4s
        overhead per 5000-step episode, so the pre-filter is unnecessary.
        When there is no text on screen Tesseract returns an empty string
        quickly, and the classifier returns EXPLORING with no bonus — correct.
        """
        self._last_ocr_step = self._current_step
        self.ocr_calls += 1

        # Always crop the bottom dialog region and run OCR
        crop = self._detector.crop_dialog_region(frame_rgb)
        text = self._ocr.read(crop)
        self._last_text = text
        situation, conf = self._classifier.classify(text)

        if situation not in (Situation.UNKNOWN, Situation.EXPLORING):
            log.debug(
                f"[TextAdvisor] OCR='{text[:60]}' → "
                f"{situation.name} (conf={conf:.2f})"
            )

        self.situation_counts[situation] += 1

        recommended_action = SITUATION_ACTION.get(situation, "B")
        action_idx = ACTION_NAME_TO_IDX.get(recommended_action, 1)

        self._embedding = SituationEmbedding(
            situation=situation,
            recommended_action_idx=action_idx,
            confidence=conf,
            steps_in_situation=self._steps_in_situation,
            last_ocr_age=0.0,   # freshly computed
        )

    def _compute_bonus(self, situation: Situation, action_idx: int) -> float:
        """
        Give a reward bonus if the agent pressed the recommended button.
        Only applies when confidence is high enough.
        """
        if situation == Situation.EXPLORING:
            return 0.0

        if self._embedding.confidence < 0.5:
            return 0.0

        recommended_idx = self._embedding.recommended_action_idx
        if action_idx != recommended_idx:
            return 0.0

        bonus = SITUATION_BONUS.get(situation, 0.0) * self._embedding.confidence
        if bonus > 0:
            self.hints_matched     += 1
            self.total_bonus_given += bonus
            log.debug(
                f"[TextAdvisor] Hint matched! "
                f"situation={situation.name}  "
                f"action={ACTION_IDX_TO_NAME.get(action_idx,'?')}  "
                f"bonus={bonus:.3f}"
            )

        return bonus


# ---------------------------------------------------------------------------
# Integration snippet (copy into pokemon_rl.py)
# ---------------------------------------------------------------------------

INTEGRATION_GUIDE = """
HOW TO INTEGRATE text_advisor.py INTO pokemon_rl.py
=====================================================

1. IMPORTS (top of pokemon_rl.py):
   from text_advisor import TextAdvisor, SituationEmbedding

2. Config — update RAM size:
   # observation_space ram shape becomes 12 + 5 = 17
   RAM_SIZE = 12 + SituationEmbedding.SIZE   # = 17

3. PokemonEnv.__init__:
   self._advisor = TextAdvisor()

   # Update observation space:
   self.observation_space = spaces.Dict({
       "image": spaces.Box(0, 255, (84, 84, 1), dtype=np.uint8),
       "ram":   spaces.Box(0.0, 1.0, (RAM_SIZE,), dtype=np.float32),
   })

4. PokemonEnv.reset():
   # After self._stats.reset(info) and self._rewards.reset():
   self._advisor.reset()

5. PokemonEnv.step() — BEFORE building obs, AFTER running action:
   # raw_obs is the max-pooled frame BEFORE grayscale/resize
   # You need to save the raw frame in _run_action and return it:
   text_bonus = self._advisor.update(raw_obs_rgb, action_idx)
   step_reward += self._rewards.add("text", text_bonus)

6. PokemonEnv._build_obs() — append embedding:
   extra = self._advisor.get_embedding().to_array()
   ram   = np.concatenate([existing_ram_array, extra])

7. RewardAccumulator — add text component:
   text: float = 0.0
   # Add "text" to reset() and to_dict():
   "reward_text": self.text,

8. PokemonCallback._record_rewards():
   self.logger.record("reward/from_text", m["reward_text"])

9. _run_action() — return raw frame alongside info:
   # Change return to include raw (pre-grayscale) frame:
   return raw_frame, np.max(np.stack(frames), axis=0), info
   # Then update all callers accordingly.

INSTALL:
   pip install easyocr
   # OR for tesseract:
   pip install pytesseract && apt install tesseract-ocr
"""

if __name__ == "__main__":
    print(INTEGRATION_GUIDE)