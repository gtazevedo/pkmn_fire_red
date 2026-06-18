import numpy as np

class RamAdvisor:
    """
    Advisor baseado em flags de RAM e decodificação do buffer de diálogo de EWRAM (0x02021D18).
    Upgrade robusto para leitura direta de texto e classificação avançada de menus.
    """
    # [FIX v25] Zerado para evitar o exploit do agente farmando placas de diálogo
    DIALOG_BONUS      = 0.0
    YES_NO_BONUS      = 0.25   # Recompensar B em prompts de YES/NO (evita loops infinitos de diálogo/saves)
    LEVEL_UP_BONUS    = 0.10   # Recompensar A em telas de Level Up
    HEAL_BONUS        = 0.20   # Recompensar A em telas de cura no Centro Pokémon
    MOVE_SELECT_BONUS = 0.20   # Recompensar A em telas de seleção de golpes
    BATTLE_BONUS      = 0.0    # anti-farm: zero por-press

    A_IDX = 0
    B_IDX = 1

    # Situations mapping
    SIT_EXPLORE     = 1
    SIT_DIALOG      = 2
    SIT_BATTLE      = 3
    SIT_YES_NO      = 4
    SIT_LEVEL_UP    = 5
    SIT_HEAL        = 6
    SIT_MOVE_SELECT = 7

    # Pokémon Gen 3 character decoding table (Hex to ASCII)
    CHAR_MAP = {}
    for code in range(0xBB, 0xD4 + 1):
        CHAR_MAP[code] = chr(ord('A') + (code - 0xBB))
    for code in range(0xD5, 0xEE + 1):
        CHAR_MAP[code] = chr(ord('a') + (code - 0xD5))
    for code in range(0xA1, 0xAA + 1):
        CHAR_MAP[code] = chr(ord('0') + (code - 0xA1))

    CHAR_MAP[0x00] = ' '
    CHAR_MAP[0x1B] = 'e'  # Mapeia POKéMON é
    CHAR_MAP[0xAB] = '!'
    CHAR_MAP[0xAC] = '?'
    CHAR_MAP[0xAD] = '.'
    CHAR_MAP[0xAE] = '-'
    CHAR_MAP[0xB8] = ','
    CHAR_MAP[0xB0] = '..'
    CHAR_MAP[0xB1] = '"'
    CHAR_MAP[0xB2] = '"'
    CHAR_MAP[0xB3] = "'"
    CHAR_MAP[0xB4] = "'"
    CHAR_MAP[0xFE] = '\n'  # Quebra de linha

    def __init__(self):
        self.steps_in_situation  = 0
        self._last_situation     = self.SIT_EXPLORE
        self._current_situation  = self.SIT_EXPLORE
        self.total_dialog_steps  = 0
        self.total_battle_steps  = 0
        self.total_yes_no_steps  = 0
        self.total_level_up_steps = 0
        self.total_heal_steps    = 0
        self.total_move_select_steps = 0
        self.total_bonus         = 0.0
        self.hints_matched       = 0
        self._last_decoded_text  = ""

    @classmethod
    def decode_ram_text(cls, ram_slice) -> str:
        chars = []
        for b in ram_slice:
            if b == 0xFF:  # Terminator
                break
            if b in cls.CHAR_MAP:
                chars.append(cls.CHAR_MAP[b])
            elif b == 0x00:
                chars.append(' ')
            else:
                pass
        return "".join(chars).lower()

    def update(self, in_battle: bool, script_lock: bool, action_idx: int, ram_array: np.ndarray = None) -> float:
        situation = self.SIT_EXPLORE
        decoded_text = ""

        text_changed = False
        # Só lê o EWRAM de texto se o jogador estiver travado numa animação/script de NPC ou na batalha.
        # Isso previne o "text farming exploit", onde o agente ficava apertando "A" para o vento lendo o lixo na RAM.
        if (script_lock or in_battle) and ram_array is not None and len(ram_array) > 138520 + 256:
            # Lê o buffer de diálogo de EWRAM na posição 0x02021D18 (índice 138520)
            dialogue_bytes = ram_array[138520:138520+256]
            decoded_text = self.decode_ram_text(dialogue_bytes)
            text_changed = (decoded_text != self._last_decoded_text)
            self._last_decoded_text = decoded_text

        # Classifica a situação usando regras de texto se o texto existir
        if decoded_text.strip():
            # YES/NO Prompts
            if ("yes" in decoded_text and "no" in decoded_text) or "would you like" in decoded_text or "do you want to" in decoded_text or "yes/no" in decoded_text:
                situation = self.SIT_YES_NO
                self.total_yes_no_steps += 1
            # Fight Move Selection Screen
            elif "pp " in decoded_text or "pp  " in decoded_text:
                situation = self.SIT_MOVE_SELECT
                self.total_move_select_steps += 1
            # Level Up
            elif "grew to level" in decoded_text or "grew to lv" in decoded_text:
                situation = self.SIT_LEVEL_UP
                self.total_level_up_steps += 1
            # Heal Pokemon
            elif "healed" in decoded_text or "restored to full" in decoded_text:
                situation = self.SIT_HEAL
                self.total_heal_steps += 1
            # Fallback a DIALOG ou BATTLE dependendo de flags se há texto geral
            elif in_battle:
                situation = self.SIT_BATTLE
                self.total_battle_steps += 1
            else:
                situation = self.SIT_DIALOG
                self.total_dialog_steps += 1
        else:
            # Caso contrário, usa as heurísticas de flags anteriores
            if in_battle:
                situation = self.SIT_BATTLE
                self.total_battle_steps += 1
            elif script_lock:
                situation = self.SIT_DIALOG
                self.total_dialog_steps += 1

        if situation == self._last_situation:
            self.steps_in_situation += 1
        else:
            self.steps_in_situation = 0
        self._last_situation    = situation
        self._current_situation = situation

        return self._compute_bonus(situation, action_idx, text_changed)

    def get_embedding(self) -> np.ndarray:
        sit = self._current_situation
        
        # Define a ação recomendada:
        # A para DIALOG, BATTLE, LEVEL_UP, HEAL, MOVE_SELECT (avançar ou agir)
        # B para YES_NO (negar padrão para evitar travamentos em loops/nicknames)
        if sit == self.SIT_YES_NO:
            rec = self.B_IDX
        elif sit in (self.SIT_DIALOG, self.SIT_BATTLE, self.SIT_LEVEL_UP, self.SIT_HEAL, self.SIT_MOVE_SELECT):
            rec = self.A_IDX
        else:
            rec = self.B_IDX

        conf = 1.0 if sit != self.SIT_EXPLORE else 0.0
        return np.array([
            sit / 7.0,  # Normalizado entre as 7 situações
            rec / 5.0,  # Normalizado entre as ações (0 a 5)
            conf,
            min(self.steps_in_situation, 200) / 200.0,
            0.0,
        ], dtype=np.float32)

    def reset(self) -> None:
        self.steps_in_situation = 0
        self._last_situation    = self.SIT_EXPLORE
        self._current_situation = self.SIT_EXPLORE
        self.total_dialog_steps = 0
        self.total_battle_steps = 0
        self.total_yes_no_steps = 0
        self.total_level_up_steps = 0
        self.total_heal_steps    = 0
        self.total_move_select_steps = 0
        self.hints_matched      = 0
        self.total_bonus        = 0.0
        self._last_decoded_text  = ""

    def stats_dict(self) -> dict:
        return {
            "advisor/dialog_steps":   self.total_dialog_steps,
            "advisor/battle_steps":   self.total_battle_steps,
            "advisor/yes_no_steps":   self.total_yes_no_steps,
            "advisor/level_up_steps": self.total_level_up_steps,
            "advisor/heal_steps":     self.total_heal_steps,
            "advisor/move_select_steps": self.total_move_select_steps,
            "advisor/hints_matched":  self.hints_matched,
            "advisor/total_bonus":    self.total_bonus,
        }

    MAX_TEXT_BONUS = 50.0  # Limite máximo de bônus ganho por textos por episódio

    def _compute_bonus(self, situation: int, action_idx: int, text_changed: bool) -> float:
        if situation == self.SIT_YES_NO:
            rec_action = self.B_IDX
            bonus = self.YES_NO_BONUS
        elif situation == self.SIT_LEVEL_UP:
            rec_action = self.A_IDX
            bonus = self.LEVEL_UP_BONUS
        elif situation == self.SIT_HEAL:
            rec_action = self.A_IDX
            bonus = self.HEAL_BONUS
        elif situation == self.SIT_DIALOG:
            rec_action = self.A_IDX
            # [FIX v26] Usa DIALOG_BONUS em vez de 0.5 hardcoded (era o exploit da placa)
            bonus = self.DIALOG_BONUS if text_changed else 0.0
        elif situation == self.SIT_BATTLE:
            rec_action = self.A_IDX
            bonus = self.BATTLE_BONUS
        elif situation == self.SIT_MOVE_SELECT:
            rec_action = self.A_IDX
            bonus = self.MOVE_SELECT_BONUS
        else:
            return 0.0

        if action_idx == rec_action and bonus > 0.0:
            if self.total_bonus + bonus > self.MAX_TEXT_BONUS:
                bonus = max(0.0, self.MAX_TEXT_BONUS - self.total_bonus)
            if bonus > 0:
                self.hints_matched += 1
                self.total_bonus   += bonus
            return bonus
        return 0.0
