#!/usr/bin/env python3
import sys, os
sys.path.insert(0, '/home/guilh/projects/pkmn_fire_red')
import stable_retro as retro
import random
import numpy as np
from src.reader import RamReader
from src.config import CFG

ACTIONS = [
    np.array([0,0,0,0,0,1,0,0,0,0,0,0]), # DOWN
    np.array([0,0,0,0,1,0,0,0,0,0,0,0]), # UP
    np.array([0,0,0,0,0,0,1,0,0,0,0,0]), # LEFT
    np.array([0,0,0,0,0,0,0,1,0,0,0,0]), # RIGHT
]

NO_OP = np.array([0]*12)

def step_btn(env, btn, held=6, noop=4):
    info = {}
    for _ in range(held):
        _, _, _, _, info = env.step(btn)
    for _ in range(noop):
        _, _, _, _, info = env.step(NO_OP)
    return info

def get_pos(env, info):
    ram  = env.get_ram()
    return RamReader.map_bank(info), RamReader.map_id(info), RamReader.coord(info, 'player_x', ram), RamReader.coord(info, 'player_y', ram)

def main():
    env = retro.make(game='PokemonFireRed', state=None)
    env.reset()
    with open(CFG.state_file, 'rb') as f:
        env.em.set_state(f.read())

    info = step_btn(env, NO_OP, 30, 0)
    
    # Quarto -> Sala
    for btn in [ACTIONS[3], ACTIONS[3], ACTIONS[1], ACTIONS[1], ACTIONS[1], ACTIONS[1], ACTIONS[3], ACTIONS[3], ACTIONS[1]]:
        info = step_btn(env, btn, 12, 4)
        if get_pos(env, info)[1] == 1:
            break
            
    bank, mid, x, y = get_pos(env, info)
    print(f"Sala alcançada: x={x} y={y}")

    # Random walk until exit
    for step in range(5000):
        # Biased towards DOWN and LEFT
        btn = random.choices(ACTIONS, weights=[0.4, 0.1, 0.4, 0.1])[0]
        info = step_btn(env, btn, 8, 2)
        bank, mid, x, y = get_pos(env, info)
        if bank == 3:
            break
            
    if bank == 3:
        save_path = CFG.pallet_exterior_state_file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(env.em.get_state())
        print(f"✅ Salvo: {save_path} (bank={bank} mid={mid} x={x} y={y})")
    else:
        print(f"❌ Falhou (bank={bank} mid={mid} x={x} y={y})")
    env.close()

if __name__ == '__main__':
    main()
