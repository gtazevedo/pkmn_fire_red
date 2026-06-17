import sys
import os
import stable_retro

# Adiciona o caminho do projeto para importar modulos
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.env import RamReader

import glob

def main():
    state_files = glob.glob("./progress_states/*.state")
    if not state_files:
        print("No state files found!")
        return

    env = stable_retro.make(game="PokemonFireRed", use_restricted_actions=stable_retro.Actions.DISCRETE)
    env.reset()
    
    for state_file in sorted(state_files):
        with open(state_file, "rb") as f:
            env.em.set_state(f.read())
        
        # Executa um step zerado apenas para sincronizar a RAM
        result = env.step(0)
        info = result[-1]
        
        bank = RamReader.map_bank(info)
        mid = RamReader.map_id(info)
        x = RamReader.coord(info, "x")
        y = RamReader.coord(info, "y")
        flags = RamReader.event_flags_sum(env.get_ram())
        lvl = RamReader.party_level(info)
        
        loc = "Unknown"
        if bank == 3 and mid == 0:
            loc = "Pallet Town"
        elif bank == 3 and mid == 1:
            loc = "Viridian City"
        elif bank == 3 and mid == 2:
            loc = "Pewter City"
        elif bank == 4 and mid == 0:
            loc = "Player's House 1F"
        elif bank == 4 and mid == 1:
            loc = "Player's House 2F"
        elif bank == 4 and mid == 2:
            loc = "Oak's Lab"
        elif bank == 4 and mid == 3:
            loc = "Rival's House"
        
        print(f"[{flags} Flags] STATE: {os.path.basename(state_file)} | Map: {bank}-{mid} ({loc}) | X:{x} Y:{y} | Lvl:{lvl}")


if __name__ == "__main__":
    main()
