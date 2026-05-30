import sys
import os
import numpy as np

# Adjust path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from src.env import PokemonEnv

def dump_flags():
    print("Loading PokemonFireRed via PokemonEnv...")
    env = PokemonEnv(env_id=0)
    
def dump_flags():
    env = PokemonEnv(env_id=0)
    env.reset()
    
    ram = env.env.get_ram()
    
    # We know that data.json maps address 50352136 (0x3005008) to player_x via pointer
    # GBA IRAM (Internal RAM) is usually located at 0x03000000.
    # Let's see if we can find the pointer at 0x3005008 in the ram array.
    # WRAM = 0x02000000 (256 KB)
    # IRAM = 0x03000000 (32 KB)
    
    # Let's search the ram array for a known value. We know player_x from the env:
    px = env.env.data.get_variable("player_x")
    print(f"Player X via retro API: {px}")
    
    # Let's read the 4 bytes at offset (0x3005008 - 0x3000000) in IRAM.
    # Assuming ram_array is WRAM (256KB) + IRAM (32KB) -> IRAM starts at 262144
    iram_start = 262144
    ptr_offset_in_iram = 0x5008
    ptr_idx = iram_start + ptr_offset_in_iram
    
    b1, b2, b3, b4 = int(ram[ptr_idx]), int(ram[ptr_idx+1]), int(ram[ptr_idx+2]), int(ram[ptr_idx+3])
    ptr_val = b1 | (b2 << 8) | (b3 << 16) | (b4 << 24)
    print(f"Pointer at 0x03005008: {hex(ptr_val)}")
    
    # The pointer should point to WRAM (e.g. 0x02025eb4)
    if (ptr_val >> 24) == 0x02:
        wram_offset = ptr_val & 0x00FFFFFF
        print(f"Pointer points to WRAM offset {hex(wram_offset)}")
        # Let's read player_x from there!
        wram_px = ram[wram_offset] | (ram[wram_offset+1] << 8)
        print(f"Player X read manually from WRAM: {wram_px}")
        
        # Now, where are the flags?
        # In pret/pokefirered, SaveBlock1 contains:
        # pos (4 bytes)
        # warpPos (4 bytes)
        # ... it's a huge struct. 
        # But we can just dump the first 2048 bytes of SaveBlock1 and compare Start vs Pallet Exterior!
        
        flags_start_offset = 0x0EE0
        flags_end_offset = 0x1000
        flags_block = ram[wram_offset + flags_start_offset : wram_offset + flags_end_offset]
        return list(flags_block)

def compare():
    flags_start = dump_flags()
    
    # Load pallet exterior
    env = PokemonEnv(env_id=0)
    state_file = os.path.join(os.path.dirname(__file__), 'progress_states', 'pallet_exterior.state')
    with open(state_file, "rb") as f:
        env.env.em.set_state(f.read())
        
    print("\n--- After loading Pallet Exterior ---")
    
    ram = env.env.get_ram()
    iram_start = 262144
    ptr_idx = iram_start + 0x5008
    b1, b2, b3, b4 = int(ram[ptr_idx]), int(ram[ptr_idx+1]), int(ram[ptr_idx+2]), int(ram[ptr_idx+3])
    ptr_val = b1 | (b2 << 8) | (b3 << 16) | (b4 << 24)
    wram_offset = ptr_val & 0x00FFFFFF
    flags_pallet = list(ram[wram_offset + 0x0EE0 : wram_offset + 0x1000])
    
    diffs = 0
    for i in range(len(flags_start)):
        if flags_start[i] != flags_pallet[i]:
            diffs += 1
            print(f"Diff at offset {hex(i)} of Event Flags (0xEE0): {flags_start[i]} -> {flags_pallet[i]}")
            
    print(f"Total diffs: {diffs}")

if __name__ == "__main__":
    compare()
