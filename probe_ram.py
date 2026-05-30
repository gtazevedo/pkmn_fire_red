import sys
import os
import struct

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from src.env import PokemonEnv

def probe_ram():
    env = PokemonEnv(env_id=0)
    env.reset()
    
    print("Player X address in retro:", env.env.data.get_variable("player_x"))
    
    # In stable-retro, variables are defined in data.json.
    # Player X address is 50352136.
    # 50352136 in hex is 0x3005008
    # Wait, in data.json, player_x is 50352136. Hex of 50352136 is 0x3005008!
    # Let me do the math in the script to find how it maps.
    addr = 50352136
    print(f"player_x raw address from data.json: {addr} (hex: {hex(addr)})")
    
if __name__ == "__main__":
    probe_ram()
