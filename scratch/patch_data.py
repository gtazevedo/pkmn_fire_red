import json

data_path = "/home/guilh/projects/pkmn_fire_red/.venv/lib/python3.12/site-packages/stable_retro/data/stable/PokemonFireRed/data.json"
with open(data_path, "r") as f:
    data = json.load(f)

# gBattleStruct is at 0x02023FE8 = 33698792
data["info"]["gBattleStruct_ptr"] = { "address": 33698792, "type": "<u4" }

with open(data_path, "w") as f:
    json.dump(data, f, indent=2)

print("data.json patched with gBattleStruct_ptr!")
