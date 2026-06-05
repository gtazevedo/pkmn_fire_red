import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from src.env import PokemonEnv
from src.config import CFG

# Pode receber um state file como argumento
target_state = sys.argv[1] if len(sys.argv) > 1 else None

class MockProgress:
    @property
    def current_state_file(self):
        return target_state if target_state else CFG.state_file

from src.reader import RamReader

env = PokemonEnv(env_id=99)
if target_state:
    env._progress = MockProgress()

obs = env.reset()

# Dá um step para atualizar a RAM interna do emulador com o state carregado
# Dá um step para atualizar a RAM interna do emulador com o state carregado
env.step(0)

# Pegando RAM via reader
info = {'_ram': env.env.em.get_state()}
bank = RamReader.map_bank(info)
map_id = RamReader.map_id(info)

print(f"Loaded State: {target_state if target_state else 'Random/Default'}")
print(f"Initial Map: bank={bank} id={map_id}")
env.close()
