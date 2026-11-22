# Operação, Avaliação, Testes e MLflow

Guia de uso do projeto no dia a dia, comandos disponíveis, métricas acompanhadas e resolução de problemas.

---

## 1. Pré-requisitos e Variáveis de Ambiente

- **Python 3.12+** com dependências instaladas via Poetry ([pyproject.toml](../pyproject.toml)) no ambiente virtual `.venv/`.
- **ROM limpa:** `rom.gba` (Pokémon FireRed USA v1.0) na raiz e integração `PokemonFireRed` registrada no `stable-retro`.
- **Ollama local (para o Strategist LLM):** serviço ativo em `http://localhost:11434` com o modelo configurado (`CFG.ollama_model`, padrão `qwen2.5:3b`).

| Variável de Ambiente | Padrão | Finalidade |
| :--- | :--- | :--- |
| `PKMN_SAVE_DIR` | `/mnt/d/projects/pkmn_fire_red_saves` (ou `./saves`) | Diretório base no `D:` para modelo, `checkpoints/`, `progress_states/`, `steps_state.txt`, `pokemon_tensorboard/`, `pokemon_training.log`, `screenshots/`, `whiteout_frames/` e `mlflow/` (`mlflow.db` + `mlruns/`). |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Sobrescreve o modelo local utilizado pelo `Strategist`. |
| `MLFLOW_RUN_NAME` | Automático (`{stamp}_{git}_{model}`) | Define nome explícito para o run no MLflow. |
| `MLFLOW_RUN_NOTES` | `""` | Registra `mlflow.note.content` com o resumo das mudanças da bateria. |

---

## 2. Comandos Principais

### Iniciar ou Retomar Treinamento
```bash
PYTHONPATH=. .venv/bin/python -u -m scripts.train
# ou via Makefile (Poetry):
make train
```
- Encerra runs órfãos presos em `RUNNING` (`close_orphan_runs`).
- Retoma automaticamente `pokemon_lstm_model.zip` + `_vecnorm.pkl` quando o `.stamp` coincide com `CFG.config_stamp` (ou aplica cirurgia automática com backup antes de expandir a entrada da LSTM).
- Converte `SIGTERM`/`SIGINT` em `KeyboardInterrupt` limpa, salvando checkpoint, `_vecnorm.pkl`, artefatos e fechando o run do MLflow com status `FINISHED`.

### Subir o Servidor Ollama Local (modo silencioso, quando não estiver ativo)
```bash
OLLAMA_DEBUG=ERROR ~/.local/bin/ollama serve > /mnt/d/projects/pkmn_fire_red_saves/ollama.log 2>&1
# Verificar modelo disponível:
curl -sf http://localhost:11434/api/tags
```

### Avaliar Modelo Determinístico (`scripts/evaluate_mlflow.py`)
```bash
# Avaliação no currículo completo (ex.: 6 episódios de 1000 steps)
PYTHONPATH=. .venv/bin/python -m scripts.evaluate_mlflow --episodes 6 --steps 1000

# Avaliação estrita de retenção do início do jogo a partir de Start.state.gz
PYTHONPATH=. .venv/bin/python -m scripts.evaluate_mlflow --episodes 5 --steps 2000 --from-start
```
- Instancia `PokemonEnv(..., enable_stochasticity=False)` para medição reprodutível sem ruído de *sticky actions*, *coord dropout* ou micro-caminhada no reset.
- No modo `--from-start`, mede especificamente: `eval/left_bedroom_rate`, `eval/left_house_rate`, `eval/entered_oaks_lab_rate`, `eval/got_starter_rate`, `eval/beat_rival_rate` e `eval/reached_route1_rate`.

### Visualizar Agente Jogando em Tempo Real (`scripts/enjoy.py`)
```bash
PYTHONPATH=. .venv/bin/python scripts/enjoy.py
# ou: make enjoy
```
Abre uma janela Pygame `720×480` (upscale `3×` do GBA `240×160`) a 60 FPS no melhor marco salvo em `/mnt/d/projects/pkmn_fire_red_saves/progress_states/`.

### Consultar Resumo e Interface MLflow / TensorBoard
```bash
# Resumo rápido via terminal:
PYTHONPATH=. .venv/bin/python -m scripts.summary

# Interface web do MLflow (banco e artefatos no D:):
.venv/bin/mlflow ui --backend-store-uri sqlite:////mnt/d/projects/pkmn_fire_red_saves/mlflow/mlflow.db -h 127.0.0.1 -p 5000
# ou: make ui

# TensorBoard:
.venv/bin/tensorboard --logdir /mnt/d/projects/pkmn_fire_red_saves/pokemon_tensorboard/
```
- Experimento principal no MLflow UI: `http://127.0.0.1:5000/#/experiments/1`
- Model Registry (`PokemonFireRed-RecurrentPPO`): `http://127.0.0.1:5000/#/models/PokemonFireRed-RecurrentPPO`

### Rodar a Suíte de Testes Unitários (`56` testes)
```bash
PYTHONPATH=. .venv/bin/pytest -v
# ou: make test
```
Cobre:
- [tests/test_battle.py](../tests/test_battle.py): tabela de tipos Gen 3, `score_moves()`, economia (`get_money` XOR), bolso de Pokébolas, descriptografia de espécies Gen 3, `get_battle_observation` (`BATTLE_SIZE=12`), `gDisplayedStringBattle`, bloqueio de fuga em treinadores/selvagens, penalidade anti-status spam e recuperação de cursor do `Dynamic Battle Curriculum`.
- [tests/test_reader.py](../tests/test_reader.py): ponteiros `SaveBlock1/2`, flags, itens criptografados, `collision_grid 9×9`, portas sólidas vs árvores, `is_script_locked`, `is_scripted_tutorial`, `has_teachy_tv`, `is_oak_in_lab`.
- [tests/test_progress.py](../tests/test_progress.py): *Frontier Bias 50/30/20*, *Genesis Anchor*, monotonicidade de flags, `visited_grid`, piso estacionário, fadiga de batalhas, teto de nível, cura anti-farming e imunidade de `steps_on_cur_tile` durante `script_lock`.
- [tests/test_strategist.py](../tests/test_strategist.py): decodificação TBL do `RamAdvisor`, `YES/NO` contextual (incluindo Mart), *VisionEngine*, schema JSON Ollama, guardrails 3B, anti-border-bounce, Rotas 22–25, guaritas da Viridian Forest `(15, 0)/(15, 2)` e Lab vazio (`0x02B`).
- [tests/test_surgery.py](../tests/test_surgery.py): preservação exata de saídas e expansão zero-padded da LSTM (`366 → 447 → 459`).
- [tests/test_mlflow_tracking.py](../tests/test_mlflow_tracking.py): criação de metadados, promoção de `@champion`, `close_orphan_runs`, fluxo de `scripts/train.py` e consolidação dos paths sob `CFG.base_save_dir`.

---

## 3. Métricas e Governança MLflow (`src/callbacks.py`, `src/mlflow_utils.py`)

A cada 5 iterações (`~200k` steps) ou na parada limpa, o modelo zip e o `_vecnorm.pkl` são versionados no `REGISTERED_MODEL_NAME = "PokemonFireRed-RecurrentPPO"` recebendo alias `@current`; o alias `@champion` só é promovido quando `reward_mean20` supera o recorde atual registrado no Registry.

| Grupo de Métricas | Exemplos de Chaves | Significado |
| :--- | :--- | :--- |
| **`progress/`** | `score`, `best_milestone`, `party_level`, `total_tiles_ever`, `total_maps_ever`, `frontier_ratio`, `genesis_ratio`, `start_state_ratio` | Progresso global e saúde da distribuição de amostragem do currículo. |
| **`trend/`** | `reward_mean20`, `tiles_mean20`, `maps_mean20`, `win_rate_mean20`, `combat_net_mean20`, `expl_net_mean20`, `stuck_ratio_mean20` | Médias móveis dos últimos 20 episódios para detectar platôs ou regressões. |
| **`health/` & `train/`** | `health/ev_ma10`, `health/policy_entropy`, `train/approx_kl`, `train/clip_fraction`, `train/explained_variance`, `train/learning_rate` | Estabilidade da otimização PPO (`approx_kl` alvo ~`0.015–0.025`, `ev_ma10` alto, entropia não colapsada). |
| **`battle/` & `reward/`** | `battle/win_rate`, `battle/kill_efficiency`, `battle/whiteouts`, `reward/from_explore`, `reward/from_victory`, `reward/from_strategist` | Diagnóstico fino entre ganho de exploração vs fobia/eficiência em combate. |

---

## 4. Troubleshooting Comum

- **Ollama offline durante o treino:** o `Strategist` registra aviso em `debug`, aplica cooldown de `30s` e utiliza `_heuristic_goal()` localmente sem derrubar nem travar os workers. Se o servidor caiu, suba `ollama serve` e confirme `qwen2.5:3b` via `curl -sf http://localhost:11434/api/tags`.
- **`Save state` preso ou com HP crítico:** inspecione `/mnt/d/projects/pkmn_fire_red_saves/progress_states/*.state` verificando `cb2 == 0x080565B5`, `is_script_locked(ram) == False`, movimento livre fora de tiles `TILE_WARP`, `is_oak_in_lab(ram)` em `(4, 3)` e HP `>= 50%`. Backups de estados anteriores ficam em `/mnt/d/projects/pkmn_fire_red_saves/progress_states/backup_pre_*`.
- **Incompatibilidade de checkpoint (`config_stamp`):** alterar o texto de `CFG.config_stamp` sem alterar shapes ou sem caminho na cirurgia faz `Trainer` deletar e reiniciar do zero caso `upgrade_checkpoint_if_needed()` retorne `False`. Para mudanças apenas de recompensa/currículo que preservam o `observation_space`, mantenha o `config_stamp` compatível (ou adicione migração explícita em [src/surgery.py](../src/surgery.py)).
