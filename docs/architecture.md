# Arquitetura do Agente (HRL, PPO Recorrente, Cirurgia e Anti-Memorização)

Este documento descreve como as camadas de decisão funcionam em conjunto no código atual (`config_stamp = "v31_clean_states_visited9x9"`).

---

## 1. Arquitetura Hierárquica em Três Camadas

### Camada 1 — Planejador Estratégico (`Strategist`, [src/strategist.py](../src/strategist.py))

O `Strategist` gera um sub-objetivo macro (`StrategicGoal`) convertido em um embedding `goal` de `GOAL_SIZE = 6` floats injetado na observação do PPO:

- **LLM Local Assíncrono (Ollama):** Usa `CFG.ollama_url = "http://localhost:11434/api/generate"` com modelo padrão `qwen2.5:3b` (`OLLAMA_MODEL`), timeout de `15.0s`, schema JSON estrito (`OLLAMA_RESPONSE_SCHEMA`), `keep_alive="24h"`, `temperature=0.1` e `ThreadPoolExecutor(max_workers=1)`.
- **Não bloqueia o emulador:** No disparo de evento, aplica imediatamente a meta determinística `_heuristic_goal(context)` para latência zero e submete a consulta HTTP em background; se o Ollama falhar ou expirar, ativa cooldown de `30s`.
- **Gatilhos (*Event-Driven*):**
  1. Mudança de mapa `(map_bank, map_id)`.
  2. Aumento na contagem de *Event Flags*.
  3. Transição de entrada/saída de batalha (usa apenas heurística local, sem chamar HTTP do Ollama quando apenas `in_battle` muda no mesmo mapa).
  4. Mudança na posse da `Oak's Parcel`.
  5. Estagnação no mesmo mapa (`strategist_stagnation_steps = 500`).
- **Grafo de Kanto e Guardrails para LLM 3B:**
  - `KANTO_PROGRESSION_GRAPH` e `VALID_KANTO_MAPS` mapeiam a progressão de Kanto (com as **Rotas 21–25 corrigidas**: Route 21 dividida em `(3,39)`/`(3,40)`, Route 22 `(3,41)`, Route 23 `(3,42)`, Route 24 `(3,43)` e Route 25 `(3,44)`).
  - Na Route 22 `(3, 41)`, quando `badges < 8`, orienta retornar a leste para Viridian City `(3, 1)` (evitando que o agente fique preso na Route 22 antes dos 8 ginásios); apenas com `badges >= 8` segue a oeste rumo à Route 23 `(3, 42)` e Indigo Plateau `(3, 9)`.
  - `_query_and_apply()` rejeita respostas com `target_map_bank <= 0`, mapas fora de `VALID_KANTO_MAPS`, prioridade `"heal"` indevida quando `hp_pct >= 0.45`, ou desvios fora de batalha que contrariem o mapa-alvo válido do grafo.
  - **Lab vazio vs Prof. Oak presente:** usando `RamReader.is_oak_in_lab()` (`FLAG_HIDE_OAK_IN_HIS_LAB = 0x02B`), entrar no Lab `(4, 3)` com `party_level == 0` enquanto o Prof. Oak ainda não está lá não paga o bônus de alinhamento (`+15.0`), não marca `(4, 3)` como já visitado e orienta o agente a sair ao sul rumo ao mato norte de Pallet.

### Camada 2 — Política Neural Motor/Tática (`RecurrentPPO` + `MultiInputLstmPolicy`)

A rede neural recebe observações multimodais a cada step (`FRAMES_PER_STEP = 8` frames GBA) e escolhe uma dentre **7 ações discretas** (`ActionSpace` em [src/config.py](../src/config.py)):

| Índice | Ação (`ActionSpace.NAMES`) | Efeito no GBA / Ambiente |
| :---: | :--- | :--- |
| `0` | `A` | Botão A (confirmar, falar, atacar) |
| `1` | `B` | Botão B (cancelar, correr em tênis, sair de menus) |
| `2` | `UP` | Direcional Cima |
| `3` | `DOWN` | Direcional Baixo |
| `4` | `LEFT` | Direcional Esquerda |
| `5` | `RIGHT` | Direcional Direita |
| `6` | `TACTICAL_MACRO` | NO-OP no emulador; interceptado pelo ambiente para acionar o *Soldier* |

Durante o **Dynamic Battle Curriculum** (`enable_battle_curriculum=True` e `global_win_rate() < 0.20`), se `was_in_battle` estiver ativo, as teclas direcionais/Start/Select (`UP`, `DOWN`, `LEFT`, `RIGHT`, `START`, `SELECT`) são substituídas por `NO_OP`, focando a política em aprender a navegar o menu de ataque com `A`/`B`.

### Camada 3 — Braço Robótico (`Soldier` em [src/env.py](../src/env.py))

Quando o PPO seleciona `TACTICAL_MACRO`, o ambiente executa macros frame-a-frame de acordo com o contexto real da RAM:

- **Poção em batalha (`_run_potion_macro`):** só executa se `_in_battle_now and not _script_lock`, `potions > 0`, `0 < cur_hp < 0.45 * max_hp`, até `max_potion_macro_attempts_per_battle = 2` por batalha e com `potion_macro_cooldown_steps = 12`. Se o número de poções cair ou o HP subir, concede `+100.0`; caso contrário, aplica `wasted_macro_penalty = -0.25`.
- **Ensino de TM fora de batalha (`_run_tm_macro`):** só executa se `not _in_battle_now and not _script_lock`, `tms > 0` e `< max_tm_macro_attempts_per_episode (2)`. Se o número de TMs cair, concede `+500.0`; se falhar, `-0.25`.
- Qualquer acionamento de `TACTICAL_MACRO` fora dessas condições aplica imediatamente `-0.25` para eliminar spam.

---

## 2. Espaço de Observação (`PokemonEnv.observation_space`)

O ambiente retorna um `gymnasium.spaces.Dict` com 5 chaves (normalização `VecNormalize` aplicada apenas às recompensas: `norm_obs=False, norm_reward=True, clip_reward=10.0`):

```text
Dict({
  "image":          Box(0, 255, (84, 84, 1), uint8)     # Frame cinza reduzido com max-pooling temporal
  "ram":            Box(0.0, 1.0, (23,), float32)       # Estado interno normalizado + RamAdvisor + combate
  "collision_grid": Box(0.0, 1.0, (9, 9), float32)      # Mapa local 9x9 de colisão, mato, warps e NPCs
  "visited_grid":   Box(0.0, 1.0, (9, 9), float32)      # Memória espacial local 9x9 de visitas no episódio
  "goal":           Box(0.0, 1.0, (6,), float32)        # Sub-objetivo estratégico do Strategist
})
```

### Detalhamento de `"ram"` (`RAM_SIZE = 23`)

| Índices | Campo | Normalização | Origem |
| :---: | :--- | :--- | :--- |
| `0` | `player_x` (com *coord dropout* no treino) | `/ 255.0` | SaveBlock1 (`x`) |
| `1` | `player_y` (com *coord dropout* no treino) | `/ 255.0` | SaveBlock1 (`y`) |
| `2` | `map_id` | `/ 255.0` | `info` |
| `3` | `map_bank` | `/ 255.0` | `info` |
| `4` | `is_in_battle` | `{0.0, 1.0}` | `gMain_callback2` (`0x0801xxxx`) |
| `5` | `script_lock` | `{0.0, 1.0}` | IWRAM + `info` (antes do override no step) |
| `6` | `player_moving` | `{0.0, 1.0}` | `info` |
| `7` | `enemy_hp` (soma 6 slots) | `/ 800.0` | `gEnemyParty` |
| `8` | `party_hp` (soma 6 slots) | `/ 5000.0` | `gPlayerParty` |
| `9` | `party_level` (soma 6 slots) | `/ 600.0` | `gPlayerParty` |
| `10` | `badges` (contagem de bits `0..8`) | `/ 8.0` | SaveBlock1 `0x0FE4` |
| `11` | `episode_steps_progress` | `steps / max_steps` | `EpisodeStats` |
| `12` | `potions_count` ("Sexto Sentido") | `min(n, 99) / 99.0` | Bolso de Itens (XOR decrypt) |
| `13` | `tms_count` ("Sexto Sentido") | `min(n, 99) / 99.0` | Bolso de TM/HM (XOR decrypt) |
| `14` | `p1_status` ("Sexto Sentido") | `{0.0, 1.0}` | Byte `80` da struct do Pokémon 1 |
| `15` | `advisor_situation` | `sit / 7.0` (`1..7`) | `RamAdvisor` |
| `16` | `advisor_recommended_action` | `rec / 5.0` (`0`=`A`, `1`=`B`) | `RamAdvisor` |
| `17` | `advisor_confidence` | `{0.0, 1.0}` | `RamAdvisor` (`1.0` se != `SIT_EXPLORE`) |
| `18` | `advisor_steps_in_situation` | `min(steps, 200) / 200.0` | `RamAdvisor` |
| `19` | Reservado (`0.0`) | `0.0` | `RamAdvisor` |
| `20` | `battle_idle_ratio` | `min(idle, 100) / 100.0` | `EpisodeStats` |
| `21` | `battle_outcome_potential` | `(init_hp - hp) / init_hp` | `EpisodeStats` |
| `22` | `steps_in_battle_norm` | `min(steps, 500) / 500.0` | `EpisodeStats` |

### Detalhamento de `"goal"` (`GOAL_SIZE = 6`)

- `[0..3]`: *one-hot* de `PRIORITIES = ("explore", "heal", "battle_gym", "use_hm")`.
- `[4]`: `clamp(target_map_id, 0, 255) / 255.0`.
- `[5]`: `NAV_HINTS.index(navigation_hint) / 5.0`, onde `NAV_HINTS = ("none", "north", "south", "east", "west", "interact")`.

---

## 3. Cirurgia de Pesos (`src/surgery.py`)

Para evoluir o `observation_space` sem descartar dezenas de milhões de steps treinados, `Trainer._load_or_create_model()` em [src/callbacks.py](../src/callbacks.py) aciona automaticamente backup + `upgrade_checkpoint_if_needed()` sempre que o `config_stamp` salvo difere do atual e as chaves antigas são um subconjunto compatível das novas:

1. `compute_key_feature_ranges()` calcula o fatiamento exato do vetor concatenado que entra na LSTM do `CombinedExtractor` (ex.: na transição `v30 → v31` com a inclusão de `visited_grid 9×9`, a entrada da LSTM expandiu de **`366` → `447`** dimensões).
2. `_expand_lstm_input_tensor()` cria tensores novos inicializados em zero (preservando a resposta inicial exatamente igual ao checkpoint antigo) e copia coluna a coluna para `lstm_actor.weight_ih_l0` e `lstm_critic.weight_ih_l0`.
3. Os buffers de momento do otimizador Adam (`exp_avg` e `exp_avg_sq`), os buffers internos `_last_obs` / `_last_original_obs` e o arquivo `_vecnorm.pkl` são patchados com as mesmas dimensões.

---

## 4. Hiperparâmetros PPO, Frame Skip e Pacote Anti-Memorização

### Treinamento e Estabilidade PPO ([src/config.py](../src/config.py))

| Parâmetro | Valor Atual (`CFG`) | Observação |
| :--- | :--- | :--- |
| `num_cpu` | `6` | Workers paralelos em `SubprocVecEnv` |
| `learn_batch` | `40_000` | Rollout por iteração de treino (salva a cada batch; registra MLflow a cada 5 iters) |
| `n_steps` / `batch_size` / `n_epochs` | `1024` / `512` / `3` | Estabilização aplicada pós-Batch 9 (reduziu `approx_kl` de `0.054` para `~0.020`) |
| `gamma` / `gae_lambda` | `0.997` / `0.95` | Horizonte efetivo longo para navegação entre mapas |
| `ent_coef` / `target_kl` | `0.018` / `0.025` | Exploração sustentada com freio contra atualizações bruscas |
| `lr_start` → `lr_end` | `6e-5` → `3e-5` | Decaimento linear **global** sobre `lr_total_budget = 50_000_000` (`make_lr_schedule`) |
| Frame skip | `frames_held=6`, `frames_noop=2` | `8` frames GBA/step; max-pooling em `frames[4:8]` reduz *flicker* de sprites |

### Pacote Anti-Memorização (`enable_stochasticity=True` no treino, `False` na avaliação)

Para impedir que a rede recorrente memorize uma sequência rígida de frames/coordenadas a partir dos *save states* fixos (especialmente `Start.state.gz`), quatro técnicas atuam em conjunto sem quebrar a compatibilidade do checkpoint:

1. **Dropout de Coordenadas Absolutas (`coord_dropout_prob = 0.15`):** em 15% dos steps de treino, zera `px_norm` e `py_norm` no vetor `ram[0:2]` dentro de `_build_obs()`, mantendo `collision_grid`, `visited_grid`, `image` e `goal` intactos para forçar navegação por geometria local e memória visual.
2. **NO-OP *Jitter* no `reset()` (`reset_random_noop_max = 15`):** executa `0..15` frames NO-OP aleatórios após carregar o estado, alterando RNG do emulador e fase de movimento dos NPCs.
3. **Micro-Caminhada Segura no `reset()` (`reset_random_walk_max = 3`):** quando o episódio inicia em overworld destravado (`cb2 == 0x080565B5`), `_random_safe_walk()` dá `0..3` passos aleatórios escolhendo **apenas direções cujo tile adjacente seja `TILE_WALKABLE`** (nunca pisando em warps, NPCs ou grama se bloqueado), interrompendo imediatamente caso o mapa mude ou um script dispare.
4. **Sticky Actions no Overworld (`sticky_action_prob = 0.08`):** seguindo Machado et al. (2018), fora de batalha e fora de `script_lock`, há 8% de chance de repetir o movimento cardinal anterior (`UP/DOWN/LEFT/RIGHT`), quebrando rotas decoradas passo a passo.
