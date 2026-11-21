# Sistema de Recompensas, Proteções e Currículo Dinâmico

As regras de recompensa e amostragem ficam centralizadas em [src/config.py](../src/config.py) (`CFG`), [src/progress.py](../src/progress.py) (`ProgressManager`, `EpisodeStats`, `RewardAccumulator`) e [src/env.py](../src/env.py).

---

## 1. Arquitetura de Recompensas e Anti-Reward-Hacking

Todas as parcelas são acumuladas em `RewardAccumulator` para auditoria no MLflow/TensorBoard. O princípio é **manter a exploração e o avanço na história como fontes dominantes de retorno**, neutralizando exploits encontrados em versões anteriores (placas de texto, troca de porta em loop, cura com HP cheio e fobia de batalha).

### Exploração e Navegação

| Componente | Valor / Fórmula (`CFG`) | Proteção / Regra Anti-Exploit |
| :--- | :--- | :--- |
| **Tile novo** (`explore`) | `explore_weight = 1.5` (`+2.0` se `map_bank == 3`) × `novelty_factor` | `novelty_factor = 0.50 + 0.50 / (1 + log1p(min(n_persistent, 50)))`. O piso estacionário `0.50` evita que mapas iniciais zerem completamente o gradiente após milhares de episódios; limitado por mapa: `35.0` no interior (`bank >= 4`) e `2500.0` no exterior. |
| **Novo mapa** (`map_disc`) | `new_map_bonus = 5.0` (interior) / `15.0` (rotas `bank == 3`) | Pago 1× globalmente por política (`all_maps_visited`) e somente após `stats.steps >= 2` (evita bônus espúrio ao nascer na porta). Zera bounds direcionais na transição. |
| **Bússola direcional** (`milestone`) | `north_bonus_per_tile = 1.0` / tile novo no eixo | Segue `strategist.current_goal.navigation_hint` (`north` diminui `y_min`, `south` aumenta `y_max`, `east` aumenta `x_max`, `west` diminui `x_min`). |
| **Alinhamento Estratégico** (`strategist`) | `strategist_goal_bonus = 15.0` | Exige alcançar `(target_map_bank, target_map_id)` diferente do mapa de emissão, **não** estar em `_visited_maps_ep` (bloqueia oscilar na borda de dois mapas) e não estar em entrada de Lab vazio (`0x02B` ativa). |

### Progresso na História e Marcos

| Componente | Valor (`CFG`) | Regra de Pagamento |
| :--- | :--- | :--- |
| **Event Flags** (`milestone`) | `+40.0` × novas flags | Pago somente se `total_flags > max_event_sum` (monotônico no episódio; `max_event_sum` é semeado pelo maior valor entre RAM e nome do `.state`). |
| **Oak's Parcel** (`milestone`) | `+150.0` (pegar) / `+300.0` (entregar) | Ao pegar no Viridian Mart (`349`), limpa `explore_reward_per_map`, `tile_visits` e bounds para incentivar a viagem de volta ao Lab `(4, 3)`. Entrega exige voltar sem a Parcel no Lab `(4, 3)`. |
| **Teachy TV** (`milestone`) | `+100.0` (`teachy_tv_bonus`) | 1× por episódio ao concluir o tutorial de captura do Old Man em Viridian (recebendo key item `366`). |
| **Insígnias** (`badge`) | `badge_weight = 100.0` / insígnia | Monotônico via `last_badges`. |

### Combate, Evolução e Cura

| Componente | Valor (`CFG`) | Regras e Guardrails |
| :--- | :--- | :--- |
| **First Strike** (`damage`) | `first_strike_bonus = 6.0` × `fatigue` | Pago 1× por batalha no primeiro dano causado ao inimigo. |
| **Dano proporcional** (`damage`) | `(dmg / initial_enemy_hp) * 10.0 * fatigue` | Usa `%` de HP removido (`hp_pct_scale = 10.0`), evitando distorção por diferença de HP máximo. |
| **Near-KO / Vitória** (`victory`) | `near_ko_bonus = 5.0` (`<10%` HP); `victory_bonus = 20.0` × `fatigue` | Ao vencer (`enemy_hp == 0`), ativa `post_battle_grace = 200` steps antes de encerrar por horizonte máximo, dando tempo para receber XP/level-up. |
| **Fadiga de Batalhas** (`fatigue`) | Por mapa: `1.0 → 0.5 → 0.25 → 0.10` (com Parcel em rota: `0.5 → 0.2 → 0.05`) | Evita grind infinito sem zerar a motivação em encontros necessários. |
| **Level Up** (`levelup`) | `levelup_weight = 35.0` por nível até `level_cap` (`0.5x` se `in_battle and has_parcel`) | `level_cap = 12 + 8 * badges`. Sem `has_parcel`, paga valor cheio mesmo após múltiplas batalhas na rota para motivar ganho de nível rumo à Viridian Forest e Brock. |
| **Adversário Máximo** (`op_lvl`) | `max_op_level_reward_scale = 0.2` × `max_enemy_lvl_ever` | Incentiva enfrentar treinadores e áreas mais fortes. |
| **Cura no Centro Pokémon** (`heal`) | `heal_pct * heal_reward_scale` (`25.0`) | **Anti-farming estrito:** só paga se `last_party_hp / max_hp < 0.40` (`heal_min_prior_hp_ratio`), ganho `>= max(3, int(0.15 * max_hp))` e tamanho da party inalterado. |
| **Whiteout** (`stuck`) | `whiteout_penalty = -10.0` + `terminate_on_whiteout = True` | Penalidade moderada (evita fobia de batalhas) com cooldown pós-whiteout por tile; `wild_grind_penalty = 0.0`, `low_hp_battle_penalty = 0.0`. |

---

## 2. Early Stopping e Proteções de Episódio

Em vez de aplicar punições contínuas severas que distorcem a função de valor (lição v27b/v28 em [DESIGN_GUIDELINES.md](../DESIGN_GUIDELINES.md)), o ambiente usa encerramento limpo e rastreamento de travamentos:

- **`STUCK` (`stuck_threshold = 150`):** `steps_on_cur_tile` é incrementado em movimento livre, mas **zerado** sempre que `_script_lock=True` ou `_in_battle_now=True`. Ao atingir `150` steps parado em overworld destravado, aplica `stuck_penalty = -0.08` e encerra (`terminated = True`).
- **`STAIRS LOOP`:** se houver 3 transições consecutivas `A → B → A` exclusivamente entre interiores (`map_bank == 4`) com intervalo `< 150` steps entre transições, encerra o episódio.
- **`DIALOGUE FREEZE` (`dialogue_stuck_terminate = 450`):** encerra caso o buffer de texto EWRAM permaneça estático por `450` steps fora de batalha (suspenso enquanto `RamReader.is_scripted_tutorial(info)` for verdadeiro).
- **Anti-Farming de Batalha:** se `battle_steps_total >= 200`, `battle_ratio > farm_ratio_threshold` (sorteado uniformemente em `[0.80, 0.95]` por episódio) e `kill_eff < 0.003` com `party_level >= 15`, encerra com `farm_terminate = True`.

---

## 3. Currículo Dinâmico (`ProgressManager` — Frontier Bias 50/30/20 + Genesis Anchor)

Para evitar que o agente passe semanas preso em Pallet Town ou, inversamente, esqueça como jogar o início, o `ProgressManager` mantém um pool dinâmico em `progress_states/`:

```text
Pool ordenado por _progression_key(path) = (stage, parcel_sub, rank, flags + rank/100)
  stage 0: início (Start.state.gz, pallet_exterior.state, events <= 123 sem rank)
  stage 1: pré-entrega e viagem da Parcel (m01..m09)
  stage 2: pós-entrega / Pokédex em Pallet Lab (events_124.state)
  stage 3: exploração pós-Pokédex (rank >= 10 ou flags >= 125)
```

Quando há `>= 3` estados no pool, cada episódio sorteia o estado inicial conforme:

- **~50% Fronteira (`frontier_prob = 0.50`):** últimos `max(1, min(3, n // 3))` estados do ranking. Dentro deles, o estado de ponta (`frontier_states[-1]`) recebe **`frontier_top_state_share = 0.50`** (~25% do treino total) e os demais estados de fronteira dividem o restante.
- **~30% Intermediário (`intermediate_prob = 0.30`):** estados intermediários do pool para consolidar transições (Viridian Mart, retorno da Parcel, etc.).
- **~20% Inicial (`initial_prob = 0.20`) com Genesis Anchor:** dos `initial_states`, `genesis_anchor_share = 0.60` fica permanentemente ancorado nos primeiros `genesis_anchor_count = 2` estados (`Start.state.gz` e `pallet_exterior.state`), garantindo ~12% de episódios desde o zero independentemente de quantos marcos futuros sejam salvos.

### Critérios para Salvamento Orgânico (`is_clean_overworld_save_frame`)

Novos `.state` de evento (`check_and_save`) ou marco geográfico (`check_and_save_map_milestone` com `steps_in_current_map >= 2`) **só podem ser gravados no disco** quando todas as condições abaixo são verdadeiras:
- `gMain_callback2 == 0x080565B5` (overworld livre, fora de menu/cutscene/transição de mapa).
- `not _script_lock`, `not _in_battle_now`, `not _is_whiteout`.
- O centro `grid[4, 4]` não é um tile `TILE_WARP`.
- `party_level > 0` e `party_hp >= max(1, ceil(party_max_hp * 0.50))` (evita salvar estados moribundos que causam espiral de whiteout).

Entre os 6 workers paralelos, `ProgressManager.sample_state()` chama `_load_best_from_disk(quiet=True)` e o `PokemonCallback` sincroniza `steps_state.txt` para atualizar o horizonte de episódio entre fases (`phase1 < 500k -> 5k`, `phase2 < 2M -> 10k`, `phase3 -> 8k`).
