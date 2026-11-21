# Leitura Direta de RAM, Grades Locais e Percepção

Uma premissa central do projeto é **Zero OCR**: extrair texto e estado diretamente dos barramentos de memória do Game Boy Advance (IWRAM + EWRAM) em microssegundos, evitando latência de OCR tradicional e garantindo **100% de fidelidade** na leitura de coordenadas, menus e progresso.

---

## 1. Mapa de Endereços RAM — Pokémon FireRed (USA v1.0)

Implementado em [src/reader.py](../src/reader.py) (`RamReader`) e complementado pelas variáveis do emulador em `data.json`:

### Ponteiros de IWRAM (`_IRAM_START = 262144` / `0x03000000`)

| Endereço GBA | Símbolo / Uso | Leitura no Projeto |
| :--- | :--- | :--- |
| `0x03005008` | `gSaveBlock1Ptr` | Ponteiro dinâmico (ASLR em EWRAM) para `SaveBlock1` (`ptr & 0x00FFFFFF` após checagem `(ptr >> 24) == 0x02`). |
| `0x0300500C` | `gSaveBlock2Ptr` | Ponteiro dinâmico para `SaveBlock2`, usado para obter a chave de criptografia XOR da mochila. |
| `0x03005040` | `gBackupMapLayout` | Cabeçalho `(width: s32, height: s32, map_ptr: u32)` do mapa carregado, base da `collision_grid 9×9`. |
| `0x03000EB0` | `gScriptContext1.mode` | Byte diferente de `0` indica script de evento/diálogo em andamento. |
| `0x03000F9C` | `sLockFieldControls` | Byte diferente de `0` indica controles de campo travados (`RamReader.is_script_locked()`). |

### Callbacks de Loop Principal (`gMain_callback2`)

| Valor / Faixa | Estado Detectado | Método / Uso |
| :--- | :--- | :--- |
| `0x080565B5` (`134571445`) | Overworld normal (mapa jogável) | Usado em `is_clean_overworld_save_frame`, detecção de `_not_in_overworld` e destravamento pós-warp. |
| `0x08010000` – `0x08017FFF` | Engine de Batalha (`CB2_InitBattle` / `CB2_BattleMain`) | `RamReader.is_in_battle(info)`. |
| `0x08100000` – `0x0810FFFF` | Tutoriais / cutscenes scriptadas (ex.: captura de Weedle do Old Man `0x08107F11`/`0x08107EE1`) | `RamReader.is_scripted_tutorial(info)`: suspende o contador de `DIALOGUE FREEZE` durante a animação. |
| `0x0805674D` | Placa/diálogo aberto logo após `set_state()` | Loop pós-reset em `PokemonEnv.reset()` pressiona `B` até liberar o callback. |

### EWRAM e Offsets Relativos a `SaveBlock1` / `SaveBlock2`

| Região | Offset | Significado | Detalhes |
| :--- | :--- | :--- | :--- |
| **EWRAM** | `0x024284` | `gPlayerParty` (6×100 bytes) | Byte `84` = nível; `u16@86` = HP atual; `u16@88` = HP máx; `u32@80` = status negativo (`Poison`, etc.). |
| **EWRAM** | `0x02402C` | `gEnemyParty` (6×100 bytes) | Lê nível, HP atual e HP máx de cada slot inimigo (validando `0 < maxhp <= 800`). |
| **EWRAM** | `0x036E38` | `gObjectEvents` (16×36 bytes) | `flags@0`, `curr_x@16 - 7`, `curr_y@18 - 7`. Ignora slot 0 / flag `0x10000` (player) e invisíveis (`0x2000`). |
| **EWRAM** | `0x02021D18` (idx `138520`) | Buffer de diálogo (256 bytes) | Texto codificado na tabela Gen 3 TBL lido pelo `RamAdvisor`. |
| **SaveBlock1** | `+0x0000`, `+0x0002` | `(player_x, player_y)` (`s16`) | Corrige leituras espúrias (`21884`) vindas de ponteiros estáticos do `data.json`. |
| **SaveBlock1** | `+0x03B8` | `bagPocket_KeyItems` (30×4 bytes) | `349` = Oak's Parcel (`has_oaks_parcel`); `366` (`ITEM_TEACHY_TV`) = Teachy TV (`has_teachy_tv`). |
| **SaveBlock1** | `+0x0310` / `+0x02B8` | `bagPocket_Items` (42×4 bytes) | Quantidade descriptografada com `enc_key` (`SaveBlock2 + 0x0F20`); filtra IDs `13..17` (Poções). |
| **SaveBlock1** | `+0x0464` / `+0x040C` | `bagPocket_TMHM` (58×4 bytes) | Contagem de TMs/HMs na mochila usando `enc_key`. |
| **SaveBlock1** | `+0x0EE0`..`+0x1000` | *Event Flags* (288 bytes = `2.304` bits) | `event_flags_sum()` soma todos os bits ativos via `byte.bit_count()`. |
| **SaveBlock1** | `+0x0FE4` | Insígnias (`0x820 >> 3`) | Número de bits ligados de `0..8`. |
| **SaveBlock1** | `+0x0FE5` (bit `0x02`) | `FLAG_SYS_POKEDEX_GET (0x829)` | Indica obtenção da Pokédex (`has_pokedex()`). |
| **SaveBlock1** | `+0x0EE5` (bit `0x08`) | `FLAG_HIDE_OAK_IN_HIS_LAB (0x02B)` | Ligada = Prof. Oak fora do Lab; desligada = Prof. Oak no Lab (`is_oak_in_lab()`). |

---

## 2. Grades Espaciais Locais `9×9`

Centradas no jogador (posição `[4, 4]` representa o tile atual):

### `collision_grid` ([src/reader.py](../src/reader.py#L397-L470))

Calculada a partir do mapa em `gBackupMapLayout` + margem de borda (`_MAP_BORDER_OFFSET = 7`) + `get_active_npcs()`:

| Valor Bruto | Normalizado (`/ 4.0`) | Constante | Regra de Classificação |
| :---: | :---: | :--- | :--- |
| `0` | `0.00` | `TILE_BLOCKED` | Fora dos limites, `metatile_id == 0x03FF`, ou `collision != 0` que **não** seja porta sólida nem NPC. |
| `1` | `0.25` | `TILE_WALKABLE` | `collision == 0` e fora de `WARP_METATILES`/`GRASS_METATILES`. |
| `2` | `0.50` | `TILE_GRASS` | `collision == 0` e `metatile_id in GRASS_METATILES` (`0x008..0x016`). |
| `3` | `0.75` | `TILE_WARP` | `metatile_id in SOLID_DOOR_METATILES` (`{0x062, 0x2A3, 0x2AC}`) **OU** `(collision == 0 and metatile_id in WARP_METATILES)` (ex.: escada `0x007`, tapetes `0x012..0x03A` e saída do Oak's Lab `0x290..0x292`). |
| `4` | `1.00` | `TILE_NPC` | `(tx, ty)` ocupado por NPC ativo e visível de `gObjectEvents`. |

> **Por que exigir `collision == 0` para `WARP_METATILES`?** No tileset de Pallet/Viridian, árvores, cercas e água reutilizam IDs numéricos presentes em interiores (`0x01C`, `0x060..0x06A`, etc.), porém com `collision != 0`. Exigir `collision == 0` (exceto para as portas sólidas `SOLID_DOOR_METATILES`) evita que árvores e paredes apareçam falsamente como portas na rede neural.

### `visited_grid` ([src/progress.py](../src/progress.py#L596-L613))

Para cada célula `(tx, ty)` no quadrado `9×9` ao redor do jogador no mapa `(bank, mid)`:

$$\text{visited\_grid}[r, c] = \frac{\min(\text{tile\_visits}[(bank, mid, tx, ty)], 5)}{5.0}$$

Fornece à política memória imediata dos tiles já explorados no episódio atual, reduzindo vai-e-vem cego dentro de mapas.

---

## 3. `RamAdvisor`: Decodificação TBL, Diálogos e `YES/NO` Contextual ([src/advisor.py](../src/advisor.py))

1. **Decodificação Gen 3 (`decode_ram_text`):** converte bytes de `ram[138520 : 138520 + 256]` até o terminador `0xFF` (`0xBB..0xD4` → `A..Z`, `0xD5..0xEE` → `a..z`, `0xA1..0xAA` → `0..9`, `0x1B` → `é` mapeado para `e`). Só é lido quando `script_lock or in_battle` é verdadeiro (impede leitura de lixo de RAM como exploração).
2. **Classificação de 7 Situações:**
   - `SIT_EXPLORE (1)`, `SIT_DIALOG (2)`, `SIT_BATTLE (3)`, `SIT_YES_NO (4)`, `SIT_LEVEL_UP (5)`, `SIT_HEAL (6)`, `SIT_MOVE_SELECT (7)`.
3. **`YES/NO` Contextual (`SIT_YES_NO`):**
   - `RamAdvisor.update(...)` recebe `party_level`, `map_key` e `party_hp_pct`.
   - Recomenda **`A` (`A_IDX`)** quando `party_level == 0 and map_key in ((-1, -1), (4, 3))` (escolher o Pokémon inicial no Laboratório) **OU** quando `party_hp_pct < 0.80` em prompt de cura / Centro Pokémon (`map_key in {(5, 4), (6, 1)}` ou palavras-chave `"rest", "tired", "heal", "restore", "health"`).
   - Recomenda **`B` (`B_IDX`)** nos demais casos (evitando loops em perguntas de *nickname* após o time já ter nível `> 0`, PC, salvar jogo ou Centro Pokémon com HP cheio).
4. **Anti-Exploit de Texto e Anti-Congelamento:**
   - `DIALOG_BONUS = 0.0` (eliminou *farming* de placas).
   - Avançar fala inédita fora de batalha pressionando `A` ou `B`: `+0.15` (`dialogue_advance_bonus`); repetir texto já visto na sessão: `-0.001` (`dialog_repeat_penalty`). Limitado a `MAX_TEXT_BONUS = 50.0` por episódio.
   - `dialogue_stuck_steps` soma quando `script_lock and not in_battle` e o buffer de texto não muda; se não pressionar `A`/`B` após `25` steps (`dialogue_stuck_grace`), aplica `-0.02`/step; aos `450` steps (`dialogue_stuck_terminate`), encerra o episódio (`DIALOGUE FREEZE`), exceto quando `RamReader.is_scripted_tutorial(info)` zera o contador (tutorial de captura do Old Man).

---

## 4. `VisionEngine` ([src/vision.py](../src/vision.py))

Pipeline auxiliar em OpenCV sobre o frame RGB `240×160` que complementa a leitura de RAM:

- **`WHITEOUT`:** média de luminância `< 15` (transições de tela preta ou derrota total).
- **`BATTLE`:** verifica simultaneamente as regiões das caixas de HP inimiga (`gray[16:45, 12:100]`) e do jogador (`gray[88:115, 136:228]`) com média `> 200` e presença de texto escuro (`< 50`).
- **`DIALOGUE`:** avalia a caixa inferior (`gray[112:152, 8:232]`), exigindo `mean > 220` ou (`mean > 175` com `bright_ratio > 0.65` e `dark_ratio > 0.02`, filtro calibrado para não confundir pisos claros do Lab com caixa de texto).
- **`OVERWORLD`:** fallback padrão.
