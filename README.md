# Pokémon FireRed RL: Agente de Aprendizado por Reforço Profundo

Este repositório contém um agente de Aprendizado por Reforço Profundo (Deep Reinforcement Learning) de última geração, projetado para aprender a jogar **Pokémon FireRed** no Game Boy Advance (GBA) usando o framework **stable-retro** e algoritmos baseados em políticas de gradiente recorrentes (PPO com LSTM).

---

## 📁 Arquitetura do Projeto

O código foi projetado seguindo as melhores práticas da engenharia de software, dividindo responsabilidades de forma modular e clara:

```text
pkmn_fire_red/
├── src/                      # Módulos de código-fonte
│   ├── __init__.py           # Inicializador do pacote python
│   ├── config.py             # Parâmetros tuneáveis do modelo (CFG) e ActionSpace
│   ├── reader.py             # Leitor de baixo nível de RAM do GBA (RamReader)
│   ├── advisor.py            # Sistema advisor com decodificação TBL de EWRAM (RamAdvisor)
│   ├── progress.py           # Gerenciador de checkpoints/milestones e estatísticas de episódios
│   ├── env.py                # Wrapper Gymnasium em torno do stable-retro (PokemonEnv)
│   └── callbacks.py          # Monitor de TensorBoard e motor de treinamento (Trainer)
│
├── tools/                    # Scripts auxiliares e ferramentas de diagnóstico
│   ├── save_state.py         # Interface interativa para salvar save states personalizados
│   └── text_advisor.py       # Histórico da tentativa de OCR antiga (Tesseract)
│
├── checkpoints/              # Pasta para backups automáticos de modelos
├── progress_states/          # Save states das cidades descobertas (milestones)
├── pokemon_tensorboard/      # Logs detalhados para visualização no TensorBoard
├── whiteout_frames/          # Capturas de tela no momento em que o time é derrotado
│
├── rom.gba                   # ROM original de Pokémon FireRed
├── pokemon_rl.py             # Script de entrada leve para iniciar o TREINAMENTO
└── watch_agent.py            # Script de entrada leve para assistir o agente via Pygame
```

---

## 🚀 Como Executar

### 1. Requisitos e Dependências
Certifique-se de estar usando um ambiente virtual Python 3.12 (ou compatível). Instale as dependências principais:
```bash
pip install stable-retro stable-baselines3 sb3-contrib pygame opencv-python gymnasium
```

*Nota: Você deve colocar uma ROM válida e limpa de Pokémon FireRed renomeada como `rom.gba` na raiz do projeto (ou usar `stable-retro` para importar a ROM).*

### 2. Executar o Treinamento
Para iniciar ou retomar o treinamento em background (com 6 workers paralelos por padrão):
```bash
python pokemon_rl.py
```
O treinamento utiliza um sistema de carimbo de data/hora (`config_stamp`). Se houver um modelo treinado na pasta raiz (`pokemon_lstm_model.zip`), ele continuará a aprender **automaticamente** do último passo sem perder progresso.

*Dica: Você pode usar `tmux` no Linux para manter o processo ativo se fechar o terminal:*
```bash
sudo apt install tmux
tmux new -s treino
python pokemon_rl.py
# Pressione Ctrl+B depois D para desanexar do tmux
```

### 3. Assistir ao Agente Jogando (Tempo Real)
Criamos um visualizador interativo em tempo real usando **Pygame** que roda o modelo LSTM treinado no modo determinístico:
```bash
python watch_agent.py
```
A janela renderizará a tela do GBA com upscale de 3x em 60 FPS, permitindo que você assista à tomada de decisão do agente frame a frame. Pressione `ESC` na tela do Pygame para fechar o visualizador com segurança.

### 4. Monitoramento com TensorBoard
Para assistir às curvas de recompensas, taxa de vitórias, níveis ganhos e métricas de exploração geográfica em tempo real:
```bash
tensorboard --logdir ./pokemon_tensorboard/ --samples_per_plugin scalars=10000
```
Abra o navegador em `http://localhost:6006`.

---

## 🧠 Mecânicas Avançadas e IA de Combate

### 1. Leitura Direta de Diálogos EWRAM (Microssegundos)
Ao invés de rodar OCRs pesados de IA na CPU (como Tesseract ou EasyOCR) que derrubavam a velocidade de simulação, o agente lê diretamente a memória dinâmica de diálogo no endereço de Work RAM **`0x02021D18`** (EWRAM).
*   Os bytes são traduzidos em microssegundos usando o mapa de caracteres TBL nativo de Pokémon de 3ª Geração.
*   Isso nos fornece leitura de texto com **100% de precisão e zero overhead de latência**, mantendo a velocidade de simulação em incríveis **800+ FPS**.

### 2. Classificação de Situações de Jogo
Com base no texto lido da RAM, o `RamAdvisor` mapeia o estado atual nas seguintes situações:
*   `SIT_EXPLORE (1)`: Exploração padrão no mapa overworld.
*   `SIT_DIALOG (2)`: Conversa com NPCs, tutoriais ou introdução do Prof. Oak. (Recompensa por pressionar **A**).
*   `SIT_BATTLE (3)`: Tela inicial de combate (FIGHT / BAG / POKéMON / RUN).
*   `SIT_YES_NO (4)`: Prompts de Sim/Não. (Recompensa por pressionar **B** para evitar loops infinitos ou salvar sobre estados).
*   `SIT_LEVEL_UP (5)`: Telas de ganho de level. (Recompensa por pressionar **A**).
*   `SIT_HEAL (6)`: Cura de Pokémon no Centro Pokémon. (Recompensa por pressionar **A**).
*   `SIT_MOVE_SELECT (7)`: **[NOVO]** Tela de seleção de golpes dentro do menu de combate (FIGHT). Detectada quando a string `"pp "` ou `"pp  "` está ativa na RAM. (Recompensa de **`0.20`** por pressionar **A** para atacar).

### 3. Proteção Contra Stuck e Map Loops
*   **Detector de Stuck:** Aplica penalidade incremental de `-0.05` caso o agente permaneça no exato mesmo tile por mais de 150 passos fora de batalhas ou menus trancados.
*   **Detector de Stairs Loop:** Evita que o agente se vicie em entrar e sair de transições de mapas (como subir e descer as escadas do quarto em loops rápidos). Se uma transição de mapa for feita em menos de 100 passos após a anterior, uma severa punição de `-3.0` é aplicada.
*   **Stochastic Farm Ratio:** Para evitar o "farming passivo de batalhas" (onde o agente entra na grama alta apenas para ganhar bônus de entrada sem lutar), o sistema sorteia aleatoriamente um limite de tolerância de ociosidade de batalha (`_farm_ratio_threshold`) a cada episódio. Se o agente passar a maior parte do tempo em batalhas sem infligir dano, o episódio é abortado prematuramente com uma penalidade pesada.