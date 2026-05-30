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
│   ├── reader.py             # Leitor de baixo nível de RAM do GBA (RamReader e Flags)
│   ├── advisor.py            # Sistema advisor com decodificação TBL de EWRAM (RamAdvisor)
│   ├── progress.py           # Currículo Dinâmico baseado em Event Flags Universais
│   ├── env.py                # Wrapper Gymnasium em torno do stable-retro (PokemonEnv)
│   └── callbacks.py          # Monitor de TensorBoard e motor de treinamento (Trainer)
│
├── research/                 # Scripts de mineração e descoberta (mantidos fora de prod)
│   ├── mine_flags.py         # Extrator de offsets absolutos da SaveBlock1
│   ├── probe_ram.py          # Rastreador de deltas de RAM
│   └── global_pret.h         # Header C original de referência do Pokémon
│
├── tools/                    # Ferramentas auxiliares e de diagnóstico
│   ├── save_state.py         # Interface interativa para salvar save states
│   └── text_advisor.py       # Histórico da tentativa de OCR antiga (Tesseract)
│
├── checkpoints/              # Pasta para backups automáticos de modelos
├── progress_states/          # Save states dinâmicos salvos pelo próprio agente ao longo do tempo
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
Para iniciar ou retomar o treinamento em background (com múltiplos workers paralelos nativos):
```bash
python pokemon_rl.py
```
O treinamento utiliza um sistema contínuo. Se houver um modelo treinado na pasta raiz (`pokemon_lstm_model.zip`), ele continuará a aprender **automaticamente** do último passo sem perder progresso.

### 3. Assistir ao Agente Jogando (Tempo Real)
Criamos um visualizador interativo em tempo real usando **Pygame** que roda o modelo LSTM treinado no modo determinístico:
```bash
python watch_agent.py
```
A janela renderizará a tela do GBA com upscale em 60 FPS, permitindo que você assista à tomada de decisão do agente frame a frame.

### 4. Monitoramento com TensorBoard
Para assistir às métricas detalhadas de recompensa, exploração de mapas, nível do time e eventos descobertos:
```bash
tensorboard --logdir ./pokemon_tensorboard/ --samples_per_plugin scalars=10000
```
Abra o navegador em `http://localhost:6006`.

---

## 🧠 Currículo Dinâmico e "Anti-Farming" de Eventos

A evolução monumental deste projeto (a partir da v18.3) foi a remoção total das obsoletas "Milestones" hardcoded. Agora o agente tem um instinto de **Curiosidade Pura** focado na história.

### 1. Soma Universal de Eventos (Event Flags)
Inspirado por arquiteturas de RL massivas (e confirmando offset original do código fonte em C do Game Boy Advance), o agente realiza um dump do bloco `SaveBlock1` a partir do ponteiro raiz dinâmico da IRAM (`0x03005008`). Ele varre uma array exata de **288 bytes (2304 bits)** a partir do offset `0x0EE0` que abriga todas as *Event Flags* do jogo.

Ao invés de programarmos o agente para "ir para Viridian City", nós o recompensamos sempre que ele consegue **acender um novo bit** dessa matriz (o que significa pegar um item novo, conversar com um NPC chave, vencer um ginásio ou ativar uma cutscene).

### 2. Algoritmo de High-Score "Anti-Farming"
Alguns bits do jogo podem ser ligados e desligados em loops infinitos pelos jogadores (como acessar o PC do quarto). Para impedir que o agente manipule o jogo explorando essas repetições de flags, desenvolvemos uma lógica que guarda o **Recorde Global** de Eventos Ativados no episódio. 

A inteligência só ganha a colossal recompensa de Evento se o total atual de flags ativas for estritamente **maior** do que o recorde absoluto alcançado até o momento, eliminando qualquer brecha de exploração infinita (*reward hacking*).

### 3. State Saving Orgânico
Para não forçar o agente a refazer 30 minutos de gameplay a cada reset do emulador, a rede automaticamente realiza um dump completo e cria um `.state` na pasta `progress_states` a cada 5 novas flags inéditas encontradas na história do treinamento. Nos próximos reinícios, ele sorteará esses saves, gerando uma Árvore de Exploração infinita (*Curriculum Learning* dinâmico).

---

## ⚔️ Leitura de RAM Direta (Zero-Lag)

### 1. Leitura Direta de Diálogos EWRAM (Microssegundos)
Ao invés de rodar OCRs pesados de IA na CPU (como Tesseract ou EasyOCR) que derrubavam a velocidade de simulação, o agente lê diretamente a memória dinâmica de diálogo no endereço de Work RAM **`0x02021D18`** (EWRAM).
*   Os bytes são traduzidos em microssegundos usando o mapa de caracteres TBL nativo de Pokémon de 3ª Geração.
*   Isso nos fornece leitura de texto com **100% de precisão e zero overhead de latência**, mantendo a simulação no seu pico máximo de FPS.

### 2. LSTM (Memória Temporal) + Frame Parity
A arquitetura é envelopada num `RecurrentPPO` rodando em cima de um `MultiInputPolicy`. Isso garante não apenas o mapeamento do mundo em convoluções 2D, mas também entrega um array escalar brutal via rede *Dense* direto do emulador. Além disso, a presença do LSTM nativo ensina a rede a esperar a conclusão de animações sem a necessidade artificial de injetar *Frame Skips* absurdos que tornariam os menus e batalhas letárgicos. O agente age precisamente quando deve.