"""
watch_agent.py
==============
Visualizador gráfico em tempo real para o agente de Pokémon FireRed RL.
Carrega o modelo treinado (PPO Recorrente) e os dados de normalização (VecNormalize)
e renderiza a tela em uma janela Pygame.

Como Executar:
--------------
  .venv/bin/python watch_agent.py
"""

import os
import sys
import time
import pygame
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Certifica de que a pasta raiz do projeto está no path para importar src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.env import PokemonEnv
from src.config import CFG, ActionSpace

def main():
    print("=== Inicializando o Visualizador do Agente ===")
    
    # 1. Instancia o ambiente com as configurações exatas do treino
    # (Inference mode: env_id=99, render_mode pode ser customizado)
    def make_env():
        return PokemonEnv(env_id=99)
    
    venv = DummyVecEnv([make_env])
    
    # 2. Carrega as estatísticas do VecNormalize
    vecnorm_path = CFG.model_path + "_vecnorm.pkl"
    if os.path.exists(vecnorm_path):
        print(f"Restaurando estatísticas do normalizador de: {vecnorm_path} ...")
        venv = VecNormalize.load(vecnorm_path, venv)
        # Em modo de inferência (play/visualização), não atualizamos as médias rodantes
        venv.training = False
        venv.norm_reward = False
    else:
        print("AVISO: Normalizador vecnorm não encontrado! Executando sem normalização.")

    # 3. Carrega o modelo RecurrentPPO treinado
    model_zip = CFG.model_path + ".zip"
    if os.path.exists(model_zip):
        print(f"Carregando modelo treinado de: {model_zip} ...")
        model = RecurrentPPO.load(model_zip, env=venv)
    else:
        print(f"ERRO: Modelo treinado não encontrado em {model_zip}!")
        print("Por favor, certifique-se de que o modelo foi treinado e salvo.")
        venv.close()
        return

    # 4. Inicializa o Pygame para renderização visual
    pygame.init()
    # Resolução nativa GBA: 240x160. Aplicamos upscale de 3x para melhor visualização (720x480)
    scale_factor = 3
    native_w, native_h = 240, 160
    window_w, window_h = native_w * scale_factor, native_h * scale_factor
    
    screen = pygame.display.set_mode((window_w, window_h))
    pygame.display.set_caption("Visualizador Pokémon FireRed RL - PPO Recorrente")
    clock = pygame.time.Clock()

    print("\nIniciando gameplay do agente...")
    print("Controles na janela do Pygame:")
    print("  - ESC ou fechar janela: Sair")
    
    # Força o visualizador a sempre iniciar no melhor milestone salvo (ex: Viridian City)
    raw_env = venv.envs[0]
    if raw_env._progress._best_milestone_idx >= 0:
        print(f"\n[Visualizador] Forçando inicialização no melhor marco salvo: {raw_env._progress._best_state_file}\n")
        import types
        raw_env._progress.current_state_file = types.MethodType(
            lambda self: self._best_state_file, 
            raw_env._progress
        )

    obs = venv.reset()
    
    # Estados iniciais da memória LSTM do modelo recorrente
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    running = True
    try:
        while running:
            # Captura eventos do Pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            # 5. Seleciona a ação ideal usando a rede neural com sua memória recorrente LSTM
            action_idx, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True, # Modo determinístico: seleciona sempre a melhor ação
            )
            
            # Executa a ação no emulador
            obs, rewards, dones, infos = venv.step(action_idx)
            episode_starts[0] = dones[0]

            # 6. Renderiza o frame em tempo real na tela Pygame
            # A classe PokemonEnv armazena o último frame bruto em self._last_raw_frame
            # Acessamos o ambiente interno do DummyVecEnv
            raw_env = venv.envs[0]
            raw_frame = raw_env._last_raw_frame if hasattr(raw_env, "_last_raw_frame") else None

            if raw_frame is not None:
                # Transpõe o frame RGB do emulador de (H, W, C) para o formato esperado pelo Pygame (W, H, C)
                frame_transposed = np.transpose(raw_frame, (1, 0, 2))
                surface = pygame.surfarray.make_surface(frame_transposed)
                # Redimensiona para o tamanho com upscale
                scaled_surface = pygame.transform.scale(surface, (window_w, window_h))
                screen.blit(scaled_surface, (0, 0))
                pygame.display.flip()

            # Roda a 60 FPS (velocidade normal do GBA). 
            # Aumente esse valor se quiser assistir o agente jogando em alta velocidade!
            clock.tick(60)

    except Exception as e:
        print(f"Erro durante a execução do visualizador: {e}")
    finally:
        print("\nFechando emulador e Pygame...")
        venv.close()
        pygame.quit()
        print("Concluído.")

if __name__ == "__main__":
    main()
