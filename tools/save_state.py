import stable_retro as retro
import pygame
import numpy as np

def main():
    env = retro.make(game='PokemonFireRed', state=None)
    obs = env.reset()

    pygame.init()
    width, height = 240 * 2, 160 * 2
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Play Pokemon! Press ESC to Save")

    state_path = "/home/guilh/projects/pkmn_fire_red/.venv/lib/python3.12/site-packages/stable_retro/data/stable/PokemonFireRed/Start.state.gz"
    clock = pygame.time.Clock()
    running = True

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()

            if keys[pygame.K_ESCAPE]:
                print("\nESC pressed! Saving state...")
                break

            # Your Exact Emulator Action Space (12 buttons)
            action = np.zeros(12, dtype=np.int8)
            
            # --- THE CORRECTED MAPPING ---
            if keys[pygame.K_x]: action[0] = 1         # B (Index 0)
            if keys[pygame.K_BACKSPACE]: action[2] = 1 # Select (Index 2)
            if keys[pygame.K_RETURN]: action[3] = 1    # Start (Index 3)
            if keys[pygame.K_UP]: action[4] = 1        # Up (Index 4)
            if keys[pygame.K_DOWN]: action[5] = 1      # Down (Index 5)
            if keys[pygame.K_LEFT]: action[6] = 1      # Left (Index 6)
            if keys[pygame.K_RIGHT]: action[7] = 1     # Right (Index 7)
            
            # THE MAGIC FIX: 'A' is actually Index 8!
            if keys[pygame.K_z] or keys[pygame.K_SPACE]: 
                action[8] = 1                          # A (Index 8)

            obs, reward, term, trunc, info = env.step(action)
            
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            surf = pygame.transform.scale(surf, (width, height))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(60)

    except Exception as e:
        print(f"Error during gameplay: {e}")
    finally:
        print(f"\nSaving clean state to {state_path}...")
        with open(state_path, "wb") as f:
            f.write(env.em.get_state())
        print("SUCCESS! Clean state saved.")
        env.close()
        pygame.quit()

if __name__ == "__main__":
    main()
