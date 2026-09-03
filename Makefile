.PHONY: train enjoy clean ui info

# Inicia o treinamento
train:
	~/.local/bin/poetry run python -m scripts.train

# Inicia a bateria de testes de hiperparâmetros
tune:
	~/.local/bin/poetry run python -m scripts.tune

# Abre a janela gráfica para visualizar o agente jogando (necessita Pygame e tela ativa)
enjoy:
	~/.local/bin/poetry run python -m scripts.enjoy

# Abre a interface gráfica do MLflow no navegador
ui:
	~/.local/bin/poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db -h 127.0.0.1 -p 5000

# Limpa caches do python
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	
# Mostra informações dos paths configurados
info:
	@echo "Paths atuais:"
	@echo "PKMN_SAVE_DIR: $${PKMN_SAVE_DIR:-/mnt/d/projects/pkmn_fire_red_saves}"
