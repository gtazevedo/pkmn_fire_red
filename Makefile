.PHONY: train enjoy clean ui info

# Inicia o treinamento
train:
	python -m scripts.train

# Abre a janela gráfica para visualizar o agente jogando (necessita Pygame e tela ativa)
enjoy:
	python -m scripts.enjoy

# Abre a interface gráfica do MLflow no navegador
ui:
	mlflow ui -h 127.0.0.1 -p 5000

# Limpa caches do python
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	
# Mostra informações dos paths configurados
info:
	@echo "Paths atuais:"
	@echo "PKMN_SAVE_DIR: $${PKMN_SAVE_DIR:-/mnt/d/projects/pkmn_fire_red_saves}"
