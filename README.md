# Projeto – Simulador de Sistemas Multi-Agente

Projeto desenvolvido no âmbito da unidade curricular de Agentes Autónomos.

Afonso Alves Nº111157

Alexandre Costa Nº111206

## Estrutura
- `src/sim/` – motor de simulação, ambientes, agentes e sensores
- `params/` – ficheiros de configuração (train/test)
- `outputs/` – resultados gerados (ignorado no Git)

## Ambientes
- **Farol** – política fixa e Q-learning tabular
- **Foraging com Ninho** – política fixa e Novelty Search

## Execução

### Treino Foraging
```bash
python run.py params/foraging_novelty_train.json
```
### Treino Farol
```bash
python run.py params/farol_learning_train.json
```
