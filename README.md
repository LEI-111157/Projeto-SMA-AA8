# Projeto – Simulador de Sistemas Multi-Agente

Projeto desenvolvido no âmbito da unidade curricular de Agentes Autónomos.

- Afonso Alves Nº111157
- Alexandre Costa Nº111206

## Estrutura
- `src/sim/` – motor de simulação, ambientes, agentes e sensores
- `params/` – ficheiros de configuração (train/test)
- `outputs/` – resultados gerados (ignorado no Git)

## Ambientes
- **Farol** – política fixa e Q-learning tabular
- **Foraging com Ninho** – política fixa e Novelty Search

## Requisitos
- Python 3.10+
- matplotlib para geração de gráficos de curva de aprendizagem

## Gerar Curvas de Aprendizagem

### Farol 
- Dentro da pasta src/ correr python plot_learning_curve.py outputs/....(depende de onde forem gerados os outputs, esta pasta outputs contém ficheiros csv que são gerados quando se corre o farol ou o foraging)

### Foraging
- Dentro da pasta src/ correr python plot_learning_curve.py outputs/....(depende de onde forem gerados os outputs, esta pasta outputs contém ficheiros csv que são gerados quando se corre o farol ou o foraging)

## Execução

### Dentro da diretoria src/

### Fixed Farol 
```bash
python run.py params/farol_fixed.json
```

### Fixed Foraging
```bash
python run.py params/foraging_fixed.json
````

### Treino Foraging
```bash
python run.py params/foraging_novelty_train.json
```

### Treino Farol
```bash
python run.py params/farol_learning_train.json
```
### Foraging Teste 
```bash
python run.py params/foraging_novelty_test.json
```

### Farol Teste 
```bash
python run.py params/farol_learning_test.json
```
## Para reduzir o impacto da aleatoridade, é possível avaliar ambas as políticas em múltiplas seeds e obter mean/std das métricas

### Farol (N=30)
- Basta correr o ficheiro batch_eval.py

### Foraging (N=30) 
- Basta correr o ficheiro batch_eval_foraging.py

