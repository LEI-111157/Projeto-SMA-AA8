import json
import random
from sim.agente import Agente
from sim.actions import Action


class AgentePoliticaFixa(Agente):
    """
    Agente baseline com politica fixa.

    Não aprende , serve de comparacao para Q-learning/Novelty
    Usa direcao fornecida pelos sensores para se aproximar do alvo, em caso de bloqueio introduz aleatoridade.

    Foraging:
    -Se tiver a transportar recurso segue para o ninho, caso contrario, segue para o recurso mais proximo.
    Farol:
    -Movimenta-se em direcao ao Goal.

    """
    def __init__(self, seed=42):
        super().__init__()
        self.rng = random.Random(seed)
        #Contador de colisoes para evitar ficar preso na parede ou em obstaculo.
        self.blocked_streak = 0
        #Necessario no foraging
        self.carrying = False

    @staticmethod
    def cria(nome_do_ficheiro_parametros: str):
        # agora respeita a seed do JSON
        with open(nome_do_ficheiro_parametros, "r", encoding="utf-8") as f:
            data = json.load(f)
        seed = data.get("seed", 42)
        return AgentePoliticaFixa(seed=seed)

    def age(self) -> Action:
        #Escolhe uma acao que reduza a distancia ao Goal,se houver varios "candidatos" escolhe aleatoriamente
        obs = self._ultima_obs

        #Se bater duas vezes forca um passo aleatorio
        if self.blocked_streak >= 2:
            return self.rng.choice([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT])

        #Escolha do alvvo com base no tipo de problema e no estado do agente.
        if obs.get("carrying", False):
            #Voltar ao ninho pra depositar
            dx = obs.get("nest_dx", 0)
            dy = obs.get("nest_dy", 0)
        elif "food_dx" in obs:
            #Procurar recurso
            dx = obs.get("food_dx", 0)
            dy = obs.get("food_dy", 0)
        else:
            #Mover em direcao ao Goal (Caso do Farol)
            dx = obs.get("goal_dx", 0)
            dy = obs.get("goal_dy", 0)
        #Traduz dx/dy (-1,0,1) em acoes
        options = []
        if dx == 1:
            options.append(Action.RIGHT)
        elif dx == -1:
            options.append(Action.LEFT)

        if dy == 1:
            options.append(Action.DOWN)
        elif dy == -1:
            options.append(Action.UP)
        #Se nao houver uma direcao sugerida, fica parado.
        if not options:
            return Action.STAY

        return self.rng.choice(options)

    def avaliacaoEstadoAtual(self, recompensa: float) -> None:
        super().avaliacaoEstadoAtual(recompensa)
        if recompensa == -5.0:
            self.blocked_streak += 1
        else:
            self.blocked_streak = 0
