import random
from typing import Optional
from sim.ambiente import Ambiente
from sim.agente import Agente
from sim.actions import Action


class AmbienteFarol(Ambiente):
    """
        Ambiente de Farol

        Objetivo do episodio:
        - o agente (A) deve chegar ao objetivo (G)

        Regras:
        -atingir objetivo: +100
        -passo: -1
        -colisao: -5

        """
    def __init__(self, width=8, height=8, obstacle_ratio=0.18, seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.obstacle_ratio = obstacle_ratio
        self.rng = random.Random(seed)

        self.obstacles: set[tuple[int, int]] = set()
        self.agent_pos: tuple[int, int] = (0, 0)
        self.goal: tuple[int, int] = (width - 1, height - 1)

    def reset(self):
        self.goal = self._random_cell()
        self.agent_pos = self._random_cell(exclude={self.goal})

        #Nao meter os obstaculos em cima do agente ou do goal
        self.obstacles = set()
        n_obs = int(self.width * self.height * self.obstacle_ratio)
        forbidden = {self.goal, self.agent_pos}

        while len(self.obstacles) < n_obs:
            p = self._random_cell()
            if p not in forbidden:
                self.obstacles.add(p)

    def observacaoPara(self, agente: Agente) -> dict:
        #Constroi a observacao do agente a partir dos sensores
        obs = {}
        for s in agente._sensores:
            obs.update(s.sense(self, self.agent_pos))
        obs["agent"] = self.agent_pos
        obs["goal"] = self.goal
        return obs

    def atualizacao(self) -> None:
        return

    def agir(self, accao: Action, agente: Agente):
        #Aplica a acao do Agente. Inclui success de forma explicita.
        ax, ay = self.agent_pos
        nx, ny = ax, ay

        #Propor nova posicao
        if accao == Action.UP:
            ny -= 1
        elif accao == Action.DOWN:
            ny += 1
        elif accao == Action.LEFT:
            nx -= 1
        elif accao == Action.RIGHT:
            nx += 1

        #Validar o movimento
        blocked = False
        if not (0 <= nx < self.width and 0 <= ny < self.height):
            blocked = True
        elif (nx, ny) in self.obstacles:
            blocked = True
        #Reward base por passo e colisao
        if blocked:
            recompensa = -5.0
            nx, ny = ax, ay
        else:
            recompensa = -1.0
        #Condicao de termino
        self.agent_pos = (nx, ny)
        terminou = (self.agent_pos == self.goal)
        if terminou:
            recompensa = 100.0

        #Debug
        info = {"blocked": blocked, "success": terminou}

        obs = self.observacaoPara(agente)
        return obs, recompensa, terminou, info

    def render_text(self) -> str:
        #Representacao textual do mapa
        rows = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                p = (x, y)
                if p == self.agent_pos:
                    row.append("A")
                elif p == self.goal:
                    row.append("G")
                elif p in self.obstacles:
                    row.append("#")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)

    def _random_cell(self, exclude: Optional[set[tuple[int, int]]] = None):
        #Escolhe uma celula que nao esteja no set de celulas de exclusao.
        exclude = exclude or set()
        while True:
            p = (self.rng.randrange(self.width), self.rng.randrange(self.height))
            if p not in exclude:
                return p
