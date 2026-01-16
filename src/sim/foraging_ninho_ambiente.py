import random
from typing import Optional

from sim.ambiente import Ambiente
from sim.agente import Agente
from sim.actions import Action


class AmbienteForagingNinho(Ambiente):
    """
    Ambiente de Foraging Ninho

    Objetivo do episodio:
    -recolher recursos(F) e deposita-los no ninho(N)
    - o episodio acaba quando depositados == n_recursos

    Regras:
    -o agente transporta no máximo 1 recurso
    -ao apanhar um recurso: +20
    -ao depositar no ninho: +30
    -passo: -1
    -colisao: -5
    -entregar todos os recursos: +50
    """
    def __init__(
        self,
        width=8,
        height=8,
        obstacle_ratio=0.12,
        n_recursos=6,
        seed: Optional[int] = None
    ):
        self.width = width
        self.height = height
        self.obstacle_ratio = obstacle_ratio
        self.n_recursos = n_recursos
        self.rng = random.Random(seed)

        #Elementos do mapa
        self.obstacles: set[tuple[int, int]] = set()
        self.recursos: set[tuple[int, int]] = set()
        self.ninho: tuple[int, int] = (0, 0)
        self.agent_pos: tuple[int, int] = (0, 0)

        #Contadores agregados (usados para metricas e condicao de sucesso)
        self.coletados = 0
        self.depositados = 0

    def reset(self):
        self.coletados = 0
        self.depositados = 0

        #Posicao inicial do agente e do ninho
        self.agent_pos = self._random_cell()
        self.ninho = self._random_cell(exclude={self.agent_pos})

        # obstáculos
        self.obstacles = set()
        n_obs = int(self.width * self.height * self.obstacle_ratio)
        forbidden = {self.agent_pos, self.ninho}

        while len(self.obstacles) < n_obs:
            p = self._random_cell()
            if p not in forbidden:
                self.obstacles.add(p)

        # recursos(F)
        self.recursos = set()
        forbidden2 = forbidden | self.obstacles
        while len(self.recursos) < self.n_recursos:
            p = self._random_cell()
            if p not in forbidden2:
                self.recursos.add(p)

    def observacaoPara(self, agente: Agente) -> dict:
        #Constroi a observacao do agente. A observacao contem sensores,estado interno e contadores(collected/deposited)
        obs = {}
        for s in agente._sensores:
            obs.update(s.sense(self, self.agent_pos))
        #Posicao do agente
        obs["agent"] = self.agent_pos
        #Estado interno do agente
        obs["carrying"] = bool(getattr(agente, "carrying", False))
        #Contaadores
        obs["collected"] = self.coletados
        obs["deposited"] = self.depositados
        return obs

    def atualizacao(self) -> None:
        return

    def agir(self, accao: Action, agente: Agente):
        #Aplica a acao do agente
        ax, ay = self.agent_pos
        nx, ny = ax, ay

        #Propor nova posicao
        if accao == Action.UP: ny -= 1
        elif accao == Action.DOWN: ny += 1
        elif accao == Action.LEFT: nx -= 1
        elif accao == Action.RIGHT: nx += 1
        #Verificar colisoes
        blocked = False
        if not (0 <= nx < self.width and 0 <= ny < self.height):
            blocked = True
        elif (nx, ny) in self.obstacles:
            blocked = True
        #Reward base do passo
        recompensa = -1.0
        #Penalizacao por tentativa invalida, mantem posicao
        if blocked:
            recompensa = -5.0
            nx, ny = ax, ay
        #Atualiza posicao
        self.agent_pos = (nx, ny)

        # recolher recurso
        if not agente.carrying and self.agent_pos in self.recursos:
            self.recursos.remove(self.agent_pos)
            agente.carrying = True
            self.coletados += 1
            recompensa += 20.0

        # depositar no ninho
        if agente.carrying and self.agent_pos == self.ninho:
            agente.carrying = False
            self.depositados += 1
            recompensa += 30.0
        #Criterio de termino/sucesso
        terminou = (self.depositados == self.n_recursos)
        if terminou:
            recompensa += 50.0
        #debug
        info = {
            "blocked": blocked,
            "carrying": agente.carrying,
            "depositados": self.depositados
        }

        obs = self.observacaoPara(agente)
        return obs, recompensa, terminou, info

    def render_text(self) -> str:
        #Representacao textual do mapa (grelha)
        rows = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                p = (x, y)
                if p == self.agent_pos:
                    row.append("A")
                elif p == self.ninho:
                    row.append("N")
                elif p in self.obstacles:
                    row.append("#")
                elif p in self.recursos:
                    row.append("F")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)

    def _random_cell(self, exclude: Optional[set[tuple[int, int]]] = None):
        exclude = exclude or set()
        while True:
            p = (self.rng.randrange(self.width), self.rng.randrange(self.height))
            if p not in exclude:
                return p
