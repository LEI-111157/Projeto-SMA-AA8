import json
import random
import pickle
from collections import defaultdict

from sim.agente import Agente
from sim.actions import Action


class AgenteLearning(Agente):
    """
    Agente de Q-learning tabular (usado no problema do Farol).

    Ideia geral:
    - Em TRAIN: aprende uma Q-table
    - Em TEST: carrega a Q-table e joga com exploração desativada (100% exploit)

    - epsilon e probabilidade de EXPLOIT (seguir a melhor acao),
      e (1 - epsilon) e a probabilidade de explorar.
    - durante o treino, epsilon cresce (vai explorando menos com o tempo).

    """

    def __init__(self, seed=42, learning=None, mode="train", qtable_path=None):
        super().__init__()
        self.rng = random.Random(seed)

        learning = learning or {}
        self.alpha = float(learning.get("alpha", 0.1))
        self.gamma = float(learning.get("gamma", 0.95))

        # Epsilon = probabilidade de EXPLOIT
        self.epsilon = float(learning.get("epsilon_start", 0.05))
        self.epsilon_max = float(learning.get("epsilon_max", 0.95))
        self.epsilon_growth = float(learning.get("epsilon_growth", 1.005))

        self.mode = mode  # "train" | "test"
        self.qtable_path = qtable_path

        # Q-table: usamos defaultdict(float) mas ao ler usamos get() para nao criar chaves sem querer
        self.Q = defaultdict(float)

        # Guardamos o (estado, acao) interior para fazer a atualização quando chega a recompensa
        self.prev_state = None
        self.prev_action = None

        self.actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

        # No modo TEST, as vezes o agente pode cair em ciclos; guardamos estados recentes
        self._recent_states = []
        self._recent_max = 8

        # Em TEST: desliga exploracao e carrega Q-table (se existir)
        if self.mode == "test":
            self.epsilon = 1.0
            if self.qtable_path:
                self.load_q(self.qtable_path)

    @staticmethod
    def cria(nome_do_ficheiro_parametros: str):
        with open(nome_do_ficheiro_parametros, "r", encoding="utf-8") as f:
            data = json.load(f)

        seed = data.get("seed", 42)
        learning = data.get("learning", {})
        mode = data.get("mode", "train")
        qtable_path = data.get("qtable_path", None)

        return AgenteLearning(seed=seed, learning=learning, mode=mode, qtable_path=qtable_path)

    def save_q(self, path: str) -> None:
        #Guarda a Q-table no disco para reutilizacao no modo de teste
        with open(path, "wb") as f:
            pickle.dump(dict(self.Q), f)

    def load_q(self, path: str) -> None:
        #Carrega uma Q-table previamente treinada
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.Q = defaultdict(float, d)

    def reset_episode(self):
        #Limpa vvariaveis temporarias do episodio
        self.prev_state = None
        self.prev_action = None
        self._recent_states = []

    def end_episode(self):
        #No fim do episodio aumenta-se o epsilon
        if self.mode == "train":
            self.epsilon = min(self.epsilon_max, self.epsilon * self.epsilon_growth)

    def _bin_dist(self, d):
        if d is None:
            return -1
        d = int(d)
        if d <= 2:
            return 0
        if d <= 5:
            return 1
        return 2

    def _state_from_obs(self, obs: dict):
       #Constroi um estado discreto a partir da observacao
        gdx = int(obs.get("goal_dx", 0))
        gdy = int(obs.get("goal_dy", 0))
        manhattan_bin = self._bin_dist(obs.get("manhattan", None))

        up = int(obs.get("cell_0_-1", 0))
        down = int(obs.get("cell_0_1", 0))
        left = int(obs.get("cell_-1_0", 0))
        right = int(obs.get("cell_1_0", 0))

        return (gdx, gdy, manhattan_bin, up, down, left, right)


    def _state_is_known(self, state) -> bool:

        #Consideramos um estado 'conhecido' se existir algum valor Q significativo

        for a in self.actions:
            if abs(self.Q.get((state, a), 0.0)) > 1e-9:
                return True
        return False

    def _fallback_action_farol(self) -> Action:
        #Heurística de segurança (apenas em TEST): se o estado nao existir na Q-table, tenta aproximar do goal sem colidir.
        obs = self._ultima_obs
        gdx = int(obs.get("goal_dx", 0))
        gdy = int(obs.get("goal_dy", 0))

        up = int(obs.get("cell_0_-1", 0))
        down = int(obs.get("cell_0_1", 0))
        left = int(obs.get("cell_-1_0", 0))
        right = int(obs.get("cell_1_0", 0))

        candidates = []

        # Preferencia pela componente maior (x ou y), como “andar na direção certa” primeiro
        if abs(gdx) >= abs(gdy):
            if gdx > 0:
                candidates.append((Action.RIGHT, right))
            elif gdx < 0:
                candidates.append((Action.LEFT, left))
            if gdy > 0:
                candidates.append((Action.DOWN, down))
            elif gdy < 0:
                candidates.append((Action.UP, up))
        else:
            if gdy > 0:
                candidates.append((Action.DOWN, down))
            elif gdy < 0:
                candidates.append((Action.UP, up))
            if gdx > 0:
                candidates.append((Action.RIGHT, right))
            elif gdx < 0:
                candidates.append((Action.LEFT, left))

        # Backups (se as preferidas estiverem bloqueadas)
        candidates += [
            (Action.UP, up),
            (Action.DOWN, down),
            (Action.LEFT, left),
            (Action.RIGHT, right),
        ]

        for act, cell in candidates:
            if cell == 0:  # livre
                return act

        return self.rng.choice(self.actions)

    def _best_action(self, state):
     #Escolhe a melhor acao segundo a Q-table. Em caso de empate, faz um desempate “direcional” para reduzir loops.
        qs = [(self.Q.get((state, a), 0.0), a) for a in self.actions]
        max_q = max(q for q, _ in qs)
        best_actions = [a for q, a in qs if q == max_q]

        # Desempate: tenta aproximar do goal usando dx/dy
        obs = self._ultima_obs
        gdx = int(obs.get("goal_dx", 0))
        gdy = int(obs.get("goal_dy", 0))

        preferred = []
        if abs(gdx) >= abs(gdy):
            if gdx > 0:
                preferred.append(Action.RIGHT)
            elif gdx < 0:
                preferred.append(Action.LEFT)
            if gdy > 0:
                preferred.append(Action.DOWN)
            elif gdy < 0:
                preferred.append(Action.UP)
        else:
            if gdy > 0:
                preferred.append(Action.DOWN)
            elif gdy < 0:
                preferred.append(Action.UP)
            if gdx > 0:
                preferred.append(Action.RIGHT)
            elif gdx < 0:
                preferred.append(Action.LEFT)

        for p in preferred:
            if p in best_actions:
                return p

        # Ultimo recurso: empate resolvido aleatoriamente
        return self.rng.choice(best_actions)

    def _choose_action(self, state):
        """
        Seleção de ação:
        - TEST: sempre exploit (e fallback se o estado for desconhecido)
        - TRAIN: epsilon-greedy com a convencao epsilon=probabilidade de exploit
        """
        if self.mode == "test":
            if not self._state_is_known(state):
                return self._fallback_action_farol()

            # Anti-loop simples: se ficar muitas vezes no mesmo estado, tenta “segunda melhor”
            self._recent_states.append(state)
            if len(self._recent_states) > self._recent_max:
                self._recent_states.pop(0)

            if self._recent_states.count(state) >= 3:
                qs = sorted(
                    [(self.Q.get((state, a), 0.0), a) for a in self.actions],
                    key=lambda x: x[0],
                    reverse=True
                )
                best = qs[0][1]
                for _, a in qs[1:]:
                    if a != best:
                        return a
                others = [a for a in self.actions if a != best]
                return self.rng.choice(others) if others else best

            return self._best_action(state)

        # TRAIN: epsilon e probabilidade de exploit (seguir o melhor)
        if self.rng.random() < self.epsilon:
            return self._best_action(state)
        return self.rng.choice(self.actions)

    def age(self) -> Action:
        #Escolhe acao com base no estado atual e guarda (estado,acao) para atualizacao posterior.
        state = self._state_from_obs(self._ultima_obs)
        action = self._choose_action(state)
        self.prev_state = state
        self.prev_action = action
        return action

    def avaliacaoEstadoAtual(self, recompensa: float) -> None:
        """
        Atualizacao do Q-learning

        Usamos:
            Q(s,a) <- Q(s,a) + alpha * (r + gamma*max_a' Q(s',a') - Q(s,a))
        """
        super().avaliacaoEstadoAtual(recompensa)

        if self.mode == "test":
            return
        if self.prev_state is None or self.prev_action is None:
            return

        s = self.prev_state
        a = self.prev_action
        r = float(recompensa)
        s2 = self._state_from_obs(self._ultima_obs)

        max_next = max(self.Q.get((s2, a2), 0.0) for a2 in self.actions)
        old = self.Q.get((s, a), 0.0)

        self.Q[(s, a)] = old + self.alpha * (r + self.gamma * max_next - old)
