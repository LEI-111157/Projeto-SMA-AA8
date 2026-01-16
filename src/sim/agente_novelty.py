# sim/agente_novelty.py
import pickle
import random
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

from sim.agente import Agente
from sim.actions import Action


@dataclass
class NoveltyConfig:
    """
    Parametros do Novelty Search.

    - k: nº de vizinhos usados para calcular novelty (media das distancias aos k mais proximos)
    - archive_add_threshold: quao “diferente” tem de ser um comportamento para entrar no arquivo
    - sigma: forca da mutacao (gaussiana) ao gerar novas politicas a partir de elites
    - random_policy_prob: probabilidade de gerar politica totalmente aleatoria (diversidade)
    - archive_max: limite de memoria do arquivo
    - elite_keep: nº maximo de elites guardadas (por novelty)
    """
    k: int = 15
    archive_add_threshold: float = 0.6
    sigma: float = 0.35
    random_policy_prob: float = 0.30
    archive_max: int = 800
    elite_keep: int = 25


class AgenteNovelty(Agente):
    """
    Agente baseado em Novelty Search para o Foraging com Ninho.

    Ideia:
    - Durante o treino, nao tentamos maximizar reward diretamente em cada episodio.
      Em vez disso, procuramos diversidade comportamental (novelty) para explorar o espaco.
    - Em paralelo, guardamos a melhor politica segundo um objective simples (depositar/colher)
      para depois usar em TEST/batch_eval.

    Decisões importantes:
    - Exploracao é decidida POR EPISODIO (nao por passo) para nao destruir trajetorias longas.
    - A politica é uma heuristica parametrizada por 4 pesos (pesos mudam por episódio).
    - Sensores food_dx/food_dy e nest_dx/nest_dy são direcoes (-1/0/+1), não distancias.
    """

    def __init__(
        self,
        seed: int = 42,
        mode: str = "train",
        novelty: Optional[dict] = None,
        policy_path: Optional[str] = None,
    ):
        super().__init__()
        self.rng = random.Random(seed)
        self.mode = mode
        self.policy_path = policy_path

        novelty = novelty or {}
        self.cfg = NoveltyConfig(
            k=int(novelty.get("k", 15)),
            archive_add_threshold=float(novelty.get("archive_add_threshold", 0.6)),
            sigma=float(novelty.get("sigma", 0.35)),
            random_policy_prob=float(novelty.get("random_policy_prob", 0.30)),
            archive_max=int(novelty.get("archive_max", 800)),
            elite_keep=int(novelty.get("elite_keep", 25)),
        )

        self.actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

        # O ambiente usa este atributo para gerir recolha/deposito.
        self.carrying = False

        # Variaveis de episódio
        self._episode_steps = 0
        self._end_obs = None

        # Anti-backtracking: memoriza as duas ultimas posicoes
        self._prev_pos = None
        self._prev_prev_pos = None

        # Flag de exploracao por episódio (no treino)
        self._episode_explore = False

        # Estruturas do Novelty Search:
        # - archive: lista de comportamentos (BD) que foram suficientemente diferentes
        # - elites: top politicas segundo novelty (para gerar mutacoes)
        self.archive: List[Tuple[Tuple[float, ...], List[float]]] = []
        self.elites: List[Tuple[float, Tuple[float, ...], List[float]]] = []

        # Politica atual: 4 pesos que controlam heuristicas de decisao
        self.weights: List[float] = self._random_weights()

        # Separado do novelty: guardamos a melhor politica segundo um objective pratico
        # (para produzir uma policy “boa” para teste).
        self.best_obj_score = float("-inf")
        self.best_weights = self.weights[:]

        # Em TEST, queremos jogar com a melhor politica guardada no treino (best-only).
        if self.mode == "test" and self.policy_path:
            self.load_policy(self.policy_path)
            self.best_weights = self.weights[:]

    @staticmethod
    def cria(nome_do_ficheiro_parametros: str):
        with open(nome_do_ficheiro_parametros, "r", encoding="utf-8") as f:
            data = json.load(f)

        seed = data.get("seed", 42)
        mode = data.get("mode", "train")
        novelty = data.get("novelty", None)
        policy_path = data.get("policy_path", None)

        return AgenteNovelty(
            seed=seed,
            mode=mode,
            novelty=novelty,
            policy_path=policy_path
        )

    def _save_best_only(self, path: str, weights: List[float]) -> None:
        """
        Guardar so a policy final que interessa para TEST/batch_eval.
        Isto evita depender do arquivo/elites no modo de teste.
        """
        with open(path, "wb") as f:
            pickle.dump({"weights": weights}, f)

    def save_policy(self, path: str) -> None:
    #Guarda o estado completo (util para debug/analise), mas no teu fluxo normal o TEST so precisa da best-only.
        payload = {
            "weights": self.weights,
            "archive": self.archive,
            "elites": self.elites,
            "best_obj_score": self.best_obj_score,
            "best_weights": self.best_weights,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load_policy(self, path: str) -> None:
    #Carrega policy guardada. Suporta best-only e full payload.
        with open(path, "rb") as f:
            payload = pickle.load(f)

        # Formato simples: apenas pesos
        if isinstance(payload, dict) and "weights" in payload and "archive" not in payload:
            self.weights = payload["weights"]
            self.archive = []
            self.elites = []
            return

        # Formato completo
        self.weights = payload.get("weights", self._random_weights())
        self.archive = payload.get("archive", [])
        self.elites = payload.get("elites", [])
        self.best_obj_score = float(payload.get("best_obj_score", float("-inf")))
        self.best_weights = payload.get("best_weights", self.weights[:])

    def reset_episode(self):
        # Reinicia variaveis do episodio e escolhe a politica a usar neste episodio.
        self._episode_steps = 0
        self._end_obs = None

        self.carrying = False

        # Reset do anti-backtracking
        self._prev_pos = None
        self._prev_prev_pos = None

        # Exploracao por episodio
        p_explore_episode = 0.25
        self._episode_explore = (self.mode == "train" and self.rng.random() < p_explore_episode)

        # Em treino, cada episodio pode experimentar uma politica diferente
        if self.mode == "train":
            self.weights = self._select_next_policy()

    def end_episode(self):
        """
        No fim do episodio (treino):
        - calcula o descritor comportamental (BD)
        - calcula novelty e atualiza elites/arquivo
        - atualiza tambem a melhor politica segundo objective (para TEST)
        """
        if self.mode != "train":
            return
        if self._end_obs is None:
            return

        bd = self._behavior_descriptor(self._end_obs, self._episode_steps)
        nov = self._novelty_score(bd)

        # Guardamos elites com base em NOVELTY (diversidade)
        self._update_elites(nov, bd, self.weights)

        # Arquivo: guarda comportamentos suficientemente diferentes (ou arranque inicial)
        if nov >= self.cfg.archive_add_threshold or len(self.archive) < self.cfg.k:
            self.archive.append((bd, self.weights[:]))
            if len(self.archive) > self.cfg.archive_max:
                self.archive.pop(self.rng.randrange(len(self.archive)))

        # Separadamente, guardamos a melhor politica por OBJECTIVE (desempenho “pratico”)
        obj = self._objective_score(self._end_obs, self._episode_steps)
        if obj > self.best_obj_score:
            self.best_obj_score = obj
            self.best_weights = self.weights[:]

        # Escreve sempre best-only para o TEST usar diretamente
        if self.policy_path:
            self._save_best_only(self.policy_path, self.best_weights)

    def age(self) -> Action:
    #Escolhe a proxima acao com base na observacao atual e na politica ativa
        self._episode_steps += 1

        # Guardar posicoes recentes para penalizar voltar atras (A-B-A)
        pos = self._ultima_obs.get("agent", None)
        self._prev_prev_pos = self._prev_pos
        self._prev_pos = pos

        return self._policy_action(self._ultima_obs)

    def avaliacaoEstadoAtual(self, recompensa: float) -> None:
    #Guardamos a observacao mais recente para, no fim do episodio, construir o BD e calcular objective/novelty.
        self._end_obs = self._ultima_obs

    # ----------------- política (heurística parametrizada) -----------------

    def _random_weights(self) -> List[float]:
        """
        Pesos da politica:
        - axis_bias: preferencia por mover no eixo X ou Y quando ha escolha
        - block_bias: penalizacao por tentar direcoes bloqueadas
        - backtrack_penalty: penalizacao por voltar ao estado de há 2 passos
        - explore_noise: ruido extra quando o episodio e exploratorio
        """
        return [
            self.rng.uniform(-1.0, 1.0),
            self.rng.uniform(-1.0, 1.0),
            self.rng.uniform(-1.0, 1.0),
            self.rng.uniform(0.0, 1.0),
        ]

    def _mutate(self, w: List[float]) -> List[float]:
    #Pequena mutacao gaussiana para gerar uma politica ‘vizinha’ de uma elite.
        return [x + self.rng.gauss(0.0, self.cfg.sigma) for x in w]

    def _select_next_policy(self) -> List[float]:

        #Escolhe a politica do proximo episodio: com alguma probabilidade, gera totalmente aleatoria, caso contrario, escolhe uma elite e aplica mutacao

        if self.rng.random() < self.cfg.random_policy_prob or not self.elites:
            return self._random_weights()
        _, _, base_w = self.rng.choice(self.elites)
        return self._mutate(base_w)

    def _policy_action(self, obs: dict) -> Action:

        # Fonte de verdade: estado interno (e fallback para obs, por segurança)
        carrying = int(getattr(self, "carrying", obs.get("carrying", 0)))

        food_dx = int(obs.get("food_dx", 0))
        food_dy = int(obs.get("food_dy", 0))
        nest_dx = int(obs.get("nest_dx", 0))
        nest_dy = int(obs.get("nest_dy", 0))

        up_block = int(obs.get("cell_0_-1", 0))
        down_block = int(obs.get("cell_0_1", 0))
        left_block = int(obs.get("cell_-1_0", 0))
        right_block = int(obs.get("cell_1_0", 0))

        # Se estiver a transportar -> alvo e o ninho; senao -> alvo e comida
        tx, ty = (nest_dx, nest_dy) if carrying else (food_dx, food_dy)

        axis_bias, block_bias, back_pen, explore_noise = self.weights[:4]
        axis_bias = max(-1.0, min(1.0, axis_bias))
        block_bias = max(0.0, min(2.0, 1.0 + block_bias))
        back_pen = max(0.0, min(3.0, 1.5 + abs(back_pen)))
        explore_noise = max(0.0, min(1.0, abs(explore_noise)))

        def is_free(a: Action) -> bool:
            if a == Action.UP:
                return up_block == 0
            if a == Action.DOWN:
                return down_block == 0
            if a == Action.LEFT:
                return left_block == 0
            return right_block == 0

        def next_pos(cur, a):
            if cur is None:
                return None
            x, y = cur
            if a == Action.UP:
                return (x, y - 1)
            if a == Action.DOWN:
                return (x, y + 1)
            if a == Action.LEFT:
                return (x - 1, y)
            return (x + 1, y)

        cur_pos = obs.get("agent", None)

        def score(a: Action) -> float:
            s = 0.0

            if not is_free(a):
                s -= block_bias

            # Alinhamento com a direcao sugerida pelos sensores
            if tx == 1 and a == Action.RIGHT:
                s += 1.0
            if tx == -1 and a == Action.LEFT:
                s += 1.0
            if ty == 1 and a == Action.DOWN:
                s += 1.0
            if ty == -1 and a == Action.UP:
                s += 1.0

            # Preferencia de eixo
            prefer_x = abs(tx) >= abs(ty)
            if axis_bias > 0.3:
                prefer_x = True
            elif axis_bias < -0.3:
                prefer_x = False

            if prefer_x:
                if tx == 1 and a == Action.RIGHT:
                    s += 0.5
                if tx == -1 and a == Action.LEFT:
                    s += 0.5
            else:
                if ty == 1 and a == Action.DOWN:
                    s += 0.5
                if ty == -1 and a == Action.UP:
                    s += 0.5

            # Anti-backtracking: evita voltar ao estado de 2 passos atras.
            npos = next_pos(cur_pos, a)
            if self._prev_prev_pos is not None and npos == self._prev_prev_pos:
                s -= back_pen

            # Ruido so em episodios exploratorios
            if self._episode_explore:
                s += self.rng.uniform(-explore_noise, explore_noise)

            return s

        free_actions = [a for a in self.actions if is_free(a)]
        if not free_actions:
            return self.rng.choice(self.actions)

        ranked = sorted(((score(a), a) for a in free_actions), key=lambda x: x[0], reverse=True)
        best = ranked[0][1]

        # Em episodios exploratorios, as vezes escolhemos a 2ª melhor opcao
        if self._episode_explore and len(ranked) > 1:
            if self.rng.random() < (0.35 + 0.35 * explore_noise):
                return ranked[1][1]

        return best

    def _behavior_descriptor(self, obs_end: dict, steps: int) -> Tuple[float, ...]:
        """
        Descritor comportamental (BD) usado para novelty.

        Usamos um resumo simples do fim do episodio:
        - deposited/collected (qualidade do comportamento)
        - steps normalizado (eficiencia)
        - distancia ao ninho no fim (se “acabou perdido” ou perto do objetivo)
        """
        deposited = float(obs_end.get("deposited", 0))
        collected = float(obs_end.get("collected", 0))

        ndx = float(obs_end.get("nest_dx", 0))
        ndy = float(obs_end.get("nest_dy", 0))
        dist_nest = abs(ndx) + abs(ndy)
        dist_nest_norm = dist_nest / 14.0  # normalização para grid 8x8 (máx ~14)

        steps_norm = float(steps) / 150.0
        return (deposited, collected, steps_norm, dist_nest_norm)

    def _dist(self, a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    def _novelty_score(self, bd: Tuple[float, ...]) -> float:
        """
        Novelty = media da distância aos k vizinhos mais proximos (arquivo + elites).
        Se ainda nao ha historico, devolvemos um valor alto para “seed” inicial do processo.
        """
        pool = [x[0] for x in self.archive] + [e[1] for e in self.elites]
        if not pool:
            return 999.0
        dists = sorted(self._dist(bd, other) for other in pool)
        k = min(self.cfg.k, len(dists))
        return sum(dists[:k]) / k

    def _update_elites(self, nov: float, bd: Tuple[float, ...], w: List[float]) -> None:
        #Mantem uma lista curta das politicas mais 'novas' (por novelty).
        self.elites.append((nov, bd, w[:]))
        self.elites.sort(key=lambda x: x[0], reverse=True)
        if len(self.elites) > self.cfg.elite_keep:
            self.elites = self.elites[: self.cfg.elite_keep]

    def _objective_score(self, obs_end: dict, steps: int) -> float:
        #Objective simples para escolher a melhor policy para TEST

        deposited = float(obs_end.get("deposited", 0))
        collected = float(obs_end.get("collected", 0))
        steps_norm = float(steps) / 150.0
        return deposited * 1000.0 + collected * 10.0 - steps_norm
