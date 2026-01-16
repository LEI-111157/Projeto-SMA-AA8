import json
from dataclasses import dataclass

from sim.metrics import MetricsRecorder
from sim.actions import Action

from sim.farol_ambiente import AmbienteFarol
from sim.foraging_ninho_ambiente import AmbienteForagingNinho

from sim.agente_politica_fixa import AgentePoliticaFixa
from sim.agente_Qlearning import AgenteLearning
from sim.agente_novelty import AgenteNovelty

from sim.sensors.lighthouse_direction import LighthouseDirectionSensor
from sim.sensors.distance import DistanceSensor
from sim.sensors.local_grid import LocalGridSensor
from sim.sensors.nearest_food import NearestFoodSensor
from sim.sensors.nest_direction import NestDirectionSensor


@dataclass
class Config:
    env: str = "farol"                  # "farol" | "foraging_ninho"
    agent_type: str = "fixed"           # "fixed" | "learning" | "novelty"
    mode: str = "train"                 # "train" | "test"

    width: int = 8
    height: int = 8
    obstacle_ratio: float = 0.12
    n_recursos: int = 6
    seed: int = 42

    n_episodios: int = 100
    max_passos: int = 150

    learning: dict | None = None
    qtable_path: str | None = None

    novelty: dict | None = None
    policy_path: str | None = None


class MotorDeSimulacao:
    """
    Motor que coordena o ciclo do jogo(simulação) atraves de episodios

    Responsabilidades da classe:

    -carregar parametros a partir de JSON
    -instanciar ambiente , agente e sensores
    -executar episodios tanto de train como de teste recolhendo as métricas necessárias
    -guardar CSV de resultados e artefactos da aprendizagem (Q-table no caso do farol/ policy(best) no caso do foraging)

    """

    def __init__(self, ambiente, config: Config, verbose: bool = True):
        self._ambiente = ambiente
        self._config = config
        self._metrics = MetricsRecorder()
        self._verbose = verbose

        # Para cumprir o interface pedido no enunciado (listaAgentes)
        self._agentes = []

    #Detalhe para não poluir o batch_eval com muitos outputs de grelhas.
    def _p(self, *args, **kwargs):
        if self._verbose:
            print(*args, **kwargs)

    def listaAgentes(self):
        return list(self._agentes)

    @staticmethod
    def cria(nome_do_ficheiro_parametros: str):
        cfg = Config()
        with open(nome_do_ficheiro_parametros, "r", encoding="utf-8") as f:
            data = json.load(f)

        # copiar campos conhecidos, evita rebentar se o JSON tiver campos extra.
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        #Sub-configs
        cfg.learning = data.get("learning", None)
        cfg.mode = data.get("mode", cfg.mode)
        cfg.qtable_path = data.get("qtable_path", None)
        cfg.novelty = data.get("novelty", None)
        cfg.policy_path = data.get("policy_path", None)

        #Q-learning restrito ao Farol.
        if cfg.env == "foraging_ninho" and cfg.agent_type == "learning":
            raise ValueError(
                "Q-learning (agent_type='learning') foi desativado para Foraging. "
                "Usa agent_type='fixed' ou agent_type='novelty'."
            )

        # Ambiente
        if cfg.env == "farol":
            ambiente = AmbienteFarol(cfg.width, cfg.height, cfg.obstacle_ratio, cfg.seed)
        elif cfg.env == "foraging_ninho":
            ambiente = AmbienteForagingNinho(cfg.width, cfg.height, cfg.obstacle_ratio, cfg.n_recursos, cfg.seed)
        else:
            raise ValueError(f"Ambiente desconhecido: {cfg.env}")

        return MotorDeSimulacao(ambiente, cfg)

    def _criar_agente(self):
        #Cria o agente indicado pela configuração e instala os sensores adequados ao ambiente
        if self._config.agent_type == "fixed":
            agente = AgentePoliticaFixa(seed=self._config.seed)

        elif self._config.agent_type == "learning":
            # learning (Q-learning) é restrito ao farol
            agente = AgenteLearning(
                seed=self._config.seed,
                learning=self._config.learning,
                mode=self._config.mode,
                qtable_path=self._config.qtable_path
            )

        elif self._config.agent_type == "novelty":
            agente = AgenteNovelty(
                seed=self._config.seed,
                mode=self._config.mode,
                novelty=self._config.novelty,
                policy_path=self._config.policy_path
            )

        else:
            raise ValueError(f"Agente desconhecido: {self._config.agent_type}")

        # Sensores por ambiente
        sensores = [LocalGridSensor()]
        if self._config.env == "farol":
            sensores += [LighthouseDirectionSensor(), DistanceSensor()]
        else:  # foraging_ninho
            sensores += [NearestFoodSensor(), NestDirectionSensor()]

        #agente guarda a lista de sensores
        agente._sensores = sensores
        return agente

    def executa(self):
        #Corre a simulacao recolhendo métricas por episodio
        agente = self._criar_agente()

        # Cumprir interface: manter lista de agentes no motor
        self._agentes = [agente]

        for ep_i in range(1, self._config.n_episodios + 1):
            self._ambiente.reset()

            #Agentes que precisam de reset por episodio
            if hasattr(agente, "reset_episode"):
                agente.reset_episode()

            ep = self._metrics.start_episode()

            self._p(f"\n=== EPISÓDIO {ep_i}/{self._config.n_episodios} ===")
            self._p(self._ambiente.render_text())

            for _ in range(self._config.max_passos):
                # Observa
                obs = self._ambiente.observacaoPara(agente)
                agente.observacao(obs)
                # Decide e atua
                accao: Action = agente.age()
                obs2, recompensa, terminou, info = self._ambiente.agir(accao, agente)
                # Atualiza a percecao do agente e passa a recompensa ( se o agente usar)
                agente.observacao(obs2)
                agente.avaliacaoEstadoAtual(recompensa)
                # Metricas do episodio
                ep.steps += 1
                ep.total_reward += float(recompensa)

                #Campos para o caso do foraging
                if "collected" in obs2:
                    ep.collected = int(obs2["collected"])
                if "deposited" in obs2:
                    ep.deposited = int(obs2["deposited"])

                if terminou:
                    # A condicao de sucesso e decidida pelo ambiente
                    ep.success = bool(info.get("success", terminou))
                    break

                self._ambiente.atualizacao()
            # Fecho do episodio, usado no caso do novelty para atualizar as "elites"
            if hasattr(agente, "end_episode"):
                agente.end_episode()
            # So o Q-learning usa o epsilon
            if hasattr(agente, "epsilon"):
                ep.epsilon = float(agente.epsilon)

            self._p(f"[EP {ep_i}] steps={ep.steps} | reward={ep.total_reward:.2f} | success={ep.success}")

        # Guardar CSV
        out_csv = f"outputs/{self._config.env}_{self._config.agent_type}_{self._config.mode}.csv"
        self._metrics.to_csv(out_csv)
        self._p(f"\n[CSV] Guardado em: {out_csv}")

        # Guardar Q-table no fim do treino
        if (
            self._config.agent_type == "learning"
            and self._config.mode == "train"
            and hasattr(agente, "save_q")
            and self._config.qtable_path
        ):
            agente.save_q(self._config.qtable_path)
            self._p(f"[QTABLE] Guardada em: {self._config.qtable_path}")

        # Guardar policy no fim do treino
        if (
            self._config.agent_type == "novelty"
            and self._config.mode == "train"
            and hasattr(agente, "save_policy")
            and self._config.policy_path
        ):
            agente.save_policy(self._config.policy_path)
            self._p(f"[POLICY] Guardada em: {self._config.policy_path}")

        summary = self._metrics.summary()
        self._p("\n=== SUMMARY ===")
        self._p(summary)
        return summary
