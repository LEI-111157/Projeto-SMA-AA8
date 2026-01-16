from dataclasses import dataclass
import csv
import os


@dataclass
class EpisodeStats:
    #Estatisticas de um episodio (Nota: epsilon so e relevante no Q-learning.
    steps: int = 0
    total_reward: float = 0.0
    success: bool = False
    collected: int = 0
    deposited: int = 0
    epsilon: float = -1.0  # só faz sentido em learning/train


class MetricsRecorder:
    #Recolhe metricas por episodio e exporta CSV.
    def __init__(self):
        self.episodes: list[EpisodeStats] = []

    def start_episode(self) -> EpisodeStats:
        ep = EpisodeStats()
        self.episodes.append(ep)
        return ep

    def summary(self) -> dict:
        #Resumo agregado , usado no terminal para comparar as abordagens.
        if not self.episodes:
            return {}

        n = len(self.episodes)
        succ = sum(1 for e in self.episodes if e.success)
        avg_steps = sum(e.steps for e in self.episodes) / n
        avg_reward = sum(e.total_reward for e in self.episodes) / n
        avg_collected = sum(e.collected for e in self.episodes) / n
        avg_deposited = sum(e.deposited for e in self.episodes) / n

        return {
            "episodes": n,
            "success_rate": succ / n,
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
            "avg_collected": avg_collected,
            "avg_deposited": avg_deposited,
        }

    def to_csv(self, filepath: str) -> None:
        #Exporta as metricas por episodio para o CSV
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["episode", "steps", "total_reward", "success", "collected", "deposited", "epsilon"])
            for i, e in enumerate(self.episodes, start=1):
                w.writerow([i, e.steps, e.total_reward, int(e.success), e.collected, e.deposited, e.epsilon])
