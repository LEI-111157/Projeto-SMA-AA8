from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from sim.sensors.base import Sensor

class Agente(ABC):
    def __init__(self):
        self._sensores: List[Sensor] = []
        self._ultima_obs: dict = {}
        self._ultima_recompensa: float = 0.0

    @staticmethod
    @abstractmethod
    def cria(nome_do_ficheiro_parametros: str) -> "Agente":
        raise NotImplementedError

    def instala(self, sensor: Sensor) -> None:
        self._sensores.append(sensor)

    def observacao(self, obs: dict) -> None:
        self._ultima_obs = obs

    @abstractmethod
    def age(self):
        raise NotImplementedError

    def avaliacaoEstadoAtual(self, recompensa: float) -> None:
        self._ultima_recompensa = recompensa

    def comunica(self, mensagem: str, de_agente: "Agente") -> None:
        # para já não usamos (entra quando integrarmos MAS/FIPA na simulação)
        pass
