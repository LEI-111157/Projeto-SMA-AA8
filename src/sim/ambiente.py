from __future__ import annotations
from abc import ABC, abstractmethod
from sim.agente import Agente

class Ambiente(ABC):
    @abstractmethod
    def observacaoPara(self, agente: Agente) -> dict:
        raise NotImplementedError

    @abstractmethod
    def atualizacao(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def agir(self, accao, agente: Agente) -> tuple[dict, float, bool, dict]:
        raise NotImplementedError
