from abc import ABC, abstractmethod


class Sensor(ABC):
    @abstractmethod
    def sense(self, env, agent_pos):
        raise NotImplementedError
