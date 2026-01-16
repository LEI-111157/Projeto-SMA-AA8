from sim.sensors.base import Sensor


class LocalGridSensor(Sensor):
    def sense(self, env, agent_pos):
        ax, ay = agent_pos
        feats = {}
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                x = ax + dx
                y = ay + dy
                key = f"cell_{dx}_{dy}"
                if not (0 <= x < env.width and 0 <= y < env.height):
                    feats[key] = 1  # fora do mapa = parede
                elif (x, y) in env.obstacles:
                    feats[key] = 1  # obstáculo
                else:
                    feats[key] = 0
        return feats
