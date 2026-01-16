from sim.sensors.base import Sensor


class DistanceSensor(Sensor):
    def sense(self, env, agent_pos):
        ax, ay = agent_pos
        gx, gy = env.goal
        return {"manhattan": abs(gx - ax) + abs(gy - ay)}
