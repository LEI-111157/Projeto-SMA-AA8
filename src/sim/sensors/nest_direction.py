from sim.sensors.base import Sensor


class NestDirectionSensor(Sensor):
    def sense(self, env, agent_pos):
        ax, ay = agent_pos
        nx, ny = env.ninho

        dx = 0 if nx == ax else (1 if nx > ax else -1)
        dy = 0 if ny == ay else (1 if ny > ay else -1)

        return {"nest_dx": dx, "nest_dy": dy}
