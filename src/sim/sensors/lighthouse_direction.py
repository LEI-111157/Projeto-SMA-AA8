from sim.sensors.base import Sensor


class LighthouseDirectionSensor(Sensor):
    def sense(self, env, agent_pos):
        ax, ay = agent_pos
        gx, gy = env.goal

        dx = 0
        dy = 0
        if gx > ax:
            dx = 1
        elif gx < ax:
            dx = -1

        if gy > ay:
            dy = 1
        elif gy < ay:
            dy = -1

        return {"goal_dx": dx, "goal_dy": dy}
