from sim.sensors.base import Sensor


class NearestFoodSensor(Sensor):
    def sense(self, env, agent_pos):
        ax, ay = agent_pos
        if not env.recursos:
            return {"food_dx": 0, "food_dy": 0, "food_dist": 0}

        # recurso mais próximo por distância Manhattan
        best = None
        best_d = 10**9
        for (fx, fy) in env.recursos:
            d = abs(fx - ax) + abs(fy - ay)
            if d < best_d:
                best_d = d
                best = (fx, fy)

        fx, fy = best
        dx = 0 if fx == ax else (1 if fx > ax else -1)
        dy = 0 if fy == ay else (1 if fy > ay else -1)
        return {"food_dx": dx, "food_dy": dy, "food_dist": best_d}
