import json
import copy
import math
import os
from sim.motor_de_simulacao import MotorDeSimulacao

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def run_many(params_path: str, seeds, label: str):
    with open(params_path, "r", encoding="utf-8") as f:
        base = json.load(f)

    rates = []
    tmp = "_tmp_params.json"

    for s in seeds:
        cfg = copy.deepcopy(base)
        cfg["seed"] = int(s)

        with open(tmp, "w", encoding="utf-8") as w:
            json.dump(cfg, w, indent=2)

        motor = MotorDeSimulacao.cria(tmp)
        motor._verbose = False
        summary = motor.executa()
        rates.append(float(summary["success_rate"]))

    if os.path.exists(tmp):
        os.remove(tmp)

    print(f"{label}")
    print(f"n={len(seeds)} | success_rate mean={mean(rates):.4f} | std={std(rates):.4f}\n")
    return rates

if __name__ == "__main__":
    seeds = list(range(30))

    run_many("params/farol_fixed.json", seeds, "FAROL fixed (30 seeds)")
    run_many("params/farol_learning_test.json", seeds, "FAROL learning TEST (30 seeds)")
