# batch_eval_foraging.py
import json
import os
import tempfile
import statistics as stats

from sim.motor_de_simulacao import MotorDeSimulacao


BASE_FIXED_JSON = "params/foraging_fixed.json"
BASE_NOVELTY_JSON = "params/foraging_novelty_test.json"

NOVELTY_POLICY_PATH = "outputs/foraging_novelty_policy.pkl"

N_SEEDS = 30
SEEDS = list(range(100, 100 + N_SEEDS))
OUT_DIR = "outputs/batch_eval"



def _run_one(base_params: dict, seed: int) -> dict:
    p = dict(base_params)
    p["seed"] = seed
    p["mode"] = "test"  # garantir test

    # escrever params temporario
    os.makedirs(OUT_DIR, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)
        tmp_path = f.name

    try:
        motor = MotorDeSimulacao.cria(tmp_path)
        motor._verbose = False  # nao spammar terminal
        summary = motor.executa()
        summary["seed"] = seed
        return summary
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _aggregate(rows: list[dict]) -> dict:
    # agregamos estas chaves
    keys = ["success_rate", "avg_steps", "avg_reward", "avg_collected", "avg_deposited"]

    out = {"n": len(rows)}
    for k in keys:
        vals = [float(r.get(k, 0.0)) for r in rows]
        out[k] = {
            "mean": stats.mean(vals) if vals else 0.0,
            "std": stats.pstdev(vals) if len(vals) > 1 else 0.0
        }
    return out


def _print_report(name: str, agg: dict):
    print(f"\n=== {name} (N={agg['n']}) ===")
    for k, v in agg.items():
        if k == "n":
            continue
        print(f"{k}: mean={v['mean']:.4f} | std={v['std']:.4f}")


def main():
    # ler bases
    with open(BASE_FIXED_JSON, "r", encoding="utf-8") as f:
        base_fixed = json.load(f)

    with open(BASE_NOVELTY_JSON, "r", encoding="utf-8") as f:
        base_novelty = json.load(f)

    # garantir que novelty aponta para a policy treinada
    base_novelty["policy_path"] = NOVELTY_POLICY_PATH

    # correr FIXED
    fixed_rows = [_run_one(base_fixed, s) for s in SEEDS]
    fixed_agg = _aggregate(fixed_rows)
    _print_report("FORAGING FIXED / TEST", fixed_agg)

    # correr NOVELTY
    novelty_rows = [_run_one(base_novelty, s) for s in SEEDS]
    novelty_agg = _aggregate(novelty_rows)
    _print_report("FORAGING NOVELTY / TEST", novelty_agg)

    # guardar resultados por-seed (csv simples)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "foraging_batch_eval.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("agent,seed,success_rate,avg_steps,avg_reward,avg_collected,avg_deposited\n")
        for r in fixed_rows:
            f.write(f"fixed,{r['seed']},{r['success_rate']},{r['avg_steps']},{r['avg_reward']},{r['avg_collected']},{r['avg_deposited']}\n")
        for r in novelty_rows:
            f.write(f"novelty,{r['seed']},{r['success_rate']},{r['avg_steps']},{r['avg_reward']},{r['avg_collected']},{r['avg_deposited']}\n")

    print(f"\n[OK] CSV batch_eval guardado em: {out_csv}")


if __name__ == "__main__":
    main()
