import csv
import sys
import matplotlib.pyplot as plt


def read_csv(path: str):
    episodes = []
    rewards = []
    success = []
    epsilon = None      # pode nao existir
    collected = None    # pode nao existir
    deposited = None    # pode nao existir

    eps_list = []
    col_list = []
    dep_list = []

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=",")

        # detetar colunas disponíveis
        fieldnames = reader.fieldnames or []
        has_epsilon = "epsilon" in fieldnames
        has_collected = "collected" in fieldnames
        has_deposited = "deposited" in fieldnames

        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["total_reward"]))
            success.append(int(row["success"]))

            if has_epsilon:
                eps_list.append(float(row.get("epsilon", 0.0)))
            if has_collected:
                col_list.append(float(row.get("collected", 0.0)))
            if has_deposited:
                dep_list.append(float(row.get("deposited", 0.0)))

    if has_epsilon:
        epsilon = eps_list
    if has_collected:
        collected = col_list
    if has_deposited:
        deposited = dep_list

    return episodes, rewards, success, epsilon, collected, deposited


def moving_avg(xs, w=20):
    out = []
    s = 0.0
    for i, x in enumerate(xs):
        s += x
        if i >= w:
            s -= xs[i - w]
        out.append(s / min(i + 1, w))
    return out


def infer_title_from_filename(path: str) -> str:
    p = path.lower()
    if "farol" in p:
        return "Farol"
    if "foraging" in p:
        return "Foraging (Ninho)"
    return "Curva de Aprendizagem"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python plot_learning_curve.py <csv>")
        sys.exit(1)

    path = sys.argv[1]
    title_prefix = infer_title_from_filename(path)

    ep, reward, success, epsilon, collected, deposited = read_csv(path)

    w = 20
    reward_ma = moving_avg(reward, w)
    success_ma = moving_avg(success, w)

    #REWARD
    plt.figure()
    plt.plot(ep, reward, alpha=0.3, label="Reward")
    plt.plot(ep, reward_ma, linewidth=2, label=f"Reward (média móvel, w={w})")
    plt.xlabel("Episódio")
    plt.ylabel("Total reward")
    plt.title(f"{title_prefix} — Curva de Aprendizagem (Reward)")
    plt.legend()
    plt.grid(True)
    plt.show()

    #SUCCESS
    plt.figure()
    plt.plot(ep, success_ma)
    plt.xlabel("Episódio")
    plt.ylabel("Success rate (média móvel)")
    plt.title(f"{title_prefix} — Success rate (média móvel, w={w})")
    plt.grid(True)
    plt.show()

    #COLLECTED / DEPOSITED
    if collected is not None or deposited is not None:
        plt.figure()

        if collected is not None:
            collected_ma = moving_avg(collected, w)
            plt.plot(ep, collected_ma, label="Collected (média móvel)")

        if deposited is not None:
            deposited_ma = moving_avg(deposited, w)
            plt.plot(ep, deposited_ma, label="Deposited (média móvel)")

        plt.xlabel("Episódio")
        plt.ylabel("Recursos (média móvel)")
        plt.title(f"{title_prefix} — Collected/Deposited (média móvel, w={w})")
        plt.legend()
        plt.grid(True)
        plt.show()

    #EPSILON
    if epsilon is not None:
        plt.figure()
        plt.plot(ep, epsilon)
        plt.xlabel("Episódio")
        plt.ylabel("Epsilon")
        plt.title(f"{title_prefix} — Evolução do epsilon")
        plt.grid(True)
        plt.show()
