import numpy as np
from Data import reinforcement_test as RT
from Learning.reinforcement_learning import TDZeroEuro, SarsaEuro


if __name__ == "__main__":
    rew = RT.EuropeTravler["rewardMatrix"]
    tra = RT.EuropeTravler["transitionMatrix"]

    euro_soft = TDZeroEuro(rew, tra, policy="soft_max")
    euro_egreedy = TDZeroEuro(rew, tra, policy="e_greedy")
    euro_greedy = TDZeroEuro(rew, tra, policy="greedy")

    print(f"Reinforced Action Matrix TDZero Europe Soft Max:\n{euro_soft}")
    print(f"Reinforced Action Matrix TDZero Europe E-Greedy:\n{euro_egreedy}")
    print(f"Reinforced Action Matrix TDZero Europe Greedy:\n{euro_greedy}")

    sarsa_euro_soft = SarsaEuro(rew, tra, policy="soft_max")
    sarsa_euro_egreedy = SarsaEuro(rew, tra, policy="e_greedy")
    sarsa_euro_greedy = SarsaEuro(rew, tra, policy="greedy")

    print(f"Reinforced Action Matrix Sarsa Europe Soft Max:\n{sarsa_euro_soft}")
    print(f"Reinforced Action Matrix Sarsa Europe E-Greedy:\n{sarsa_euro_egreedy}")
    print(f"Reinforced Action Matrix Sarsa Europe Greedy:\n{sarsa_euro_greedy}")
