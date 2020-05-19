import numpy as np


def TDZeroEuro(rewardMatrix, transitionMatrix, **kwargs):

    def train():
        nonlocal Q
        nits = 0
        while nits < num_iterations:
            # Pick initial state
            s = np.random.randint(nStates)
            # Stop when the accepting state is reached
            while s != 5:
                a = policy_functions[policy](s)
                r = rewardMatrix[s, a]
                # For this example, new state is the chosen action
                sprime = a
                Q[s, a] += mu * (r + gamma*np.max(Q[sprime, :]) - Q[s, a])
                s = sprime
            nits += 1

    def egreedy(s):
        if np.random.rand() < epsilon:
            indices = np.where(transitionMatrix[s, :] != 0)
            pick = np.random.randint(np.shape(indices)[1])
            chosen = indices[0][pick]
        else:
            chosen = np.argmax(Q[s, :])
        return chosen

    def soft_max(s):
        probability = 100 * np.exp(Q[s, :]) / tau / np.sum(np.exp(Q[s, :])/tau)
        cum_sum = [np.sum(probability[:i]) for i in range(1, len(probability))]
        cum_sum.append(100.1)
        rand = np.random.randint(0, 100)
        for i in range(len(cum_sum)):
            if cum_sum[i] > rand:
                break
        return i

    policy_functions = {"e_greedy": egreedy,
                        "greedy": lambda s: np.argmax(Q[s, :]),
                        "soft_max": soft_max
                        }

    mu = kwargs.get("mu", 0.7)
    gamma = kwargs.get("gamma", 0.4)
    epsilon = kwargs.get("epsilon", 0.1)
    num_iterations = kwargs.get("num_iterations", 1000)
    policy = kwargs.get("policy", "e_greedy")
    tau = kwargs.get("tau", 1)

    nStates = np.shape(rewardMatrix)[0]
    nActions = np.shape(rewardMatrix)[1]

    Q = np.random.rand(nStates, nActions)*0.05
    bad_indices = np.where(transitionMatrix[:, :] == 0)

    Q[bad_indices] = 0

    train()

    return Q


def SarsaEuro(rewardMatrix, transitionMatrix, **kwargs):

    def train():
        nonlocal Q
        nits = 0
        while nits < num_iterations:
            # Pick initial state
            s = np.random.randint(nStates)
            a = policy_functions[policy](s)
            # Stop when the accepting state is reached
            while s != 5:
                r = rewardMatrix[s, a]
                # For this example, new state is the chosen action
                sprime = a
                aprime = policy_functions[policy](sprime)

                Q[s, a] += mu * (r + gamma*Q[sprime, aprime] - Q[s, a])
                s = sprime
                a = aprime
            nits += 1

    def egreedy(s):
        if np.random.rand() < epsilon:
            indices = np.where(transitionMatrix[s, :] != 0)
            pick = np.random.randint(np.shape(indices)[1])
            chosen = indices[0][pick]
        else:
            chosen = np.argmax(Q[s, :])
        return chosen

    def soft_max(s):
        probability = 100 * np.exp(Q[s, :]) / tau / np.sum(np.exp(Q[s, :])/tau)
        cum_sum = [np.sum(probability[:i]) for i in range(1, len(probability))]
        cum_sum.append(100.1)
        rand = np.random.randint(0, 100)
        for i in range(len(cum_sum)):
            if cum_sum[i] > rand:
                break
        return i

    policy_functions = {"e_greedy": egreedy,
                        "greedy": lambda s: np.argmax(Q[s, :]),
                        "soft_max": soft_max
                        }

    mu = kwargs.get("mu", 0.7)
    gamma = kwargs.get("gamma", 0.4)
    epsilon = kwargs.get("epsilon", 0.1)
    num_iterations = kwargs.get("num_iterations", 1000)
    policy = kwargs.get("policy", "e_greedy")
    tau = kwargs.get("tau", 1)

    nStates = np.shape(rewardMatrix)[0]
    nActions = np.shape(rewardMatrix)[1]

    Q = np.random.rand(nStates, nActions)*0.05
    bad_indices = np.where(transitionMatrix[:, :] == 0)
    Q[bad_indices] = 0

    train()

    return Q
