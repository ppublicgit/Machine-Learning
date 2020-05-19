import numpy as np
import matplotlib.pyplot as plt
import toolz as tl


def ga(strLen, fitnessFunc, num_epochs, **kwargs):

    def insert(newPop, fns, pop):
        return np.concatenate((newPop, np.kron(np.ones((int(np.round(fns)), 1)), pop)), axis=0)

    def fps(fitness):
        fitness = fitness / np.sum(fitness)
        fitness = 10.*fitness/fitness.max()

        j = 0
        while j < len(fitness) and np.round(fitness[j]) < 1:
            j += 1

        newPopulation = np.kron(
            np.ones((int(np.round(fitness[j])), 1)), population[j, :])

        for i in range(j+1, popSize):
            if np.round(fitness[i]) >= 1.:
                newPopulation = insert(newPopulation, fitness[i], population[i, :])

        indices = [*range(newPopulation.shape[0])]
        np.random.shuffle(indices)
        newPopulation = newPopulation[indices[:popSize], :]
        return newPopulation

    def spCrossover(pop):
        newPopulation = np.zeros(pop.shape)
        crossoverPoint = np.random.randint(0, stringLength, popSize)
        for i in range(0, popSize, 2):
            newPopulation[i, :crossoverPoint[i]] = pop[i, :crossoverPoint[i]]
            newPopulation[i+1, :crossoverPoint[i]
                          ] = pop[i+1, :crossoverPoint[i]]
            newPopulation[i, crossoverPoint[i]:] = pop[i+1, crossoverPoint[i]:]
            newPopulation[i+1, crossoverPoint[i]:] = pop[i, crossoverPoint[i]:]
        return newPopulation

    def uniformCrossover(pop):
        newPopulation = np.zeros(pop.shape)
        which = np.random.rand(popSize, stringLength)
        which1 = which >= 0.5
        for i in range(0, popSize, 2):
            newPopulation[i, :] = pop[i, :] * \
                which1[i, :] + pop[i+1, :]*(1-which1[i, :])
            newPopulation[i+1, :] = pop[i, :] * \
                (1-which1[i, :]) + pop[i+1, :]*which1[i, :]
        return newPopulation

    def mutate(pop):
        whereMutate = np.random.rand(pop.shape[0], pop.shape[1])
        pop[np.where(whereMutate < mutationProb)] = 1 - \
            pop[np.where(whereMutate < mutationProb)]
        return pop

    def elitism(oldPop, pop, fitness):
        best = np.argsort(fitness)
        best = np.squeeze(oldPop[best[-nElite:], :])
        indices = [*range(pop.shape[0])]
        np.random.shuffle(indices)
        pop = pop[indices, :]
        pop[0:nElite, :] = best
        return pop

    def tournament(oldPop, pop, fitness):
        newFitness = fitnessFunction(pop)
        for i in range(0, np.shape(pop)[0], 2):
            f = np.concatenate((fitness[i:i+2], newFitness[i:i+2]), axis=0)
            indices = np.argsort(f)
            if indices[-1] < 2 and indices[-2] < 2:
                pop[i, :] = oldPop[i, :]
                pop[i+1, :] = oldPop[i+1, :]
            elif indices[-1] < 2:
                if indices[0] >= 2:
                    pop[i+indices[0]-2, :] = oldPop[i+indices[-1]]
                else:
                    pop[i+indices[1]-2, :] = oldPop[i+indices[-1]]
            elif indices[-2] < 2:
                if indices[0] >= 2:
                    pop[i+indices[0]-2, :] = oldPop[i+indices[-2]]
                else:
                    pop[i+indices[1]-2, :] = oldPop[i+indices[-2]]
        return pop

    def run():
        nonlocal population
        bestfit = np.zeros(nEpochs)

        for i in range(nEpochs):
            fitness = fitnessFunction(population)
            fitness = fitness.astype("float")

            newPopulation = fps(fitness)

            if crossover == "sp":
                newPopulation = spCrossover(newPopulation)
            elif crossover == "un":
                newPopulation = uniformCrossover(newPopulation)

            newPopulation = mutate(newPopulation)

            if nElite > 0:
                newPopulation = elitism(population, newPopulation, fitness)

            if tnmt_flag:
                newPopulation = tournament(
                    population, newPopulation, fitness)

            population = newPopulation
            bestfit[i] = fitness.max()

            if not i % 100:
                print(f"Epoch: {i}\t Max Fitness {bestfit[i]}")

        if plot:
            plt.figure()
            plt.title("Best Fit versus Epoch")
            plt.suptitle(subtitle)
            plt.xlabel("Epoch")
            plt.ylabel("Best Fit")
            plt.plot([*range(nEpochs)], bestfit, "kx-")
            plt.show(block=False)

    def best_population():
        best_fitness = -np.inf
        best_pop = None
        for i in range(popSize):
            if fitnessFunction(population[i].reshape(1, stringLength)) > best_fitness:
                best_pop = i
        return population[best_pop].reshape(1, stringLength)

    popSize = int(kwargs.get("pop_size", 100))
    mutationProb = kwargs.get("mutation_prob", (1/strLen))
    crossover = kwargs.get("crossover", "un")
    nElite = kwargs.get("num_elite", 4)
    tnmt_flag = kwargs.get("tournametn", True)
    plot = kwargs.get("plot", False)
    subtitle = kwargs.get("subtitle", "")
    colour_map = kwargs.get("colour_map", False)
    num_colours = kwargs.get("num_colours", None)
    map_shape = kwargs.get("map_shape", None)

    stringLength = strLen
    fitnessFunction = fitnessFunc
    nEpochs = num_epochs

    if colour_map:
        if num_colours is None or map_shape is None:
            raise ValueError("Must set map shape and num colours if using colour map ga")
        fitnessFunction = tl.curry(fitnessFunc)(num_colours=num_colours, shape_map=map_shape)

    if popSize % 2:
        popSize += 1

    population = np.random.rand(popSize, stringLength)
    if not colour_map:
        population = np.where(population < 0.5, 0, 1)

    run()

    return best_population()
