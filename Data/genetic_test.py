import numpy as np


def fourpeaks(population):

    T = 15
    start = np.zeros((np.shape(population)[0], 1))
    finish = np.zeros((np.shape(population)[0], 1))

    fitness = np.zeros((np.shape(population)[0], 1))

    for i in range(np.shape(population)[0]):
        s = np.where(population[i, :] == 1)
        f = np.where(population[i, :] == 0)
        if np.size(s) > 0:
            start = s[0][0]
        else:
            start = 0

        if np.size(f) > 0:
            finish = np.shape(population)[1] - f[-1][-1] - 1
        else:
            finish = 0

        if start > T and finish > T:
            fitness[i] = np.maximum(start, finish)+100
        else:
            fitness[i] = np.maximum(start, finish)

    return np.squeeze(fitness)


def knapsack(population):
    maxSize = 500
    sizes = sizes = np.array([109.60, 125.48, 52.16,
                              195.55, 58.67, 61.87,
                              92.95, 93.14, 155.05,
                              110.89, 13.34, 132.49,
                              194.03, 121.29, 179.33,
                              139.02, 198.78, 192.57,
                              81.66, 128.90])

    fitness = np.sum(sizes*population, axis=1)
    fitness = np.where(fitness > maxSize, 500-2*(fitness-maxSize), fitness)

    return np.squeeze(fitness)


def create_map(pop, num_colours, shape_map):

    def convert_pop(pop):
        colour_bounds = [1/num_colours*i for i in range(1, num_colours+1)]
        colours = [chr(ord('a')+i) for i in range(num_colours)]
        colour_pop = []
        for idx in range(len(pop)):
            for cdx in range(num_colours):
                if pop[idx] <= colour_bounds[cdx]:
                    colour_pop.append(colours[cdx])
                    break
        return np.array(colour_pop)

    colour_map = convert_pop(pop)
    return np.array([colour_map[shape_map[1]*i:shape_map[1]*(i+1)] for i in range(shape_map[0])])


def map_colour(population, num_colours, shape_map):

    def count_edges(pop):
        count = 0
        pop_map = create_map(pop, num_colours, shape_map)
        for j in range(1, pop_map.shape[0]):
            for i in range(1, pop_map.shape[1]):
                if pop_map[i, j] != pop_map[i-1, j]:
                    count += 1
                if pop_map[i, j] != pop_map[i, j-1]:
                    count += 1
        if pop_map[0, 0] != pop_map[1, 0]:
            count += 1
        if pop_map[0, 0] != pop_map[0, 1]:
            count += 1
        return count

    if shape_map[0]*shape_map[1] != population.shape[1]:
        raise Exception(
            "Map shape size and population shape size do not match")

    fitness = np.zeros((population.shape[0], 1))

    for i in range(population.shape[0]):
        fitness[i] = count_edges(population[i, :])

    return np.squeeze(fitness)
