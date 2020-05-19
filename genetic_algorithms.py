from Learning.ga import ga
from Data.genetic_test import fourpeaks, knapsack, map_colour, create_map


if __name__ == "__main__":
    pop_fourpeaks = ga(30, fourpeaks, 301, plot=True, subtitle="Four Peaks")

    pop_knapsack = ga(20, knapsack, 301, plot=True, subtitle="Knapsack")

    num_colours, map_shape = 3, (5, 5)
    pop_colour_map = ga(25, map_colour, 301, \
                        colour_map=True, num_colours=num_colours, map_shape=map_shape, \
                        plot=True, subtitle="Colour Map")

    print(f"Population Four Peaks: \n{pop_fourpeaks}")
    print(f"Fitness Four Peaks: \n{fourpeaks(pop_fourpeaks)}")

    print(f"Population Knapsack: \n{pop_knapsack}")
    print(f"Fitness Knapsack: \n{knapsack(pop_knapsack)}")

    print(f"Population Colour Map: \n{pop_colour_map}")
    print(f"Colour Map: \n{create_map(pop_colour_map.T, num_colours, map_shape)}")
    print(f"Fitness Colour Map: \n{map_colour(pop_colour_map, num_colours, map_shape)}")

    input("Press Enter to exit...")
