import numpy as np
import pandas as pd
import wget
import os
from datetime import datetime
from simple_strat import moving_average, profit_moving_average, load_prices

def cal_pop_fitness(pop, prices, sol_per_pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the best profitability for each iteration.
    fitness = []
    for i in range(sol_per_pop):
        fitness.append(profit_moving_average(prices, pop[i][0], pop[i][1], pop[i][2], pop[i][3]))
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, len(pop[0])))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))[0]
        max_fitness_idx = max_fitness_idx[0]
        parents[parent_num] = pop[max_fitness_idx]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%(len(parents))
        # Index of the second parent to mate.
        parent2_idx = (k+1)%len(parents)
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k] [0:crossover_point] = parents[parent1_idx][ 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k][ crossover_point:] = parents[parent2_idx][ crossover_point:]
    return offspring

def mutation(offspring_crossover, num_mutations=1):
    mutations_counter = np.uint8(len(offspring_crossover[0]) / num_mutations)
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for idx in range(len(offspring_crossover)):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            random_value = np.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx][ gene_idx] = offspring_crossover[idx][ gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover

def change_frst_col(x):
    return [int(x[0]), int(x[1]), x[2], x[3]]


def genetic_alg(prices, num_weights, sol_per_pop, num_parents_mating, num_generations):
    """
    Genetic algorithm parameters:
        Prices
        num_weights - Number of the weights we are looking to optimize
        sol_per_pop - amounts of solutions per population
        num_parents_mating - amount of parents who will mate
        num_generations - an amount of generations
    """
    #Defining the population size.
    pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
    #Creating the initial population.
    results = {}
    for i in range(2,21):
        short = np.array([int(i) for l in range(sol_per_pop)])
        long = np.random.randint(i+1, 51, sol_per_pop)
        SL = np.random.uniform(low=0.0001, high=0.2, size=sol_per_pop)
        TP = np.random.uniform(low=0.0001, high=0.2, size=sol_per_pop)
        new_population = np.concatenate([np.reshape(short, (sol_per_pop,1)), np.reshape(long, (sol_per_pop,1)), np.reshape(SL, (sol_per_pop,1)), np.reshape(TP, (sol_per_pop,1))], axis =1)
        new_population = list(map(change_frst_col, list(new_population)))
        for generation in range(num_generations):
            # Measing the fitness of each chromosome in the population.
            fitness = cal_pop_fitness(new_population, prices, sol_per_pop)

            # Selecting the best parents in the population for mating.
            parents = select_mating_pool(new_population, fitness, num_parents_mating)
            parents = list(map(change_frst_col, list(parents)))

            # Generating next generation using crossover.
            offspring_size = (pop_size[0]-len(parents), num_weights)

            offspring_crossover = crossover(parents, offspring_size)
            offspring_crossover = list(map(change_frst_col, list(offspring_crossover)))

            # Adding some variations to the offsrping using mutation.
            offspring_mutation = mutation(offspring_crossover)
            offspring_mutation = list(map(change_frst_col, list(offspring_mutation)))

            # Creating the new population based on the parents and offspring.
            new_population[0:len(parents)] = parents
            new_population[len(parents):] = offspring_mutation

            # The best result in the current iteration.
        # Getting the best solution after iterating finishing all generations.
        #At first, the fitness is calculated for each solution in the final generation.
        fitness = cal_pop_fitness(new_population, prices, sol_per_pop)
        # Then return the index of that solution corresponding to the best fitness.
        best_match_idx = np.where(fitness == np.max(fitness))[0][0]
        results[fitness[best_match_idx]] =  new_population[best_match_idx]
    return max(results.keys()), results[max(results.keys())]

def brute_force(prices, sl_tp_step):
    best_res = 0
    best_params = []
    for s in range(2, 51):
        for l in range(s+1, 51):
            for sl in np.arange(0.0001, 0.2, sl_tp_step):
                for tp in np.arange(0.0001, 0.2, sl_tp_step):
                    res = profit_moving_average(prices, s, l, sl, tp)
                    if res > best_res:
                        best_res = res
                        best_params = [s, l, sl, tp]
    return best_res, best_params


if __name__ == '__main__':
    # константы
    LINK = 'https://api.blockchain.info/charts/market-price?format=csv'

    # импорт данных
    data, prices = load_prices(LINK, 100)

    start = datetime.now()
    res, params = genetic_alg(prices, 4, 10, 4, 4) # params: [short, long, stoploss, takeprofit]
    print("Максимальная выручка: ", res, "Параметры: ", params )
    print("время исполнения: ", datetime.now() - start)






