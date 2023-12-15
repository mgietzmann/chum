import random
import numpy as np
import mygrad as mg
from deap import base, creator, tools

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

bounds = [(0, 100),]

def init_from_bounds(bounds):
    return np.array([random.random() * (upper - lower) + lower for lower, upper in bounds])

sample_size = 10

toolbox = base.Toolbox()
toolbox.register("vector", init_from_bounds, bounds)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.vector, n=sample_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

params = [mg.tensor(1.0), mg.tensor(1.0)]

def model(vector):
    m, b = params
    return m * vector[0] + b

def move_vector(vector, rel_distance, bounds):
    direction = np.random.random(len(vector)) * 2 - 1
    direction = direction / np.linalg.norm(direction)
    ranges = np.array([upper - lower for lower, upper in bounds])
    distances = rel_distance * ranges
    direction = direction * distances
    new_vector = vector + direction
    for i in range(len(new_vector)):
        if new_vector[i] < bounds[i][0]:
            new_vector[i] = bounds[i][0]
        elif new_vector[i] > bounds[i][1]:
            new_vector[i] = bounds[i][1]
    return new_vector

def mutate(individual, prob, rel_distance, bounds):
    for i, vector in enumerate(individual):
        if random.random() < prob:
            individual[i] = move_vector(vector, rel_distance, bounds)
    return individual,

def evaluate(individual):
    covs = []
    for vector in individual:
        y = model(vector)
        y.backward()
        covs.append(
            np.array([
                [params[row].grad * params[col].grad for row in range(len(params))]
                for col in range(len(params))
            ])
        )
    score = np.abs(np.linalg.det(np.sum(covs, axis=(0))))
    return score,

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mutate", mutate, prob=0.5, rel_distance=0.1, bounds=bounds)

def main():
    pop = toolbox.population(n=50)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    return pop


if __name__ == "__main__":
    pop = main()
    print("Best individual is: %s\nwith fitness: %s" % (pop[0], pop[0].fitness.values))






