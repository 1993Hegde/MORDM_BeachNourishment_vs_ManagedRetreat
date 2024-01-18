from deap import creator, base, tools
from deap.base import Fitness, Toolbox
from deap.algorithms import eaMuPlusLambda
from deap.tools import (ParetoFront,Statistics,Logbook,selNSGA2)
from random import randrange
import multiprocessing
from multiprocessing import Pool
import random
import numpy as np
import sys
sys.path.insert(0, '../1-Cutler_et_al._reproduction')
from cutler_et_al_functions_and_inputs import *


def evaluate_pathway_sows(individual,pars):
    total_sows=1
    initial_state = (0,0,0,0)
    states_in_path = []
    old_state = initial_state
    states_in_path.append(initial_state)
    state_action = []
    for time_period in range(len(individual)-1):
        action = individual[time_period]
        combo = list(old_state)+[action]
        new_state = compute_new_state(old_state,action, pars)
        states_in_path.append(new_state)
        old_state = new_state
        state_action.append(combo)
    combo_final = old_state+[individual[-1]]
    state_action.append(combo_final)
    strategy_individual = np.array(state_action, dtype=object)
    discounted_cost_across_sow = []
    discounted_investment_cost_across_sow = []
    discounted_damage_cost_across_sow = []
    discounted_benefits_across_sow = []
    discounted_npv_across_sow = []
    npv_sum = 0
    benefits_sum = 0
    costs_sum = 0
    investment_costs_sum = 0
    damage_costs_sum = 0
    reliability_sum = 0 
    C_sow,nourish_cost_sow, relocate_cost_sow, damage_cost_sow = compute_cost(strategy_individual,pars)
    x_sow,V_sow,L_sow = xVL(strategy_individual,pars)
    reliability_sow=np.count_nonzero(x_sow)/pars["sim_length"]
    B_sow = compute_benefits(strategy_individual,pars)
    # B_sow_w_tax = compute_benefits(strategy_individual,pars)
    discount_factor_sow = [(1+pars["delta"])**i for i in range(pars["sim_length"])]
    individual_benefits_sow = [B_sow[i]/discount_factor_sow[i] for i in range(len(discount_factor_sow))]
    individual_costs_sow = [C_sow[i]/discount_factor_sow[i] for i in range(len(discount_factor_sow))]
    individual_investment_costs_sow = [(nourish_cost_sow[i]+relocate_cost_sow[i])/discount_factor_sow[i] for i in range(len(discount_factor_sow))]
    individual_damage_costs_sow = [damage_cost_sow[i]/discount_factor_sow[i] for i in range(len(discount_factor_sow))]
    individual_npv_sow = [(B_sow[i]-C_sow[i])/discount_factor_sow[i] for i in range(len(discount_factor_sow))]
    accumulated_costs_sow = np.cumsum(individual_costs_sow)[-1]
    accumulated_investment_costs_sow = np.cumsum(individual_investment_costs_sow)[-1]
    accumulated_damage_costs_sow = np.cumsum(individual_damage_costs_sow)[-1]
    accumulated_benefits_sow = np.cumsum(individual_benefits_sow)[-1]
    accumulated_npv_sow = np.cumsum(individual_npv_sow)[-1]
    discounted_cost_across_sow.append(accumulated_costs_sow)
    discounted_damage_cost_across_sow.append(accumulated_damage_costs_sow)
    discounted_investment_cost_across_sow.append(accumulated_investment_costs_sow)
    discounted_benefits_across_sow.append(accumulated_benefits_sow)
    discounted_npv_across_sow.append(accumulated_npv_sow)
    npv_sum += accumulated_npv_sow
    costs_sum+=accumulated_costs_sow
    benefits_sum+=accumulated_benefits_sow
    investment_costs_sum+=accumulated_investment_costs_sow
    damage_costs_sum+=accumulated_damage_costs_sow
    reliability_sum+=reliability_sow
    avg_benefits_sow = benefits_sum/total_sows
    avg_costs_sow = costs_sum/total_sows
    avg_damages_sow = damage_costs_sum/total_sows
    avg_investment_sows = investment_costs_sum/total_sows
    avg_npv_across_sow = npv_sum/total_sows
    avg_reliability = reliability_sum/total_sows
    total_discounted_costs = avg_damages_sow+avg_investment_sows
    if individual.count(2)>1:
        accumulated_npv = 0
    return avg_benefits_sow,total_discounted_costs



def genetic_algorithm(toolbox,MU, LAMBDA, CXPB, MUTPB, NGEN,verbose=True, hack=True):
    pop = toolbox.population(n=MU)
    hof = ParetoFront() # retrieve the best non dominated individuals of the evolution
    
    # Statistics created for compiling four different statistics over the generations
    stats = Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0) # axis=0: compute the statistics on each objective independently
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    if hack:
        _, logbook, all_generations, all_fitness = \
        eaMuPlusLambda_hack(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                                  halloffame=hof, verbose=verbose)

        return pop, stats, hof, logbook, all_generations, all_fitness
    else:
        _, logbook = \
        eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                                  halloffame=hof, verbose=verbose)

        return pop, stats, hof, logbook, all_generations,all_fitness
    
def varOr(population, toolbox, lambda_, cxpb, mutpb):
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))

    return offspring


def eaMuPlusLambda_hack(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    logbook = Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
        
    # --- Hack ---
    all_generations = {}
    all_fitness = {}
    # ------------

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)
            
        # --- Hack ---
        all_generations[gen] = population + offspring
        all_fitness[gen] = fitnesses
        # ------------

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)
        

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook, all_generations, all_fitness

def generate_random_combination(size_of_individual):
    combination = []
    remaining_sum = size_of_individual-1

    while remaining_sum >= 4:
        num = random.randint(4, remaining_sum)
        combination.append(num)
        remaining_sum -= num
    return combination

def get_valid_ind(icls, size_of_individual):
    retreat_location = randrange(size_of_individual)
    random_combination = generate_random_combination(size_of_individual)
    individual = []
    for element in random_combination:
        nourish_chromosome = [1]+[0]*(element-1)
        individual.append(nourish_chromosome)
    flat_list = [item for sublist in individual for item in sublist]
    flat_list.insert(retreat_location,2)
    genome = flat_list
    if len(genome)!=size_of_individual:
        diff = size_of_individual-len(genome)
        filler = [1]+[0]*(diff-1)
        genome+=filler
    return icls(genome)

def crossover_list(ind1, ind2, n_time_steps):
    random_location = [i for i in range(n_time_steps)]
    crossover_location = random.choice([i for i in range(len(random_location))])
    child1 =  ind1[:crossover_location]+ind2[crossover_location:]                  
    child2 =  ind2[:crossover_location]+ind1[crossover_location:] 
    child1 = creator.Individual(child1)
    child2 = creator.Individual(child2)
    return child1, child2

def mutate_list(individual, min_gap):
    n = len(individual)
    changed = False
    ones_positions = [i for i, x in enumerate(individual) if x == 1]
    if len(ones_positions) >= 2:
        for i in range(len(ones_positions) - 1):
            start = ones_positions[i]
            end = ones_positions[i + 1]
            gap = end - start - 1

            if gap >= min_gap + 1:  # Ensure at least min_gap + 1 elements in between
                gap_to_fill = gap - (min_gap + 1)
                if gap_to_fill > 0:
                    random_index = random.randint(start + 1, end - 1)
                    individual[random_index] =1
                    changed = True
    if individual.count(2)>1:
        while individual.count(2)>1:
            r_indices = [i for i, x in enumerate(individual) if x == 2]
            random_index = random.choice(r_indices)
            individual[random_index]=random.choice([0,1])
    k = random.randint(0, 1) # Random integer to decide whether to mutate retreat or not
    if k==1 and individual.count(2)!=0:
        replace_index = individual.index(2)
        individual[replace_index]=random.choice([0,1])
        # random_index = random.randint(0, len(individual) - 1)
        # individual[random_index] = 2  
    return individual, 


def run_genetic_algorithm_on_configuration(parameter_set, MU, LAMBDA, NGEN, CXPB, MUTPB):
    creator.create("Fitness", base.Fitness, weights=(1.0,-1.0))
    # an Individual is a list with one more attribute called fitness
    creator.create("Individual", list, fitness=creator.Fitness)
    size_of_individual = parameter_set["sim_length"]
    toolbox = base.Toolbox()
    toolbox.register("individual", get_valid_ind, creator.Individual, size_of_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_pathway_sows,pars = parameter_set)
    toolbox.register("mate", crossover_list, n_time_steps = size_of_individual)
    toolbox.register("mutate", mutate_list, min_gap = parameter_set["minInterval"])
    toolbox.register("select", selNSGA2)
    cpu_count = multiprocessing.cpu_count()
    print(f"CPU count: {cpu_count}")
    pool = multiprocessing.Pool(cpu_count)
    toolbox.register("map", pool.map)
    print("Precomputation complete --- run genetic algorithm")
    pop, stats, hof, logbook, all_generations, all_fitness = genetic_algorithm(toolbox,MU, LAMBDA, CXPB, MUTPB, NGEN,hack=True)
    pool.close()
    pool = Pool(processes=cpu_count)
    async_results = [pool.apply_async(evaluate_pathway_sows, args=(i, parameter_set)) for i in hof.items]
    hof_fitness = [ar.get() for ar in async_results]    
    # hof_fitness = pool.map(evaluate_pathway_sows, parameter_set, hof.items)
    pool.close()
    return pop, hof, hof_fitness,stats, logbook, all_generations, all_fitness