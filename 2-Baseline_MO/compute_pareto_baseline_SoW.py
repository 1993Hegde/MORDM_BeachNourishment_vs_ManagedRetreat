import multiprocessing
import sys
sys.path.insert(0, '../1-Cutler_et_al._reproduction')
from cutler_et_al_functions_and_inputs import *
from MO_Optimization_functions import *


slr_scenario=1
parameter_set = return_input_parameters_to_model(slr_scenario)
evaluations_per_generation = 10000
number_of_offspring = 10000
number_of_generations = 5
crossover_probability = 0.6
mutation_probability = 0.2
pop, hof,hof_fitness, stats, logbook, all_generations, all_fitness = run_genetic_algorithm_on_configuration(parameter_set, evaluations_per_generation, number_of_offspring, number_of_generations, crossover_probability, mutation_probability)


