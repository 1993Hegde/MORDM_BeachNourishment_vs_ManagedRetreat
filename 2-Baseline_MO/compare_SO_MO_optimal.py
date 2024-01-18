import multiprocessing
import sys
sys.path.insert(0, '../1-Cutler_et_al._reproduction')
from cutler_et_al_functions_and_inputs import *
from MO_Optimization_functions import *
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import multiprocessing
import pickle
from multiprocessing import Pool

# n = 5
# nums = [0, 1, 2]
# all_possible_actions = list(list(entry) for entry in product(nums, repeat=n))
# valid_actions = [i for i in all_possible_actions if i.count(2)<2]
# print("Valid actions:",len(valid_actions))

slr_scenario=1
pars = return_input_parameters_to_model(slr_scenario)
initial_state = [0,0,0,0]
optS, x_final,v_final, L_final, C_final, B_final, accumulated_npv, strategy = solve_cutler_et_al_ddp(pars, initial_state)
nourish_always = [1]*pars["sim_length"]
do_nothing_always = [0]*pars["sim_length"]
benefits_nourish_always,costs_nourish_always = evaluate_pathway_sows(nourish_always,pars)
benefits_do_nothing_always, costs_do_nothing_always = evaluate_pathway_sows(do_nothing_always,pars)
benefits_cutler_et_al, costs_cutler_et_al = evaluate_pathway_sows(list(strategy),pars)
evaluations_per_generation = 10000
number_of_offspring = 10000
number_of_generations = 100
crossover_probability = 0.6
mutation_probability = 0.2
pop, hof,hof_fitness, stats, logbook, all_generations, all_fitness = run_genetic_algorithm_on_configuration(pars, evaluations_per_generation, number_of_offspring, number_of_generations, crossover_probability, mutation_probability)
print("#######################################")
print("HOF length:",len(hof))
print("#######################################")
# cpu_count = multiprocessing.cpu_count()
# pool = Pool(processes=cpu_count)
# async_results = [pool.apply_async(evaluate_pathway_sows, args=(i, pars)) for i in valid_actions]
# all_valid_fitness = [ar.get() for ar in async_results]    
# pool.close()
# df_all = pd.DataFrame(all_valid_fitness, columns=['Benefits', 'Costs'])

df = pd.DataFrame(columns = ["Benefits", "Costs", "Generation Number"])
for n_gen in all_fitness:
    df_ngen = pd.DataFrame(all_fitness[n_gen], columns = ["Benefits", "Costs"])
    df_ngen["Generation Number"] = n_gen
    df = df.append(df_ngen)
df_hof = pd.DataFrame(hof_fitness, columns=['Benefits', 'Costs'])
plt.scatter(df['Costs'],df['Benefits'], label='All Points GA',color="palegoldenrod", alpha=0.4)
# plt.scatter(df_all["Costs"], df_all["Benefits"], label = "All feasible action sequences", color= "blue", alpha = 0.2)
plt.scatter(df_hof["Costs"], df_hof["Benefits"], label = "2 Objective Pareto Front", color= "darkred", alpha=0.8)
plt.scatter(costs_cutler_et_al,benefits_cutler_et_al, label = "NPV maximization Cutler et al.",color="green", marker = "+")
plt.scatter(costs_nourish_always,benefits_nourish_always, label='Nourish every timestep', marker = "x", color = "k")
plt.scatter(costs_do_nothing_always,benefits_do_nothing_always, label='Do Nothing every timestep', marker = "d", color = "orange")
plt.legend(bbox_to_anchor=(0.5,-0.20), loc="upper center", borderaxespad=0, ncol=2)
plt.xlabel('Costs', fontsize=18)
plt.ylabel('Benefits', fontsize=18)
plt.tight_layout()
plt.savefig("SingleObjective-2ObjCompare.png", dpi=300)

os.chdir("50_time_steps_result")
with open('pop.pickle', 'wb') as handle:
    pickle.dump(pop, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('hof.pickle', 'wb') as handle:
    pickle.dump(hof, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('hof_fitness.pickle', 'wb') as handle:
    pickle.dump(hof_fitness, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('logbook.pickle', 'wb') as handle:
    pickle.dump(logbook, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('all_generations.pickle', 'wb') as handle:
    pickle.dump(all_generations, handle, protocol=pickle.HIGHEST_PROTOCOL)        
with open('all_fitness.pickle','wb') as handle:
    pickle.dump(all_fitness, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('optS.pickle', 'wb') as handle:
    pickle.dump(optS, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('x_final.pickle', 'wb') as handle:
    pickle.dump(x_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('v_final.pickle', 'wb') as handle:
    pickle.dump(v_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('L_final.pickle', 'wb') as handle:
    pickle.dump(L_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('C_final.pickle', 'wb') as handle:
    pickle.dump(C_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('B_final.pickle', 'wb') as handle:
    pickle.dump(B_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('accumulated_npv.pickle', 'wb') as handle:
    pickle.dump(accumulated_npv, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('strategy.pickle', 'wb') as handle:
    pickle.dump(strategy, handle, protocol=pickle.HIGHEST_PROTOCOL)
os.chdir("..")
