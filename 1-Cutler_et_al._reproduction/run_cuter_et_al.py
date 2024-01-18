from cutler_et_al_functions_and_inputs import *

slr_scenario=1
pars = return_input_parameters_to_model(slr_scenario)
print(pars["minInterval"])
initial_state = [0,0,0,0]
optS, x_final,v_final, L_final, C_final, B_final, accumulated_npv, strategy = solve_cutler_et_al_ddp(pars, initial_state)
print(optS[:,:4])
print(x_final)
final_figure = plot_results_cutler_et_al(pars,x_final,v_final,"Cutler_et_al_result.png",dpi=300)



