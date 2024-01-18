import numpy as np
import math
from scipy.stats import poisson as poisson
import matplotlib.pyplot as plt
from itertools import product
from scipy.sparse import lil_matrix
import quantecon as qe



def return_input_parameters_to_model(slr_scenario):
    '''
    This function takes in a sea level rise scenario and inputs the
    low, medium and high sea level rise.
    :param slr_Scenario <[0,1,2]>: low, medium and high SLR respectively.
    '''
    pars = {}
    if slr_scenario!=0 and slr_scenario!=1 and slr_scenario!=2 and slr_scenario!=-1:
        print("ERROR : SLR scenario incorrectly specified")
    pars["scenario"] = slr_scenario
    pars["a"] = 2.355*10**-3 # Historical sea level change (m/yr)
    if slr_scenario==0:
        pars["b"] = 0 # low sea level rise acceleration
    elif slr_scenario==1:
        pars["b"] = 0.0271*10**-3 # Intermediate sea level rise acceleration
    elif slr_scenario==2:
        pars["b"] = 0.113*10**-3 # High sea level rise acceleration
    else:
        pars["b"] = 0
        pars["a"] = 0
    '''Beach and nourishment parameters'''
    pars["x_nourished"] = 6.096 # Nourished beach width (m)
    pars["x_crit"] = 0 # Beach width nourishment trigger (m)
    pars["mu"] = 1 # (x_nourished-x_crit)/x_nourished; nourished portion of beach 
    pars["init_rate"] = -0.0792 # Historical shoreline change rate (m/yr)
    pars["theta"] = 0.1 # Exponential erosion rate
    pars["r"] = 70.04  # Slope of active profile
    pars["H"] = pars["init_rate"]+pars["r"]*pars["a"]
    '''Initial conditions'''
    pars["tau_init"] = 0 # Initial years since nourishment
    pars["v_init"] = 6690000 # Initial value at risk
    pars["x_init"] = pars["x_nourished"] # Initial beach width (m)
    '''Time parameters'''
    pars["deltaT"] = 1 # Time step (yr)
    pars["Ti"] = 2020 # Year of simulation start
    pars["sim_length"] = 50# Simulation Length (yr)
    pars["Tswitch"] = 1 # Time when relocation becomes an option
    pars["T"] = np.inf # Time horizon
    '''Expected storm induced erosion'''
    pars["lambda"] = 0.35 # Storm Frequency
    pars["m"] = 1.68 # GEV location parameter
    pars["sigma"] = 4.24 # GEV scale parameter
    pars["k"] = 0.277 # GEV shape parameter
    meanGEV = pars["m"]+pars["sigma"]*(math.gamma(1-pars["k"])-1)/pars["k"]
    p = poisson.pmf(np.arange(1,5), mu=pars["lambda"])
    pars["E"] = 0
    for n in np.arange(1,5):
        M =0
        for i in range(1,n+1):
            M = M+meanGEV/i
        pars["E"]+=0.1*p[n-1]*M # Annual expected storm induced erosion
    pars["epsilon"] = 0 #Increase in storm induced erosion with SLR
    '''Property value parameters'''
    pars["d"] = 0
    pars["alpha"] = (1+0.01)**3.2808  # Property value increase due to 1m increase in beach width
    pars["beta"] = 0.5 # Property value decrease dur to 1m increase in sea level
    pars["A"] = 669000
    pars["v_init"] = pars["A"]*(pars["alpha"]**(pars["x_init"])) #Baseline propoerty value
    pars["W"] = 5*10**5 #Non-structural value at risk
    '''Benefit and cost parameters'''
    pars["delta"] = 0.0275
    pars["eta"] = 824 # Land value ($1000/m), assumes $14/sq ft and 5470 m pf beach length
    pars["l"] = 0.22 # St. Lucie County general fund tax rate
    pars["c1"] = 12000 # fixed cost of nourishment ($1000), assumes $14 million per nourishment, c2=350
    pars["c2"] = 350 # variable cost of nourishment ($1000/m), assumes $9.55/m^3, 5470 m of beach length, and 224,000 m^3 per 6.096 m nourishment
    pars["xi"] = 0 # exponential increase in c2 as time progresses (0 means cost is autonomous)
    pars["constructionCosts"] = 0
    pars["Cfunc"] = "concave"
    pars["phi_exp"] = 5.6999 # Sea level base for proportion damaged
    pars["phi_lin"] = 61.3951
    pars["phi_conc"] = 193.8357
    pars["phi_poly"] = 3.7625
    pars["kappa"] = 1.2 # Beach width base for proportion damaged
    pars["D0"] = 5.4*10**-3 # Expected proportion damaged when width = 0 sea level =0
    '''Relocation parameters'''
    pars["relocationDelay"] = 1 # Years after decision is made that relocation occurs
    pars["rho"] = 1 # Proportion of property value spent to relocate
    '''Feasibility constraints'''
    pars["minInterval"] = 1 #Mininum amount of time between two renourishments
    '''max and min values for uncertainty and sensitivity analysis'''
    pars["x_nourishedMin"] = 0.8*pars["x_nourished"]
    pars["x_nourishedMax"] = 1.2*pars["x_nourished"]
    pars["Hmin"] = -0.2
    pars["Hmax"] = 0.2
    pars["thetaMin"] = 0.8*pars["theta"]
    pars["thetaMax"] = 1.2*pars["theta"]
    pars["rMin"] = 0.8*pars["r"]
    pars["rMax"] = 1.2*pars["r"]
    pars["Emin"] = 0.8*pars["E"]
    pars["Emax"] = 1.2*pars["E"]
    pars["dMin"] = -0.05
    pars["dMax"] = 0.05
    pars["alphaMin"] = 1
    pars["alphaMax"] = 1.2
    pars["betaMin"] = .1
    pars["betaMax"] = 1
    pars["c1Min"] = 0.8*pars["c1"]
    pars["c1Max"] = 1.2*pars["c1"]
    pars["c2Min"] = 0.8*pars["c2"]
    pars["c2Max"] = 1.2*pars["c2"]
    pars["xiMin"] = 0
    pars["xiMax"] = 0.05
    pars["etaMin"] = 0.8*pars["c1"]
    pars["etaMax"] = 1.2*pars["c1"]
    pars["kappaMin"] = 1+0.8*(pars["kappa"]-1)
    pars["kappaMax"] = 1+1.2*(pars["kappa"]-1)
    pars["D0Min"] = 0.8*pars["D0"]
    pars["D0Max"] = 1.2*pars["D0"]
    pars["deltaMin"] = 0.01
    pars["deltaMax"] = 0.07
    pars["relocationDelayMin"] = 1
    pars["relocationDelayMax"] = 10
    pars["bMin"] = 0
    pars["bMax"] = 0.113*10**-3
    pars["epsilonMin"] = 0
    pars["epsilonMax"] = 0.1
    pars["AMin"] = 0.1*pars["A"]
    pars["AMax"] = 3*pars["A"]
    pars["muMin"] = 0.5
    pars["muMax"] = 1
    pars["WMin"] = 2*10**5
    pars["TMin"] = 10
    pars["Tmax"] = 10000
    return pars

def xVL(X, pars, E=[]):
    '''
    This function takes in the model inputs and the state-action matrix and computes
    the beach-width, property valuation and sea-level rise for the enterity of the matrix.
    (States alone -actions are duplicated)
    :param X
    :param pars
    '''
    t = X[:,1]
    tau = X[:,0]
    R = X[:,2]
    nourishing = X[:,3]
    L = [pars["a"]*i + pars["b"]*(i**2+2*i*(pars["Ti"]-1992)) for i in t]
    L = [round(i,4) for i in L]
    L_int = [0.5*pars["a"]*i**2 + pars["b"]*(i**3/3+i**2*(pars["Ti"]-1992)) for i in t]
    L_int = [round(i,4) for i in L_int]
    if "E" in pars:
        E = [pars["E"]*tau[i]+pars["epsilon"]*L_int[i] for i in range(len(tau))]
        E = [round(i,4) for i in E]
    gamma_erosion = [0 for i in range(len(tau))]
    for i in range(len(tau)):
        if tau[i]>0 and t[i]>0 and t[i]>=tau[i]:
            L_t = round(pars["a"]*t[i]+pars["b"]*(t[i]**2+2*t[i]*(pars["Ti"]-1992)),4)
            L_tau = pars["a"]*(t[i]-tau[i])+pars["b"]*(((t[i]-tau[i])**2)+2*(t[i]-tau[i])*(pars["Ti"]-1992))
            if tau[i]==t[i]:
                gamma_erosion[i] = pars["r"]*L_t - pars["H"]*t[i]
            else:
                gamma_erosion[i] = pars["r"]*(L_t - L_tau) - pars["H"]*tau[i]
        else:
            gamma_erosion[i] = 0
    x = [pars["x_nourished"]*nourishing[i] + max((1-pars["mu"])*pars["x_nourished"]+pars["mu"]*math.exp(-pars["theta"]*tau[i])*pars["x_nourished"]-gamma_erosion[i] - E[i],0)*(1-nourishing[i]) for i in range(len(nourishing))]
    V = [(1 + pars["d"])**t[i] * pars["A"]*(pars["alpha"]**x[i])*(pars["beta"]**L[i]) if R[i]<=pars["relocationDelay"] else 0 for i in range(len(nourishing))]
    return x, V, L

def compute_cost(X,pars):
    '''
    This function takes in the model inputs and the state-action matrix and computes
    the overall cost, nourishment cost, relocation cost and damage cost for the enterity 
    of the matrix.(States alone -actions are duplicated)
    '''
    X= np.array(X)
    x, V, L = xVL(X,pars)
    tau = X[:,0]
    t = X[:,1]
    R = X[:,2]
    A = X[:,4]
    c2 = [pars["c2"]*(1+pars["xi"])**i for i in t]
    nourish_cost = [pars["c1"]+ c2[i]*(pars["x_nourished"]-x[i])+pars["constructionCosts"] if A[i]==1 else 0 for i in range(len(c2))]
    relocate_cost = [pars["rho"]*V[i] if R[i]==pars["relocationDelay"] else 0 for i in range(len(V))]
    if pars["Cfunc"] == "linear":
        damage_cost = [pars["D0"]*((1+L[i]*pars["phi_lin"])/(pars["kappa"]**x[i]))*(V[i]+pars["W"]*(R[i]<pars["relocationDelay"])) if R[i]<pars["relocationDelay"]+1 else pars["D0"]*((1+L[i]*pars["phi_lin"])/(pars["kappa"]**x[i]))*(V[i]) for i in range(len(L))]
    elif pars["Cfunc"] == "exponential":
        damage_cost = [pars["D0"]*((pars["phi_exp"]**L[i])/(pars["kappa"]**x[i]))*(V[i]+pars["W"]*(R[i]<pars["relocationDelay"])) if R[i]<pars["relocationDelay"]+1 else pars["D0"]*((pars["phi_exp"]**L[i])/(pars["kappa"]**x[i]))*(V[i]) for i in range(len(L))]  
    elif pars["Cfunc"] == "concave":
        damage_cost = [pars["D0"]*((1+pars["phi_conc"]*(1-math.exp(-L[i])))/(pars["kappa"]**x[i]))*(V[i]+pars["W"]*(R[i]<pars["relocationDelay"])) if R[i]<pars["relocationDelay"]+1 else pars["D0"]*((pars["phi_conc"]*(1-math.exp(-L[i])))/(pars["kappa"]**x[i]))*(V[i]) for i in range(len(L))]
    elif pars["Cfunc"] == "polynomial":
        damage_cost = [pars["D0"]*((1+L[i])**(pars["phi_poly"])/(pars["kappa"]**x[i]))*(V[i]+pars["W"]*(R[i]<pars["relocationDelay"])) if R[i]<pars["relocationDelay"]+1 else pars["D0"]*((1+L[i])**(pars["phi_poly"])/(pars["kappa"]**x[i]))*(V[i]) for i in range (len(L))]
    else:
        print("Select damage function")

    feasibility_cost = [np.inf if A[i]==1 and tau[i]<pars["minInterval"]-1 else 0 for i in range(len(A))]
    C = [nourish_cost[i] + relocate_cost[i] + damage_cost[i] + feasibility_cost[i] for i in range(len(feasibility_cost))]
    return C,nourish_cost, relocate_cost, damage_cost

def compute_benefits(X,pars):
    '''
    This function takes in the model inputs and the state-action matrix and computes
    the overall benefits (nourishment+taxes) for the enterity 
    of the matrix.(States alone - actions are duplicated)
    '''
    X = np.array(X)
    x, V, L = xVL(X,pars)
    B_beach = [pars["eta"]*x[i] for i in range(len(x))]
    B_location = [pars["l"]*V[i] for i in range(len(V))]
    B = [B_beach[i] + B_location[i] for i in range(len(B_beach))]
    # B = [B_beach[i] for i in range(len(B_beach))]
    return B

def compute_new_state(old_state,action, pars):
    '''
    Given a state and an action, this function computes the new state that the system 
    transitions to
    '''
    new_state = [0,0,0,0]
    #Increment time
    new_state[1] = old_state[1]+1
    if old_state[2]==1:
        new_state[2]=2
    if action==0:
        new_state[0] = old_state[0]+1
        new_state[2] = old_state[2]
        new_state[3] = 0
    elif action==1:
        new_state[0] = 0 
        new_state[2] = old_state[2]
        new_state[3] = 1
    elif action==2:
        new_state[0] = old_state[0]+1
        new_state[3] = 0
        if old_state[2]==0: #Planning to relocate
            new_state[2] = old_state[2]+1
        else:
            new_state[2] = old_state[2]
    if old_state[2]==1:
        new_state[2] = 2
    new_state[0] = min(new_state[0], pars["sim_length"])
    new_state[1] = min(new_state[1], pars["sim_length"])
    return new_state


def compute_transition_matrix_sparse(S,A,X,pars):
    Q = lil_matrix((len(S)*len(A),len(S)), dtype=bool)
    s_indices = []
    a_indices = []
    for s in S:
        for a in A:
            s_index = S.index(s)
            a_index = A.index(a)
            s_indices.append(s_index)
            a_indices.append(a_index)
            x_a_combo = list(s)+[a]
            x_index = np.where(np.all(X==x_a_combo,axis=1))[0][0]
            s_prime = tuple(compute_new_state(s,a,pars))
            s_prime_index = S.index(s_prime)
            Q[x_index,s_prime_index]=1
    Q = Q.tocsr()
    return Q, s_indices, a_indices

def mdpsim_inf(S0i,S,Ix,X,pars):
    '''
    '''
    Si = np.zeros(pars["sim_length"],dtype=int)
    Xi = np.zeros(pars["sim_length"],dtype=int)
    Si[0] = S0i
    Xi[0] = Ix[S0i]
    for t in range(1,pars["sim_length"]):
        S_next = compute_new_state(list(S[Si[t-1]][0:-1]),X[Xi[t-1]][-1],pars)
        Si[t] = S.index(tuple(S_next))
        Xi[t] = Ix[Si[t]]
    return Xi,Si

def solve_DDP_return_NPV_action_sequence(R,Q_sparse,pars,S,X,S0i,s_indices,a_indices):
    '''
    '''
    beta = 1/(1+pars["delta"])
    ddp = qe.markov.DiscreteDP(R, Q_sparse, beta, s_indices, a_indices)
    results = ddp.solve(method='value_iteration')
    S_array = np.array(S)
    sigma_array = np.array(results.sigma).reshape(len(results.sigma),1)
    results.Xopt = np.hstack([S_array,sigma_array])
    results.Ixopt = []
    for i in results.Xopt:
        results.Ixopt.append(np.where(np.all(X==i,axis=1))[0][0])
    results.Ixopt = np.array(results.Ixopt)
    Ix = results.Ixopt
    Xi,Si = mdpsim_inf(S0i,S,Ix,X,pars)
    optS = X[Xi]
    x_final,v_final, L_final = xVL(optS,pars)
    C_final,nourish_cost_final, relocate_cost_final, damage_cost_final = compute_cost(optS,pars)
    B_final = compute_benefits(optS,pars)
    df = [(1+pars["delta"])**i for i in range(pars["sim_length"])]
    individual_benefits = [B_final[i]/df[i] for i in range(len(df))]
    individual_costs = [C_final[i]/df[i] for i in range(len(df))]
    individual_npv = [(B_final[i]-C_final[i])/df[i] for i in range(len(df))]
    accumulated_costs = np.cumsum(individual_costs)
    accumulated_benefits = np.cumsum(individual_benefits)
    accumulated_npv = np.cumsum(individual_npv)[-1]
    strategy = optS[:,-1]
    return optS, x_final, v_final, L_final, C_final, B_final, accumulated_npv, strategy

def solve_cutler_et_al_ddp(pars, initial_state):
    A = [0,1,2]
    tau = [i for i in range(0, pars["sim_length"]+1,pars["deltaT"])]
    time = [i for i in range(0, pars["sim_length"]+1,pars["deltaT"])]
    relocation = [0,1,2]
    nourished = [0,1]
    X = list(product(tau, time, relocation, nourished,A))
    S = list(product(tau, time, relocation, nourished))
    X_pr = np.array(X)
    Q_sparse,s_indices,a_indices = compute_transition_matrix_sparse(S,A,X_pr,pars)
    x, V, L = xVL(X_pr,pars)
    C,nourish_cost, relocate_cost, damage_cost = compute_cost(X,pars)
    B = compute_benefits(X,pars)
    X = np.array(X)
    T = [i for i in range(pars["sim_length"])]
    R = np.array([B[i]-C[i] for i in range(len(B))])
    S0i = S.index(tuple(initial_state))
    n_time_steps = pars["sim_length"]
    optS, x_final, v_final, L_final, C_final, B_final, accumulated_npv, strategy = solve_DDP_return_NPV_action_sequence(R,Q_sparse,pars,S,X,S0i,s_indices,a_indices)
    return optS, x_final, v_final, L_final, C_final, B_final, accumulated_npv, strategy

def plot_results_cutler_et_al(pars,x_final,v_final,file_name="Cutler_et_al_result.png",dpi=300):
    years = [i+1 for i in range(pars["sim_length"])]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(years, x_final, 'b')
    ax2.plot(years, v_final, 'orange')
    ax1.set_xlabel("Time in Years")
    ax1.set_ylabel("Beach Width (m)")
    ax2.set_ylabel("Property Valuation ($)")
    plt.savefig(file_name,  bbox_inches="tight", dpi=dpi)
    return fig