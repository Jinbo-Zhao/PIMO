import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import griddata
from scipy.stats import qmc  
from scipy import stats
import time
import gurobipy as gp
from gurobipy import GRB
import multiprocessing as mp
import os

tail_quantile_demand=0.995
shape_demand=2.0
scale_demand=2.0
percentail_demand=stats.gamma.ppf(tail_quantile_demand, shape_demand, scale_demand)

def calculate_cvar_weighted(mu, sigma, weights, alpha):
    """
    Compute the CVaR of the weighted sum of three normally distributed noises.
    
    Parameters:
    mu -- list of noise means [mu1, mu2, mu3]
    sigma -- list of noise standard deviations [sigma1, sigma2, sigma3]
    weights -- list of weights [w1, w2, w3]
    alpha -- confidence level (e.g. 0.95)
    
    Returns:
    CVaR -- Conditional Value-at-Risk at the given confidence level
    """
    mu_X = sum(w * m for w, m in zip(weights, mu))
    sigma_X = np.sqrt(sum(w**2 * s**2 for w, s in zip(weights, sigma)))
    
    cvar_alpha = -mu_X + (sigma_X * norm.pdf(norm.ppf(alpha))) / (alpha)
    
    return cvar_alpha

def calculate_expected_return(mu, weights):
    """
    Compute the expected value of the return.
    
    Parameters:
    mu -- list of noise means [mu1, mu2, mu3]
    weights -- list of weights [w1, w2, w3]
    
    Returns:
    expected_return -- expected value of the return
    """
    expected_return = sum(w * m for w, m in zip(weights, mu))
    return expected_return

def find_pareto_frontier(input_data):
    """
    Compute the Pareto frontier.
    
    Parameters:
    input_data -- input 2D numpy array, each row is a multi-objective data point
    
    Returns:
    membership -- boolean array indicating whether each point is on the Pareto frontier
    member_value -- the set of Pareto frontier points
    """
    # Initialize the list that stores Pareto frontier points
    pareto_frontier = []

    # Iterate over every data point
    for i in range(input_data.shape[0]):
        # Current data point
        c_data = np.tile(input_data[i, :], (input_data.shape[0], 1))
        # Copy the input data
        t_data = input_data.copy()
        # Mark the current point as infinity to exclude itself
        t_data[i, :] = np.inf

        # Check whether the current point is greater than or equal to other points in all dimensions
        smaller_idx = c_data >= t_data

        # If the current point is not dominated by any other point, add it to the Pareto frontier
        if not np.any(np.sum(smaller_idx, axis=1) == input_data.shape[1]):
            pareto_frontier.append(input_data[i, :])

    # Convert to a numpy array
    pareto_frontier = np.array(pareto_frontier)

    # Build the membership boolean array indicating whether each point is on the Pareto frontier
    membership = np.array([np.any(np.all(input_data[i] == pareto_frontier, axis=1)) for i in range(input_data.shape[0])])

    return membership, pareto_frontier

def generate_points_in_triangle_with_w3(n_samples):
    """
    Generate uniformly distributed points in the lower-triangular region using Latin hypercube sampling (via scipy).
    Returns an ndarray of (w1, w2, w3), ensuring w1 + w2 + w3 = 1.
    
    Parameters:
    n_samples -- number of sample points
    
    Returns:
    weights -- ndarray containing w1, w2, w3
    """
    # Use scipy's LatinHypercube to generate n_samples (w1, w2) points in [0, 1]
    rng = np.random.default_rng(42)
    sampler = qmc.LatinHypercube(d=2,seed=rng)  # Latin hypercube sampling
    samples = sampler.random(n=n_samples)

    # Keep only the points satisfying w1 + w2 <= 1
    valid_indices = np.sum(samples, axis=1) <= 1
    valid_samples = samples[valid_indices]

    w1 = valid_samples[:, 0]
    w2 = valid_samples[:, 1]
    w3 = 1 - w1 - w2  # compute w3 from w1 and w2, ensuring w1 + w2 + w3 = 1

    # Combine w1, w2, w3 into an ndarray
    weights = np.column_stack((w1, w2, w3))

    return weights

def simulate_profits(weights, mu, sigma, n_simulations):
    """
    Run a Monte Carlo simulation for each sample point to generate returns.
    
    Parameters:
    weights -- weight array (w1, w2, w3)
    mu -- expected return of each asset
    sigma -- volatility of each asset
    n_simulations -- number of simulations
    
    Returns:
    profits -- array of generated simulated returns
    """
    # Simulate for each point
    profits = np.random.normal(loc=np.dot(weights, mu), scale=np.sqrt(np.dot(weights**2, sigma**2)), size=n_simulations)
    return profits

def calculate_cvar(profits, alpha):
    """
    Compute the CVaR and VaR of a given return distribution.
    
    Parameters:
    profits -- array of returns
    alpha -- confidence level
    
    Returns:
    VaR, CVaR -- Value-at-Risk and Conditional Value-at-Risk
    """
    sorted_profits = np.sort(profits)
    var_index = int((alpha) * len(sorted_profits))
    VaR = sorted_profits[var_index]
    CVaR = np.mean(sorted_profits[:var_index])
    return VaR, CVaR

def calculate_student_CI(data, confidence):
    """
    Compute the confidence interval of the given data using the Student's t distribution.
    
    Parameters:
    data -- 1D data list or array
    confidence -- confidence level, default 0.95
    
    Returns:
    mean -- mean of the data
    ci -- (confidence interval lower bound, confidence interval upper bound)
    """
    # Convert the data to a numpy array
    data = np.array(data)
    
    # Sample mean
    mean = np.mean(data)
    
    # Sample standard error
    sem = stats.sem(data)  # standard error
    
    # Sample size
    n = len(data)
    
    # Critical value from the t distribution
    t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)  # df = n-1 degrees of freedom
    
    # Compute the confidence interval
    margin_of_error = t_critical * sem
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    
    return mean, ci_lower, ci_upper

def calculate_confidence_intervals(weights, mu, sigma, n_simulations, inner_simulations, confidence_level,risk_quantile):
    """
    Compute the confidence intervals of the expected return and CVaR.
    
    Parameters:
    weights -- weight array (w1, w2, w3)
    mu -- expected return of each asset
    sigma -- volatility of each asset
    n_simulations -- number of simulations per point
    confidence_level -- confidence level of the confidence interval
    
    Returns:
    expected_profit, expected_profit_ci, cvar, cvar_ci -- expected profit and its CI, CVaR and its CI
    """
    
    # Repeat simulations to compute the confidence intervals
    expected_profit_samples = []
    cvar_samples = []
    
    for _ in range(n_simulations):
        sample_profits = simulate_profits(weights, mu, sigma, n_simulations)
        expected_profit_samples.append(np.mean(sample_profits))
        _, sample_cvar = calculate_cvar(sample_profits, risk_quantile)
        cvar_samples.append(sample_cvar)
    
    # Compute the confidence interval
    expected_profit,lower_bound_expected_profit,upper_bound_expected_profit=    calculate_student_CI(expected_profit_samples, confidence_level)
    expected_profit_ci = (lower_bound_expected_profit, upper_bound_expected_profit)
    expected_CVaR,lower_bound_cvar,upper_bound_cvar=calculate_student_CI(cvar_samples, confidence_level)
    cvar_ci = (lower_bound_cvar, upper_bound_cvar)
    
    return expected_profit, expected_profit_ci, expected_CVaR, cvar_ci

def initial_convex_model(LB_list, UB_list, experimental_set):
    [k,s]=np.shape(experimental_set)
    [k,d]=np.shape(LB_list)
    model = gp.Model('Plausible Screening With Functional Info')
        

    # Create variables
    m = np.empty((k + 1, d), dtype=object)
    g = np.empty((k + 1, d,s), dtype=object)
    a=  np.empty(d, dtype=object)
    for i in range(1,k+1):
            for r in range(d):
                name = f'm_{i}_{r}'
                m[i][r] = model.addVar(lb=LB_list[i-1][r], ub=UB_list[i-1][r], vtype=GRB.CONTINUOUS, name=name)

    for r in range(d):
        name = f'm_{0}_{r}'
        m[0][r] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)



    for i in range(k+1):
            for r in range(d):
                for l in range(s):
                    name = f'g_{i}_{r}_{l}'
                    g[i][r][l] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)

    for r in range(d):
            name = f'a_{r}'
            a[r] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=name)

    y = np.empty((k , d), dtype=object)
    for i in range(k):
        for r in range(d):
            name = f'y_{i}_{r}'
            y[i][r] = model.addVar(vtype=GRB.BINARY, name=name)



    model.setObjective(0, sense=GRB.MINIMIZE)

    #Pareto_Optimality Constraints
    for i in range(k):
        name = f'auxi_y_sum_{i}'
        model.addConstr(gp.quicksum(y[i][r] for r in range(d)) >= 1, name=name)
        for r in range(d):
            name = f'dominance_{i}_{r}'
            model.addConstr((m[0][r] - m[i][r]) * y[i][r] <= 0, name=name)

    #Gradient Constraints
    for l in range(s):
        name=f'gradient_sum{l}'
        model.addConstr(gp.quicksum(g[0][r][l] * a[r] for r in range(d)) == 0, name=name)
    name=f'weights_sum{l}'
    model.addConstr(gp.quicksum(a[r] for r in range(d)) == 1)

    #model.setParam('OutputFlag', 0)

    #Convexity Constraints
    for i in range(1,k+1):
        #print(i)
        for j in range(1,k+1):
            for r in range(d):
                if i != j:
                    name=f'convexity_affine{i}_{j}_{r}'
                    model.addConstr( m[i][r] - m[j][r]-gp.quicksum(g[i][r][l] * (experimental_set[i-1][l]-experimental_set[j-1][l]) for l in range(s)) <= 0, name=name)



    # x0= np.empty(s, dtype=object)
    # for l in range(s):
    #     name = f'x0_{l}'
    #     x0[l] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)


    # x = np.vstack((x0, expeimental_set))



                    
    # for i in range(k+1):
    #         for j in range(1):
    #             for r in range(d):
    #                 if i != j:
    #                     model.addConstr(m[i][r] - m[j][r]-gp.quicksum(g[i][r][l] * (x[i][l]-x[j][l]) for l in range(s)) <= 0)

    # for i in range(1):
    #         for j in range(k+1):
    #             for r in range(d):
    #                 if i != j:
    #                     model.addConstr(m[i][r] - m[j][r]-gp.quicksum(g[i][r][l] * (x[i][l]-x[j][l]) for l in range(s)) <= 0)
    model.setParam('OutputFlag', 0)
    model.optimize()
    model.write('common_part.mps')



def initial_convex_model_relaxed(LB_list, UB_list, experimental_set):
    # Checking whether the relaxation is valid
    # I feel like we did not successifully screening the M out.
    # Although it appeas in the convex constraints, I am not sure if that is enough.(To be checked.)
    [k,s]=np.shape(experimental_set)
    [k,d]=np.shape(LB_list)
    model = gp.Model('Plausible Screening With Functional Info')
        

    # Create variables
    m = np.empty((k + 1, d), dtype=object)
    g = np.empty((k + 1, d,s), dtype=object)
    a=  np.empty(d, dtype=object)
    for i in range(1,k+1):
            for r in range(d):
                name = f'm_{i}_{r}'
                m[i][r] = model.addVar(lb=LB_list[i-1][r], ub=UB_list[i-1][r], vtype=GRB.CONTINUOUS, name=name)

    for r in range(d):
        name = f'm_{0}_{r}'
        m[0][r] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)



    for i in range(k+1):
            for r in range(d):
                for l in range(s):
                    name = f'g_{i}_{r}_{l}'
                    g[i][r][l] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)

    for r in range(d):
            name = f'a_{r}'
            a[r] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=name)

    y = np.empty((k , d), dtype=object)
    for i in range(k):
        for r in range(d):
            name = f'y_{i}_{r}'
            y[i][r] = model.addVar(vtype=GRB.BINARY, name=name)



    model.setObjective(0, sense=GRB.MINIMIZE)

    #Pareto_Optimality Constraints
    for i in range(k):
        name = f'auxi_y_sum_{i}'
        model.addConstr(gp.quicksum(y[i][r] for r in range(d)) >= 1, name=name)
        for r in range(d):
            name = f'dominance_{i}_{r}'
            Dominated_corner=UB_list[i-1][r]
            model.addConstr((m[0][r] - Dominated_corner) * y[i][r] <= 0, name=name)

    #Gradient Constraints
    for l in range(s):
        name=f'gradient_sum{l}'
        model.addConstr(gp.quicksum(g[0][r][l] * a[r] for r in range(d)) == 0, name=name)
    name=f'weights_sum{l}'
    model.addConstr(gp.quicksum(a[r] for r in range(d)) == 1)

    #model.setParam('OutputFlag', 0)

    #Convexity Constraints
    for i in range(1,k+1):
        #print(i)
        for j in range(1,k+1):
            for r in range(d):
                if i != j:
                    name=f'convexity_affine{i}_{j}_{r}'
                    relaxed_LHS=LB_list[i-1][r]-UB_list[j-1][r]
                    model.addConstr( relaxed_LHS-gp.quicksum(g[i][r][l] * (experimental_set[i-1][l]-experimental_set[j-1][l]) for l in range(s)) <= 0, name=name)



    # x0= np.empty(s, dtype=object)
    # for l in range(s):
    #     name = f'x0_{l}'
    #     x0[l] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)


    # x = np.vstack((x0, expeimental_set))



                    
    # for i in range(k+1):
    #         for j in range(1):
    #             for r in range(d):
    #                 if i != j:
    #                     model.addConstr(m[i][r] - m[j][r]-gp.quicksum(g[i][r][l] * (x[i][l]-x[j][l]) for l in range(s)) <= 0)

    # for i in range(1):
    #         for j in range(k+1):
    #             for r in range(d):
    #                 if i != j:
    #                     model.addConstr(m[i][r] - m[j][r]-gp.quicksum(g[i][r][l] * (x[i][l]-x[j][l]) for l in range(s)) <= 0)
    model.setParam('OutputFlag', 0)
    model.optimize()
    model.write('common_part_relaxed.mps')

def initial_convex_model_missing(LB_list, UB_list, experimental_set):
    [k,s]=np.shape(experimental_set)
    [k,d]=np.shape(LB_list)
    model = gp.Model('Plausible Screening With Functional Info')
        

    # Create variables
    m = np.empty((k + 1, d), dtype=object)
    g = np.empty((k + 1, d,s), dtype=object)
    a=  np.empty(d, dtype=object)
    for i in range(1,k+1):
            for r in range(d):
                name = f'm_{i}_{r}'
                m[i][r] = model.addVar(lb=LB_list[i-1][r], ub=UB_list[i-1][r], vtype=GRB.CONTINUOUS, name=name)

    for r in range(d):
        name = f'm_{0}_{r}'
        m[0][r] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)



    for i in range(k+1):
            for r in range(d):
                for l in range(s):
                    name = f'g_{i}_{r}_{l}'
                    g[i][r][l] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)

    for r in range(d):
            name = f'a_{r}'
            a[r] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=name)

    y = np.empty((k , d), dtype=object)
    for i in range(k):
        for r in range(d):
            name = f'y_{i}_{r}'
            y[i][r] = model.addVar(vtype=GRB.BINARY, name=name)



    model.setObjective(0, sense=GRB.MINIMIZE)

    #Pareto_Optimality Constraints
    for i in range(k):
        name = f'auxi_y_sum_{i}'
        model.addConstr(gp.quicksum(y[i][r] for r in range(d)) >= 1, name=name)
        for r in range(d):
            name = f'dominance_{i}_{r}'
            model.addConstr((m[0][r] - m[i][r]) * y[i][r] <= 0, name=name)

    #model.setParam('OutputFlag', 0)

    #Convexity Constraints
    for i in range(1,k+1):
        #print(i)
        for j in range(1,k+1):
            for r in range(d):
                if i != j:
                    name=f'convexity_affine{i}_{j}_{r}'
                    model.addConstr( m[i][r] - m[j][r]-gp.quicksum(g[i][r][l] * (experimental_set[i-1][l]-experimental_set[j-1][l]) for l in range(s)) <= 0, name=name)



    # x0= np.empty(s, dtype=object)
    # for l in range(s):
    #     name = f'x0_{l}'
    #     x0[l] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)


    # x = np.vstack((x0, expeimental_set))



                    
    # for i in range(k+1):
    #         for j in range(1):
    #             for r in range(d):
    #                 if i != j:
    #                     model.addConstr(m[i][r] - m[j][r]-gp.quicksum(g[i][r][l] * (x[i][l]-x[j][l]) for l in range(s)) <= 0)

    # for i in range(1):
    #         for j in range(k+1):
    #             for r in range(d):
    #                 if i != j:
    #                     model.addConstr(m[i][r] - m[j][r]-gp.quicksum(g[i][r][l] * (x[i][l]-x[j][l]) for l in range(s)) <= 0)
    model.setParam('OutputFlag', 0)
    model.optimize()
    model.write('common_part_missing.mps')

def point_inference(rep,discrete_feasible_region,experimental_set,d): 
    [k,s]=np.shape(experimental_set)
    x0=discrete_feasible_region[rep]
    model_temp=gp.read('common_part.mps')
    x = np.vstack((x0, experimental_set))
    m_temp = np.empty((k + 1, d), dtype=object)
    g_temp = np.empty((k + 1, d,s), dtype=object)
    for i in range(k+1):
            for r in range(d):
                name = f'm_{i}_{r}'
                m_temp[i][r]=model_temp.getVarByName(name)

    for i in range(k+1):
            for r in range(d):
                for l in range(s):
                    name = f'g_{i}_{r}_{l}'
                    g_temp[i][r][l] = model_temp.getVarByName(name)
                    
    for i in range(k+1):
            for j in range(1):
                for r in range(d):
                    if i != j:
                        name=f'convexity_affine{i}_{j}_{r}'
                        model_temp.addConstr(m_temp[i][r] - m_temp[j][r]-gp.quicksum(g_temp[i][r][l] * (x[i][l]-x[j][l]) for l in range(s)) <= 0, name=name)

    for i in range(1):
            for j in range(k+1):
                for r in range(d):
                    if i != j:
                        name=f'convexity_affine{i}_{j}_{r}'
                        model_temp.addConstr(m_temp[i][r] - m_temp[j][r]-gp.quicksum(g_temp[i][r][l] * (x[i][l]-x[j][l]) for l in range(s)) <= 0, name=name)


    #model.setParam('BestObjStop', Cutoff)
    #model.setParam('BestBdStop',Cutoff)
    model_temp.setParam('TimeLimit', 60)
    model_temp.setParam('OutputFlag', 0)
    model_temp.optimize()
    
      # Print the optimal solution
    if model_temp.status == GRB.OPTIMAL:
        return 0
        print("The problem is feasible.")

        #chekcing_constraints_validity(m,g,a,x)  
        #Inference_peek(sample_mean_storage, decision_variable, decision_variable_target, m, k,d)
    elif model_temp.status == GRB.INFEASIBLE:
        return 1
        print("The problem is infeasible.")

    elif model_temp.status == GRB.UNBOUNDED:
        return 2
        print("The problem is unbounded.")
    else:
        return 3
        print("Optimization status:", model_temp.status)
    #obj_verified=obj_value_verification(m, sample_mean_storage, sample_size_vector, sample_cov_storage,k,d)
    #print(target_idx,'verification result is',obj_verified)
    #plot_gurobi_variables_with_pareto_frontier(m, sample_mean_storage, target_idx)


def worker_PF_Confidence_interested(args):
    obj1_tracker, obj2_tracker, s, k, d, experimental_set, start_point1, step_size1, start_point2, step_size2 = args

    model_temp = gp.read('common_part.mps')

    x0 = np.empty(s, dtype=object)
    for l in range(s):
        name = f'x0_{l}'
        x0[l] = model_temp.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)

    # Special Constraints for weights model
    model_temp.addConstr(x0[0] >= 0)
    model_temp.addConstr(x0[1] >= 0)
    model_temp.addConstr(x0[0] <= percentail_demand)
    model_temp.addConstr(x0[1] <= percentail_demand)
    model_temp.addConstr(x0[0] + x0[1] <= percentail_demand)


    x = np.vstack((x0, experimental_set))
    m_temp = np.empty((k + 1, d), dtype=object)
    g_temp = np.empty((k + 1, d, s), dtype=object)

    for i in range(k + 1):
        for r in range(d):
            m_temp[i][r] = model_temp.getVarByName(f'm_{i}_{r}')

    for i in range(k + 1):
        for r in range(d):
            for l in range(s):
                g_temp[i][r][l] = model_temp.getVarByName(f'g_{i}_{r}_{l}')

    for i in range(k + 1):
        for j in range(1):
            for r in range(d):
                if i != j:
                    model_temp.addConstr(m_temp[i][r] - m_temp[j][r] - gp.quicksum(g_temp[i][r][l] * (x[i][l] - x[j][l]) for l in range(s)) <= 0)

    for i in range(1):
        for j in range(k + 1):
            for r in range(d):
                if i != j:
                    model_temp.addConstr(m_temp[i][r] - m_temp[j][r] - gp.quicksum(g_temp[i][r][l] * (x[i][l] - x[j][l]) for l in range(s)) <= 0)

    # Adding the constraints for obj1_tracker and obj2_tracker
    model_temp.addConstr(m_temp[0][0] >= start_point1 + (obj1_tracker) * step_size1)
    model_temp.addConstr(m_temp[0][0] <= start_point1 + (obj1_tracker + 1) * step_size1)
    model_temp.addConstr(m_temp[0][1] >= start_point2 + (obj2_tracker) * step_size2)
    model_temp.addConstr(m_temp[0][1] <= start_point2 + (obj2_tracker + 1) * step_size2)
    model_temp.setParam('OutputFlag', 0)
    model_temp.setParam('TimeLimit', 90)
    start=time.perf_counter()
    model_temp.optimize()
    time_duration=time.perf_counter()-start

    if model_temp.status == GRB.OPTIMAL:
        return (obj1_tracker, obj2_tracker, time_duration,1)
    elif model_temp.status == GRB.INFEASIBLE:
        return (obj1_tracker, obj2_tracker, time_duration,0)
    elif model_temp.status == GRB.UNBOUNDED:
        return (obj1_tracker, obj2_tracker, time_duration,2)
    else:
        return (obj1_tracker, obj2_tracker, time_duration,3)

def worker_PF_Confidence(args):
    obj1_tracker, obj2_tracker, s, k, d, experimental_set, start_point1, step_size1, start_point2, step_size2 = args

    model_temp = gp.read('common_part.mps')

    x0 = np.empty(s, dtype=object)
    for l in range(s):
        name = f'x0_{l}'
        x0[l] = model_temp.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)

    # Special Constraints for weights model
    model_temp.addConstr(x0[0] >= 0)
    model_temp.addConstr(x0[1] >= 0)
    model_temp.addConstr(x0[0] <= percentail_demand)
    model_temp.addConstr(x0[1] <= percentail_demand)
    model_temp.addConstr(x0[0] + x0[1] <= percentail_demand)


    x = np.vstack((x0, experimental_set))
    m_temp = np.empty((k + 1, d), dtype=object)
    g_temp = np.empty((k + 1, d, s), dtype=object)

    for i in range(k + 1):
        for r in range(d):
            m_temp[i][r] = model_temp.getVarByName(f'm_{i}_{r}')

    for i in range(k + 1):
        for r in range(d):
            for l in range(s):
                g_temp[i][r][l] = model_temp.getVarByName(f'g_{i}_{r}_{l}')

    for i in range(k + 1):
        for j in range(1):
            for r in range(d):
                if i != j:
                    model_temp.addConstr(m_temp[i][r] - m_temp[j][r] - gp.quicksum(g_temp[i][r][l] * (x[i][l] - x[j][l]) for l in range(s)) <= 0)

    for i in range(1):
        for j in range(k + 1):
            for r in range(d):
                if i != j:
                    model_temp.addConstr(m_temp[i][r] - m_temp[j][r] - gp.quicksum(g_temp[i][r][l] * (x[i][l] - x[j][l]) for l in range(s)) <= 0)

    # Adding the constraints for obj1_tracker and obj2_tracker
    model_temp.addConstr(m_temp[0][0] >= start_point1 + (obj1_tracker) * step_size1)
    model_temp.addConstr(m_temp[0][0] <= start_point1 + (obj1_tracker + 1) * step_size1)
    model_temp.addConstr(m_temp[0][1] >= start_point2 + (obj2_tracker) * step_size2)
    model_temp.addConstr(m_temp[0][1] <= start_point2 + (obj2_tracker + 1) * step_size2)
    model_temp.setParam('OutputFlag', 0)
    model_temp.setParam('TimeLimit', 90)
    start=time.perf_counter()
    model_temp.optimize()
    time_duration=time.perf_counter()-start

    if model_temp.status == GRB.OPTIMAL:
        return (obj1_tracker, obj2_tracker, time_duration,1)
    elif model_temp.status == GRB.INFEASIBLE:
        return (obj1_tracker, obj2_tracker, time_duration,0)
    elif model_temp.status == GRB.UNBOUNDED:
        return (obj1_tracker, obj2_tracker, time_duration,2)
    else:
        return (obj1_tracker, obj2_tracker, time_duration,3)

def worker_PF_Confidence_relaxed(args):
    obj1_tracker, obj2_tracker, s, k, d, experimental_set, start_point1, step_size1, start_point2, step_size2,LB_list,UB_list = args

    model_temp = gp.read('common_part_relaxed.mps')

    x0 = np.empty(s, dtype=object)
    for l in range(s):
        name = f'x0_{l}'
        x0[l] = model_temp.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)

    # Special Constraints for weights model
    model_temp.addConstr(x0[0] >= 0)
    model_temp.addConstr(x0[1] >= 0)
    model_temp.addConstr(x0[0] <= percentail_demand)
    model_temp.addConstr(x0[1] <= percentail_demand)
    model_temp.addConstr(x0[0] + x0[1] <= percentail_demand)


    x = np.vstack((x0, experimental_set))
    m_temp = np.empty((k + 1, d), dtype=object)
    g_temp = np.empty((k + 1, d, s), dtype=object)

    for i in range(k + 1):
        for r in range(d):
            m_temp[i][r] = model_temp.getVarByName(f'm_{i}_{r}')

    for i in range(k + 1):
        for r in range(d):
            for l in range(s):
                g_temp[i][r][l] = model_temp.getVarByName(f'g_{i}_{r}_{l}')

    for i in range(k + 1):
        for j in range(1):
            for r in range(d):
                if i != j:
                    relaxed_LHS=LB_list[i-1][r]
                    model_temp.addConstr(relaxed_LHS- m_temp[j][r] - gp.quicksum(g_temp[i][r][l] * (x[i][l] - x[j][l]) for l in range(s)) <= 0)

    for i in range(1):
        for j in range(k + 1):
            for r in range(d):
                if i != j:
                    relaxed_LHS=UB_list[j-1][r]
                    model_temp.addConstr(m_temp[i][r] - relaxed_LHS - gp.quicksum(g_temp[i][r][l] * (x[i][l] - x[j][l]) for l in range(s)) <= 0)

    # Adding the constraints for obj1_tracker and obj2_tracker
    model_temp.addConstr(m_temp[0][0] >= start_point1 + (obj1_tracker) * step_size1)
    model_temp.addConstr(m_temp[0][0] <= start_point1 + (obj1_tracker + 1) * step_size1)
    model_temp.addConstr(m_temp[0][1] >= start_point2 + (obj2_tracker) * step_size2)
    model_temp.addConstr(m_temp[0][1] <= start_point2 + (obj2_tracker + 1) * step_size2)
    model_temp.setParam('OutputFlag', 0)
    model_temp.setParam('TimeLimit', 90)
    start=time.perf_counter()
    model_temp.optimize()
    time_duration=time.perf_counter()-start

    if model_temp.status == GRB.OPTIMAL:
        return (obj1_tracker, obj2_tracker, time_duration,1)
    elif model_temp.status == GRB.INFEASIBLE:
        return (obj1_tracker, obj2_tracker, time_duration,0)
    elif model_temp.status == GRB.UNBOUNDED:
        return (obj1_tracker, obj2_tracker, time_duration,2)
    else:
        return (obj1_tracker, obj2_tracker, time_duration,3)

def worker_PF_Confidence_missing(args):
    obj1_tracker, obj2_tracker, s, k, d, experimental_set, start_point1, step_size1, start_point2, step_size2 = args

    model_temp = gp.read('common_part_missing.mps')

    x0 = np.empty(s, dtype=object)
    for l in range(s):
        name = f'x0_{l}'
        x0[l] = model_temp.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)

    # Special Constraints for weights model
    model_temp.addConstr(x0[0] >= 0)
    model_temp.addConstr(x0[1] >= 0)
    model_temp.addConstr(x0[0] <= percentail_demand)
    model_temp.addConstr(x0[1] <= percentail_demand)
    model_temp.addConstr(x0[0] + x0[1] <= percentail_demand)


    x = np.vstack((x0, experimental_set))
    m_temp = np.empty((k + 1, d), dtype=object)
    g_temp = np.empty((k + 1, d, s), dtype=object)

    for i in range(k + 1):
        for r in range(d):
            m_temp[i][r] = model_temp.getVarByName(f'm_{i}_{r}')

    for i in range(k + 1):
        for r in range(d):
            for l in range(s):
                g_temp[i][r][l] = model_temp.getVarByName(f'g_{i}_{r}_{l}')

    for i in range(k + 1):
        for j in range(1):
            for r in range(d):
                if i != j:
                    model_temp.addConstr(m_temp[i][r] - m_temp[j][r] - gp.quicksum(g_temp[i][r][l] * (x[i][l] - x[j][l]) for l in range(s)) <= 0)

    for i in range(1):
        for j in range(k + 1):
            for r in range(d):
                if i != j:
                    model_temp.addConstr(m_temp[i][r] - m_temp[j][r] - gp.quicksum(g_temp[i][r][l] * (x[i][l] - x[j][l]) for l in range(s)) <= 0)

    # Adding the constraints for obj1_tracker and obj2_tracker
    model_temp.addConstr(m_temp[0][0] >= start_point1 + (obj1_tracker) * step_size1)
    model_temp.addConstr(m_temp[0][0] <= start_point1 + (obj1_tracker + 1) * step_size1)
    model_temp.addConstr(m_temp[0][1] >= start_point2 + (obj2_tracker) * step_size2)
    model_temp.addConstr(m_temp[0][1] <= start_point2 + (obj2_tracker + 1) * step_size2)
    model_temp.setParam('OutputFlag', 0)
    model_temp.setParam('TimeLimit', 90)
    start=time.perf_counter()
    model_temp.optimize()
    time_duration=time.perf_counter()-start

    if model_temp.status == GRB.OPTIMAL:
        return (obj1_tracker, obj2_tracker, time_duration,1)
    elif model_temp.status == GRB.INFEASIBLE:
        return (obj1_tracker, obj2_tracker, time_duration,0)
    elif model_temp.status == GRB.UNBOUNDED:
        return (obj1_tracker, obj2_tracker, time_duration,2)
    else:
        return (obj1_tracker, obj2_tracker, time_duration,3)
      
def parallel_PF_Confidence( s, k, d, experimental_set, resolution_1, resolution_2, start_point1, step_size1, start_point2, step_size2):
    args_list = [(obj1_tracker, obj2_tracker,  s, k, d, experimental_set, start_point1, step_size1, start_point2, step_size2)
                 for obj1_tracker in range(resolution_1)
                 for obj2_tracker in range(resolution_2)]
    
    with mp.Pool() as pool:
        results = pool.map(worker_PF_Confidence, args_list)

    # Initialize PF_Confidence as a matrix of None values
    PF_Confidence = np.empty((resolution_1, resolution_2), dtype=object)
    time_stats = np.empty((resolution_1, resolution_2), dtype=object)
    for result in results:
        obj1_tracker, obj2_tracker, time_duration,status = result
        PF_Confidence[obj1_tracker, obj2_tracker] = status
        time_stats [obj1_tracker, obj2_tracker] = time_duration
 

    return PF_Confidence,time_stats

def parallel_PF_Confidence_relaxed( s, k, d, experimental_set, resolution_1, resolution_2, start_point1, step_size1, start_point2, step_size2,LB_list,UB_list):
    args_list = [(obj1_tracker, obj2_tracker,  s, k, d, experimental_set, start_point1, step_size1, start_point2, step_size2,LB_list,UB_list)
                 for obj1_tracker in range(resolution_1)
                 for obj2_tracker in range(resolution_2)]
    
    with mp.Pool() as pool:
        results = pool.map(worker_PF_Confidence_relaxed, args_list)

    # Initialize PF_Confidence as a matrix of None values
    PF_Confidence = np.empty((resolution_1, resolution_2), dtype=object)
    time_stats = np.empty((resolution_1, resolution_2), dtype=object)
    for result in results:
        obj1_tracker, obj2_tracker, time_duration,status = result
        PF_Confidence[obj1_tracker, obj2_tracker] = status
        time_stats [obj1_tracker, obj2_tracker] = time_duration
    
    return PF_Confidence,time_stats

def parallel_PF_Confidence_missing( s, k, d, experimental_set, resolution_1, resolution_2, start_point1, step_size1, start_point2, step_size2):
    args_list = [(obj1_tracker, obj2_tracker,  s, k, d, experimental_set, start_point1, step_size1, start_point2, step_size2)
                 for obj1_tracker in range(resolution_1)
                 for obj2_tracker in range(resolution_2)]
    
    with mp.Pool() as pool:
        results = pool.map(worker_PF_Confidence_missing, args_list)

    # Initialize PF_Confidence as a matrix of None values
    PF_Confidence = np.empty((resolution_1, resolution_2), dtype=object)
    time_stats = np.empty((resolution_1, resolution_2), dtype=object)
    for result in results:
        obj1_tracker, obj2_tracker, time_duration,status = result
        PF_Confidence[obj1_tracker, obj2_tracker] = status
        time_stats [obj1_tracker, obj2_tracker] = time_duration
    return PF_Confidence,time_stats

def parallel_PF_Confidence_missing( s, k, d, experimental_set, resolution_1, resolution_2, start_point1, step_size1, start_point2, step_size2):
    args_list = [(obj1_tracker, obj2_tracker,  s, k, d, experimental_set, start_point1, step_size1, start_point2, step_size2)
                 for obj1_tracker in range(resolution_1)
                 for obj2_tracker in range(resolution_2)]
    
    with mp.Pool() as pool:
        results = pool.map(worker_PF_Confidence_interested, args_list)

    # Initialize PF_Confidence as a matrix of None values
    PF_Confidence = np.empty((resolution_1, resolution_2), dtype=object)
    time_stats = np.empty((resolution_1, resolution_2), dtype=object)
    for result in results:
        obj1_tracker, obj2_tracker, time_duration,status = result
        PF_Confidence[obj1_tracker, obj2_tracker] = status
        time_stats [obj1_tracker, obj2_tracker] = time_duration
    return PF_Confidence,time_stats

def worker_ES_Confidence(args):
    obj1_tracker, obj2_tracker, s, k, d, experimental_set, start_point1, step_size1, start_point2, step_size2 = args

    model_temp = gp.read('common_part.mps')

    x0 = np.empty(s, dtype=object)
    for l in range(s):
        name = f'x0_{l}'
        x0[l] = model_temp.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)

    x = np.vstack((x0, experimental_set))
    m_temp = np.empty((k + 1, d), dtype=object)
    g_temp = np.empty((k + 1, d, s), dtype=object)

    for i in range(k + 1):
        for r in range(d):
            m_temp[i][r] = model_temp.getVarByName(f'm_{i}_{r}')

    for i in range(k + 1):
        for r in range(d):
            for l in range(s):
                g_temp[i][r][l] = model_temp.getVarByName(f'g_{i}_{r}_{l}')

    for i in range(k + 1):
        for j in range(1):
            for r in range(d):
                if i != j:
                    model_temp.addConstr(m_temp[i][r] - m_temp[j][r] - gp.quicksum(g_temp[i][r][l] * (x[i][l] - x[j][l]) for l in range(s)) <= 0)

    for i in range(1):
        for j in range(k + 1):
            for r in range(d):
                if i != j:
                    model_temp.addConstr(m_temp[i][r] - m_temp[j][r] - gp.quicksum(g_temp[i][r][l] * (x[i][l] - x[j][l]) for l in range(s)) <= 0)

    model_temp.addConstr(x0[0] >= start_point1 + (obj1_tracker) * step_size1)
    model_temp.addConstr(x0[0] <= start_point1 + (obj1_tracker + 1) * step_size1)

    model_temp.addConstr(x0[1] >= start_point2 + (obj2_tracker) * step_size2)
    model_temp.addConstr(x0[1] <= start_point2 + (obj2_tracker + 1) * step_size2)

    # Special Constraints for weights model
    model_temp.addConstr(x0[0] >= 0)
    model_temp.addConstr(x0[1] >= 0)
    model_temp.addConstr(x0[0] <= percentail_demand)
    model_temp.addConstr(x0[1] <= percentail_demand)
    model_temp.addConstr(x0[0] + x0[1] <= percentail_demand)

    model_temp.setParam('TimeLimit', 90)

    start=time.perf_counter()
    model_temp.optimize()
    time_duration=time.perf_counter()-start

    if model_temp.status == GRB.OPTIMAL:
        return (obj1_tracker, obj2_tracker, time_duration,1)
    elif model_temp.status == GRB.INFEASIBLE:
        return (obj1_tracker, obj2_tracker, time_duration,0)
    elif model_temp.status == GRB.UNBOUNDED:
        return (obj1_tracker, obj2_tracker, time_duration,2)
    else:
        return (obj1_tracker, obj2_tracker, time_duration,3)

def worker_ES_Confidence_Interested(args):
    obj1_tracker, obj2_tracker, s, k, d, experimental_set, start_point1, step_size1, start_point2, step_size2,x_lower_bound,x_upper_bound,y_lower_bound,y_upper_bound = args

    model_temp = gp.read('common_part.mps')

    x0 = np.empty(s, dtype=object)
    for l in range(s):
        name = f'x0_{l}'
        x0[l] = model_temp.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)

    x = np.vstack((x0, experimental_set))
    m_temp = np.empty((k + 1, d), dtype=object)
    g_temp = np.empty((k + 1, d, s), dtype=object)

    for i in range(k + 1):
        for r in range(d):
            m_temp[i][r] = model_temp.getVarByName(f'm_{i}_{r}')

    for i in range(k + 1):
        for r in range(d):
            for l in range(s):
                g_temp[i][r][l] = model_temp.getVarByName(f'g_{i}_{r}_{l}')

    for i in range(k + 1):
        for j in range(1):
            for r in range(d):
                if i != j:
                    model_temp.addConstr(m_temp[i][r] - m_temp[j][r] - gp.quicksum(g_temp[i][r][l] * (x[i][l] - x[j][l]) for l in range(s)) <= 0)

    for i in range(1):
        for j in range(k + 1):
            for r in range(d):
                if i != j:
                    model_temp.addConstr(m_temp[i][r] - m_temp[j][r] - gp.quicksum(g_temp[i][r][l] * (x[i][l] - x[j][l]) for l in range(s)) <= 0)

    model_temp.addConstr(x0[0] >= start_point1 + (obj1_tracker) * step_size1)
    model_temp.addConstr(x0[0] <= start_point1 + (obj1_tracker + 1) * step_size1)

    model_temp.addConstr(x0[1] >= start_point2 + (obj2_tracker) * step_size2)
    model_temp.addConstr(x0[1] <= start_point2 + (obj2_tracker + 1) * step_size2)

    # Special Constraints for weights model
    model_temp.addConstr(x0[0] >= 0)
    model_temp.addConstr(x0[1] >= 0)
    model_temp.addConstr(x0[0] <= percentail_demand)
    model_temp.addConstr(x0[1] <= percentail_demand)
    model_temp.addConstr(x0[0] + x0[1] <= percentail_demand)
    
    model_temp.addConstr(m_temp[0][0] >= x_lower_bound)
    model_temp.addConstr(m_temp[0][0] <= x_upper_bound)
    model_temp.addConstr(m_temp[0][1] >= y_lower_bound)
    model_temp.addConstr(m_temp[0][1] <= y_upper_bound)

    model_temp.setParam('TimeLimit', 90)

    start=time.perf_counter()
    model_temp.optimize()
    time_duration=time.perf_counter()-start

    if model_temp.status == GRB.OPTIMAL:
        return (obj1_tracker, obj2_tracker, time_duration,1)
    elif model_temp.status == GRB.INFEASIBLE:
        return (obj1_tracker, obj2_tracker, time_duration,0)
    elif model_temp.status == GRB.UNBOUNDED:
        return (obj1_tracker, obj2_tracker, time_duration,2)
    else:
        return (obj1_tracker, obj2_tracker, time_duration,3)

def worker_ES_Confidence_relaxed(args):
    obj1_tracker, obj2_tracker, s, k, d, experimental_set, start_point1, step_size1, start_point2, step_size2,LB_list,UB_list = args

    model_temp = gp.read('common_part_relaxed.mps')

    x0 = np.empty(s, dtype=object)
    for l in range(s):
        name = f'x0_{l}'
        x0[l] = model_temp.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)

    x = np.vstack((x0, experimental_set))
    m_temp = np.empty((k + 1, d), dtype=object)
    g_temp = np.empty((k + 1, d, s), dtype=object)

    for i in range(k + 1):
        for r in range(d):
            m_temp[i][r] = model_temp.getVarByName(f'm_{i}_{r}')

    for i in range(k + 1):
        for r in range(d):
            for l in range(s):
                g_temp[i][r][l] = model_temp.getVarByName(f'g_{i}_{r}_{l}')

    for i in range(k + 1):
        for j in range(1):
            for r in range(d):
                if i != j:
                    relaxed_LHS=LB_list[i-1][r]
                    model_temp.addConstr(relaxed_LHS- m_temp[j][r] - gp.quicksum(g_temp[i][r][l] * (x[i][l] - x[j][l]) for l in range(s)) <= 0)

    for i in range(1):
        for j in range(k + 1):
            for r in range(d):
                if i != j:
                    relaxed_LHS=UB_list[j-1][r]
                    model_temp.addConstr(m_temp[i][r] - relaxed_LHS - gp.quicksum(g_temp[i][r][l] * (x[i][l] - x[j][l]) for l in range(s)) <= 0)

    model_temp.addConstr(x0[0] >= start_point1 + (obj1_tracker) * step_size1)
    model_temp.addConstr(x0[0] <= start_point1 + (obj1_tracker + 1) * step_size1)

    model_temp.addConstr(x0[1] >= start_point2 + (obj2_tracker) * step_size2)
    model_temp.addConstr(x0[1] <= start_point2 + (obj2_tracker + 1) * step_size2)

    # Special Constraints for weights model
    model_temp.addConstr(x0[0] >= 0)
    model_temp.addConstr(x0[1] >= 0)
    model_temp.addConstr(x0[0] <= percentail_demand)
    model_temp.addConstr(x0[1] <= percentail_demand)
    model_temp.addConstr(x0[0] + x0[1] <= percentail_demand)

    model_temp.setParam('TimeLimit', 90)

    start=time.perf_counter()
    model_temp.optimize()
    time_duration=time.perf_counter()-start

    if model_temp.status == GRB.OPTIMAL:
        return (obj1_tracker, obj2_tracker, time_duration,1)
    elif model_temp.status == GRB.INFEASIBLE:
        return (obj1_tracker, obj2_tracker, time_duration,0)
    elif model_temp.status == GRB.UNBOUNDED:
        return (obj1_tracker, obj2_tracker, time_duration,2)
    else:
        return (obj1_tracker, obj2_tracker, time_duration,3)


def worker_ES_Confidence_missing(args):
    obj1_tracker, obj2_tracker, s, k, d, experimental_set, start_point1, step_size1, start_point2, step_size2 = args

    model_temp = gp.read('common_part_missing.mps')

    x0 = np.empty(s, dtype=object)
    for l in range(s):
        name = f'x0_{l}'
        x0[l] = model_temp.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)

    x = np.vstack((x0, experimental_set))
    m_temp = np.empty((k + 1, d), dtype=object)
    g_temp = np.empty((k + 1, d, s), dtype=object)

    for i in range(k + 1):
        for r in range(d):
            m_temp[i][r] = model_temp.getVarByName(f'm_{i}_{r}')

    for i in range(k + 1):
        for r in range(d):
            for l in range(s):
                g_temp[i][r][l] = model_temp.getVarByName(f'g_{i}_{r}_{l}')

    for i in range(k + 1):
        for j in range(1):
            for r in range(d):
                if i != j:
                    model_temp.addConstr(m_temp[i][r] - m_temp[j][r] - gp.quicksum(g_temp[i][r][l] * (x[i][l] - x[j][l]) for l in range(s)) <= 0)

    for i in range(1):
        for j in range(k + 1):
            for r in range(d):
                if i != j:
                    model_temp.addConstr(m_temp[i][r] - m_temp[j][r] - gp.quicksum(g_temp[i][r][l] * (x[i][l] - x[j][l]) for l in range(s)) <= 0)

    model_temp.addConstr(x0[0] >= start_point1 + (obj1_tracker) * step_size1)
    model_temp.addConstr(x0[0] <= start_point1 + (obj1_tracker + 1) * step_size1)

    model_temp.addConstr(x0[1] >= start_point2 + (obj2_tracker) * step_size2)
    model_temp.addConstr(x0[1] <= start_point2 + (obj2_tracker + 1) * step_size2)

    # Special Constraints for weights model
    model_temp.addConstr(x0[0] >= 0)
    model_temp.addConstr(x0[1] >= 0)
    model_temp.addConstr(x0[0] <= percentail_demand)
    model_temp.addConstr(x0[1] <= percentail_demand)
    model_temp.addConstr(x0[0] + x0[1] <= percentail_demand)

    model_temp.setParam('TimeLimit', 90)

    start=time.perf_counter()
    model_temp.optimize()
    time_duration=time.perf_counter()-start

    if model_temp.status == GRB.OPTIMAL:
        return (obj1_tracker, obj2_tracker, time_duration,1)
    elif model_temp.status == GRB.INFEASIBLE:
        return (obj1_tracker, obj2_tracker, time_duration,0)
    elif model_temp.status == GRB.UNBOUNDED:
        return (obj1_tracker, obj2_tracker, time_duration,2)
    else:
        return (obj1_tracker, obj2_tracker, time_duration,3)
    
def parallel_ES_Confidence( s, k, d, experimental_set, resolution_1, resolution_2, start_point1, step_size1, start_point2, step_size2):
    args_list = [(obj1_tracker, obj2_tracker, s, k, d, experimental_set, start_point1, step_size1, start_point2, step_size2)
                 for obj1_tracker in range(resolution_1)
                 for obj2_tracker in range(resolution_2)]
    
    with mp.Pool() as pool:
        results = pool.map(worker_ES_Confidence, args_list)

    # Initialize ES_Confidence as a matrix of None values
    ES_Confidence = np.empty((resolution_1, resolution_2), dtype=object)
    Time_Stats = np.empty((resolution_1, resolution_2), dtype=object)
    for result in results:
        obj1_tracker, obj2_tracker, time_duration,status = result
        ES_Confidence[obj1_tracker, obj2_tracker] = status
        Time_Stats [obj1_tracker, obj2_tracker] = time_duration
    return ES_Confidence, Time_Stats
def parallel_ES_Confidence_Interested( s, k, d, experimental_set, resolution_1, resolution_2, start_point1, step_size1, start_point2, step_size2,x_lower_bound,x_upper_bound,y_lower_bound,y_upper_bound):
    args_list = [(obj1_tracker, obj2_tracker, s, k, d, experimental_set, start_point1, step_size1, start_point2, step_size2,x_lower_bound,x_upper_bound,y_lower_bound,y_upper_bound)
                 for obj1_tracker in range(resolution_1)
                 for obj2_tracker in range(resolution_2)]
    
    with mp.Pool() as pool:
        results = pool.map(worker_ES_Confidence_Interested, args_list)

    # Initialize ES_Confidence as a matrix of None values
    ES_Confidence = np.empty((resolution_1, resolution_2), dtype=object)
    Time_Stats = np.empty((resolution_1, resolution_2), dtype=object)
    for result in results:
        obj1_tracker, obj2_tracker, time_duration,status = result
        ES_Confidence[obj1_tracker, obj2_tracker] = status
        Time_Stats [obj1_tracker, obj2_tracker] = time_duration
    return ES_Confidence, Time_Stats

def parallel_ES_Confidence_relaxed( s, k, d, experimental_set, resolution_1, resolution_2, start_point1, step_size1, start_point2, step_size2,LB_list,UB_list):
    args_list = [(obj1_tracker, obj2_tracker, s, k, d, experimental_set, start_point1, step_size1, start_point2, step_size2,LB_list,UB_list)
                 for obj1_tracker in range(resolution_1)
                 for obj2_tracker in range(resolution_2)]
    
    with mp.Pool() as pool:
        results = pool.map(worker_ES_Confidence_relaxed, args_list)

    # Initialize ES_Confidence as a matrix of None values
    ES_Confidence = np.empty((resolution_1, resolution_2), dtype=object)
    Time_Stats = np.empty((resolution_1, resolution_2), dtype=object)
    for result in results:
        obj1_tracker, obj2_tracker, time_duration,status = result
        ES_Confidence[obj1_tracker, obj2_tracker] = status
        Time_Stats [obj1_tracker, obj2_tracker] = time_duration
    return ES_Confidence, Time_Stats

def parallel_ES_Confidence_missing( s, k, d, experimental_set, resolution_1, resolution_2, start_point1, step_size1, start_point2, step_size2):
    args_list = [(obj1_tracker, obj2_tracker, s, k, d, experimental_set, start_point1, step_size1, start_point2, step_size2)
                 for obj1_tracker in range(resolution_1)
                 for obj2_tracker in range(resolution_2)]
    
    with mp.Pool() as pool:
        results = pool.map(worker_ES_Confidence_missing, args_list)

    # Initialize ES_Confidence as a matrix of None values
    ES_Confidence = np.empty((resolution_1, resolution_2), dtype=object)
    Time_Stats = np.empty((resolution_1, resolution_2), dtype=object)
    for result in results:
        obj1_tracker, obj2_tracker, time_duration,status = result
        ES_Confidence[obj1_tracker, obj2_tracker] = status
        Time_Stats [obj1_tracker, obj2_tracker] = time_duration
    return ES_Confidence, Time_Stats