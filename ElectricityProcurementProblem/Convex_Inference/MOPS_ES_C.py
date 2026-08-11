from MOPS_Cvar import *
import multiprocessing as mp
import os


def worker(args):
    rep,discrete_feasible_region,experimental_set,d = args
    return point_inference(rep,discrete_feasible_region,experimental_set,d)

def parallel_point_inference(sys_num, discrete_feasible_region, experimental_set, d):
    with mp.Pool() as pool:
        results = pool.map(worker, [(rep,discrete_feasible_region,experimental_set,d) for rep in range(sys_num)])
    return results

if __name__ == "__main__":


    # Read the actual w1, w2 data points from the CSV file
    tail_quantile_demand=0.995
    shape_demand=2.0
    scale_demand=2.0
    percentail_demand=stats.gamma.ppf(tail_quantile_demand, shape_demand, scale_demand)
    w1_values = np.linspace(0, percentail_demand, 101)
    w2_values = np.linspace(0, percentail_demand, 101)

    # Generate feasible region
    W1, W2 = np.meshgrid(w1_values, w2_values)
    W3 = 1 - W1 - W2
    weights = np.vstack([W1.ravel(), W2.ravel()]).T
    feasible_region = weights[(weights >= 0).all(axis=1) & (weights.sum(axis=1) <=percentail_demand+0.001)]


    LB_list=np.load('LB_list_R1.npy')
    UB_list=np.load('UB_list_R1.npy')
    experimental_set=np.load('experimental_set_R1.npy')

    experimental_set=experimental_set[:,:2]
    [k,s]=np.shape(experimental_set)
    obj_num=2
    d = obj_num# number of objectives

    initial_convex_model(LB_list, UB_list, experimental_set)
   





    resolution_1=100
    resolution_2=100
    scope1=percentail_demand
    scope2=percentail_demand
    step_size1=scope1/resolution_1
    step_size2=scope2/resolution_2
    start_point1=0
    start_point2=0
    
    ES_Confidence,ES_time_stats = parallel_ES_Confidence( s, k, d, experimental_set, resolution_1, resolution_2, start_point1, step_size1, start_point2, step_size2)
    np.save('ES_Confidence.npy', ES_Confidence)
    np.save('ES_time_stats.npy', ES_time_stats)

    import os
    print("Current working directory is:", os.getcwd())

    # resolution_1 = 20
    # resolution_2 = 24
    # scope1 = 0.2
    # scope2 = 6
    # step_size1 = scope1 / resolution_1
    # step_size2 = scope2 / resolution_2
    # start_point1 = -1.1
    # start_point2 = 0
    # PF_Confidence = parallel_PF_Confidence(s, k, d, experimental_set, resolution_1, resolution_2, start_point1, step_size1, start_point2, step_size2)
    # np.save('PF_Confidence.npy', PF_Confidence)