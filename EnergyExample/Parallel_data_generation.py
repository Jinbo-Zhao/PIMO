import numpy as np
import multiprocessing
import pandas as pd
import scipy.stats as stats
import time
import os
# Define parameters
mu = np.array([0.5, 1, 2])  # Mean list
sigma = np.array([2, 1.5, 1])      # Standard deviation list
#mu = 1 + mu
sigma = np.sqrt(sigma)

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
#feasible_region=weights

# Define simulation function
def simulate_loss(mu, sigma, weights, size, shape_demand, scale_demand):
    samples_array = np.array([np.random.lognormal(mean=np.log(mu[i]), sigma=sigma[i], size=size) for i in range(len(mu))])
    # gamma distribution
    # shape=2.0，scale=2.0
    demand = np.random.gamma(shape=shape_demand, scale=scale_demand, size=size)
    remaining_demand = np.maximum(demand - np.sum(weights), 0)
    purchase_amounts = np.vstack([
    np.full(size, weights[0]),
    np.full(size, weights[1]),
    remaining_demand])
    loss = np.sum(purchase_amounts * samples_array, axis=0)
    # weights_price_impact = np.exp(weights)-1
    # weights_price_impact=weights
    # weighted_samples = np.dot(weights_price_impact, samples_array)
    return loss


# Define CVaR calculation
def calculate_cvar_weighted(mu, sigma, weights, alpha, shape_demand, scale_demand):
    size = 10**5  # Reduced size for efficiency
    samples = simulate_loss(mu, sigma, weights, size, shape_demand, scale_demand)
    sorted_samples = np.sort(samples)  # Sort samples directly
    var_alpha = np.percentile(sorted_samples, (alpha) * 100)#0.95 quantile
    subtracted_samples=samples-var_alpha
    postive_part=np.where(subtracted_samples > 0, subtracted_samples, 0)
    cvar_alpha=(np.mean(postive_part)/(1-alpha))+var_alpha  
    cvar_variance=np.var(postive_part, ddof=1)/((1-alpha)**2)
    expected = np.mean(samples)
    variance = np.var(samples, ddof=1)
    return expected, cvar_alpha, variance,cvar_variance

risk_quantile = 0.95

# Define parallel calculation
def calculate_cvar_parallel(rep):
    weight = feasible_region[rep]
    expected_loss, cvar_loss,variance,cvar_variance = calculate_cvar_weighted(mu, sigma, weight, risk_quantile,shape_demand,scale_demand)
    return expected_loss, cvar_loss,variance,cvar_variance

# # Main program
# if __name__ == "__main__":
#     risk_quantile = 0.95
#     start_time = time.time()
#     # Parallel computation
#     with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
#         results = pool.map(calculate_cvar_parallel, range(feasible_region.shape[0]))

#     # Unpack results
#     Mean_loss, CVaR, variance,cvar_variance= zip(*results)
#     results_df = pd.DataFrame({
#     'Mean_Loss': Mean_loss,
#     'CVaR': CVaR,
#     'Variance': variance,
#     'CVaR_Variance':cvar_variance,
#     })

#     # Save the DataFrame to a CSV file
#     results_df.to_csv("cvar_results_10^7.csv", index=False)

#     print("Results saved to 'cvar_results.csv'.")
#     print(f"Time taken: {time.time() - start_time} seconds")

if __name__ == "__main__":
    risk_quantile = 0.95
    start_time = time.time()

    batch_size = 300   # Save once per batch
    save_dir = "cvar_batches_exp"
    os.makedirs(save_dir, exist_ok=True)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:  # You can change this to your core count, e.g. 12
        for start in range(0, feasible_region.shape[0], batch_size):
            end = min(start + batch_size, feasible_region.shape[0])
            print(f"Processing {start}–{end} / {feasible_region.shape[0]} ...")
            
            # Run this batch in parallel
            results = pool.map(calculate_cvar_parallel, range(start, end))
            Mean_loss, CVaR, variance, cvar_variance = zip(*results)

            # Build the batch results
            batch_df = pd.DataFrame({
                'w1': feasible_region[start:end, 0],
                'w2': feasible_region[start:end, 1],
                'Mean_Loss': Mean_loss,
                'CVaR': CVaR,
                'Variance': variance,
                'CVaR_Variance': cvar_variance
            })

            # Save this batch
            batch_file = os.path.join(save_dir, f"cvar_batch_{start}_{end}.csv")
            batch_df.to_csv(batch_file, index=False)
            print(f" Saved {batch_file}")
            print(f"Time taken: {time.time() - start_time} seconds")

    print(f"Total time: {time.time() - start_time:.1f} seconds")
    print(f" All partial files saved in: {save_dir}")