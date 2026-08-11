"""
Data reading module
"""

import os
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from IPython.display import display
from joblib import Parallel, delayed

# Data loading configuration
# Note: Print statements are removed to avoid spam when imported by parallel workers
# Set VERBOSE_DATA_LOADING=1 environment variable if you need debug output

DATA_DIR = "instance_data"
files = {
    "baseline_cost": os.path.join(DATA_DIR, "baseline_cost.csv"),
    "randomized_cost": os.path.join(DATA_DIR, "randomized_cost.csv"),
    "baseline_demand": os.path.join(DATA_DIR, "baseline_demand.csv"),
    "demand_distribution": os.path.join(DATA_DIR, "demand_distribution.csv"),
}

# Optional verbose mode for debugging
_VERBOSE = os.environ.get("VERBOSE_DATA_LOADING", "0") == "1"

if _VERBOSE:
    print("Current working directory:", os.getcwd())
    print("\nData directory:", os.path.abspath(DATA_DIR))
    print("\nChecking files:")
    for k, v in files.items():
        exists = "OK" if os.path.exists(v) else "NOT FOUND"
        print(f"  [{exists}] {k}: {v}")

# Read CSVs (silently in normal mode)
baseline_cost = pd.read_csv(files["baseline_cost"], index_col=0)
randomized_cost = pd.read_csv(files["randomized_cost"], index_col=0)
baseline_demand = pd.read_csv(files["baseline_demand"], index_col=0)
demand_distribution = pd.read_csv(files["demand_distribution"], index_col=0)

# Extract number of goods, warehouses, and ports
n_goods = randomized_cost.shape[0]  # Number of rows in cost matrices
n_pairs = randomized_cost.shape[1]  # Number of port-warehouse pairs
n_warehouses = demand_distribution.shape[1]  # Number of warehouses
n_ports = n_pairs // n_warehouses  # Calculate number of ports

if _VERBOSE:
    print("Loaded:")
    for name, df in {
        "baseline_cost": baseline_cost,
        "randomized_cost": randomized_cost,
        "baseline_demand": baseline_demand,
        "demand_distribution": demand_distribution,
    }.items():
        print(f"  - {name}: shape={df.shape}")
    print("Instance Parameters:")
    print(f"  - Number of goods: {n_goods}")
    print(f"  - Number of ports: {n_ports}")
    print(f"  - Number of warehouses: {n_warehouses}")
    print(f"  - Number of port-warehouse pairs: {n_pairs}")
    print(f"\nVerification: {n_ports} ports * {n_warehouses} warehouses = {n_ports * n_warehouses} pairs")

# Status code constants
STATUS_SKIPPED_ZERO_DEMAND = -100

def generate_all_realizations(
    demand_distribution_df,      # (n_goods, n_warehouses) each row is a probability vector
    baseline_demand_df,          # (n_goods, 1) Poisson means for baseline demand
    baseline_cost_df,            # (1, n_pairs) or (n_goods, n_pairs) baseline costs
    randomized_cost_df,          # (n_goods, n_pairs) exponential distribution scales
    rng=None                     # numpy RandomState or None (uses global random state)
):
    """
    Generate demand and cost realizations for all goods in a single batch (no explicit Python for-loop).

    Parameters
    ----------
    demand_distribution_df : pd.DataFrame
        Probability vectors for allocating demand across warehouses (rows = goods, cols = warehouses).
    baseline_demand_df : pd.DataFrame
        Baseline mean demand for each good (Poisson parameter, one column).
    baseline_cost_df : pd.DataFrame
        Baseline costs (either 1 × n_pairs, or n_goods × n_pairs if different per good).
    randomized_cost_df : pd.DataFrame
        Scale parameters for exponential additional costs (n_goods × n_pairs).
    rng : np.random.RandomState or None, optional
        Random number generator. If None, uses numpy's global random state.
        For reproducibility, pass np.random.RandomState(seed).
        For repeated calls with different random values, pass None or a shared RandomState instance.

    Returns
    -------
    realized_demands_df : pd.DataFrame
        Actual demand allocation across warehouses (n_goods × n_warehouses).
    realized_costs_df : pd.DataFrame
        Realized total costs for each port-warehouse pair (n_goods × n_pairs).
    """
    # Input validation
    n_goods, n_warehouses = demand_distribution_df.shape
    n_pairs = randomized_cost_df.shape[1]
    
    if baseline_demand_df.shape[0] != n_goods:
        raise ValueError(f"baseline_demand_df has {baseline_demand_df.shape[0]} rows, expected {n_goods}")
    
    if randomized_cost_df.shape[0] != n_goods:
        raise ValueError(f"randomized_cost_df has {randomized_cost_df.shape[0]} rows, expected {n_goods}")
    
    if baseline_cost_df.shape[0] not in (1, n_goods):
        raise ValueError(f"baseline_cost_df must have 1 or {n_goods} rows, got {baseline_cost_df.shape[0]}")
    
    if baseline_cost_df.shape[1] != n_pairs or randomized_cost_df.shape[1] != n_pairs:
        raise ValueError(f"Cost dataframes must have {n_pairs} columns")
    
    # Use provided rng or numpy's global random state
    if rng is None:
        rng = np.random

    # ---- Demand generation ----
    # Draw total demand for each good from a Poisson distribution
    baseline_demand = baseline_demand_df.values.ravel()      # shape (n_goods,)
    total_demands = rng.poisson(baseline_demand)             # shape (n_goods,)

    # Allocate each good's demand across warehouses using multinomial sampling
    P = demand_distribution_df.values                        # (n_goods, n_warehouses)
    realized_demands = np.stack(
        [rng.multinomial(int(total_demands[i]), P[i]) for i in range(n_goods)],
        axis=0
    )  # shape (n_goods, n_warehouses)

    # ---- Cost generation ----
    # Broadcast baseline costs if only one row is provided
    if baseline_cost_df.shape[0] == 1:
        base_costs = np.tile(baseline_cost_df.values, (n_goods, 1))  # (n_goods, n_pairs)
    else:
        base_costs = baseline_cost_df.values                          # (n_goods, n_pairs)

    # Sample additional costs from exponential distributions
    additional_costs = rng.exponential(randomized_cost_df.values)     # (n_goods, n_pairs)
    realized_costs = base_costs + additional_costs

    # ---- Convert to DataFrames ----
    realized_demands_df = pd.DataFrame(
        realized_demands, index=demand_distribution_df.index, columns=demand_distribution_df.columns
    )
    realized_costs_df = pd.DataFrame(
        realized_costs, index=randomized_cost_df.index, columns=randomized_cost_df.columns
    )

    return realized_demands_df, realized_costs_df

