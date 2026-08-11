"""
Simulation optimization module
"""

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import time
from joblib import Parallel, delayed

# Import generate_all_realizations from data_reader
from data_reader import generate_all_realizations

class MCFTemplate:
    """
    Bipartite transportation / MCF for a single commodity:
      - I sources (ports) × J sinks (warehouses) = I*J edges
      - Vars: x[i,j] >= 0 (flows), s[i] >= 0 (supplies)
      - Cons:
          (1) For each sink j:  sum_i x[i,j] = demand[j]
          (2) For each src  i:  sum_j x[i,j] - s[i] = 0
      - Obj: minimize sum_{i,j} cost[i,j] * x[i,j]

    Build once; reuse by updating RHS and objective for each commodity.
    """

    def __init__(
        self,
        port_names,
        warehouse_names,
        pair_cols_ordered=None,
        threads=0,
        log_to_console=False,
        method=-1,      # -1 auto, 1 dual simplex often best for repeated LPs
        presolve=-1     # -1 auto
    ):
        self.ports = list(port_names)
        self.whs   = list(warehouse_names)
        self.I = len(self.ports)
        self.J = len(self.whs)
        self.E = self.I * self.J

        # Canonical (port,warehouse) column order
        self.pair_cols = (
            [f"{p},{w}" for p in self.ports for w in self.whs]
            if pair_cols_ordered is None else list(pair_cols_ordered)
        )

        self._build_model(threads, log_to_console, method, presolve)

    def _build_model(self, threads, log_to_console, method, presolve):
        m = gp.Model("MCF_single_good")
        m.Params.OutputFlag = 1 if log_to_console else 0
        if threads is not None:
            m.Params.Threads = threads
        if method is not None:
            m.Params.Method = method
        if presolve is not None:
            m.Params.Presolve = presolve

        # Variables x[i,j] and s[i] in deterministic order (i outer, j inner)
        x = {(i, j): m.addVar(lb=0.0, name=f"x[{self.ports[i]}->{self.whs[j]}]")
             for i in range(self.I) for j in range(self.J)}
        s = {i: m.addVar(lb=0.0, name=f"s[{self.ports[i]}]") for i in range(self.I)}
        m.update()

        # Constraints: sink demands (RHS will be updated per commodity)
        sink_cons = {
            j: m.addConstr(gp.quicksum(x[(i, j)] for i in range(self.I)) == 0.0,
                           name=f"demand[{self.whs[j]}]")
            for j in range(self.J)
        }

        # Constraints: source supply definition
        src_cons = {
            i: m.addConstr(gp.quicksum(x[(i, j)] for j in range(self.J)) - s[i] == 0.0,
                           name=f"supply_def[{self.ports[i]}]")
            for i in range(self.I)
        }

        # Empty objective; we update per commodity
        m.setObjective(gp.LinExpr(), GRB.MINIMIZE)

        # Cache ordered var lists for fast attribute setting/reading
        self.model = m
        self.x = x
        self.s = s
        self.sink_cons = sink_cons
        self.src_cons = src_cons
        self.x_list = [x[(i, j)] for i in range(self.I) for j in range(self.J)]
        self.s_list = [s[i] for i in range(self.I)]

    def solve_one(self, costs_row: np.ndarray, demands_row: np.ndarray):
        m = self.model

        # Update sink RHS
        for j in range(self.J):
            self.sink_cons[j].RHS = float(demands_row[j])

        # Objective: c^T x   (use quicksum for max compatibility)
        m.setObjective(gp.quicksum(costs_row[k] * self.x_list[k] for k in range(self.E)), GRB.MINIMIZE)

        m.optimize()
        status = m.Status
        if status != GRB.OPTIMAL:
            return None, None, np.nan, status

        flows = np.array([v.X for v in self.x_list]).reshape(self.I, self.J)
        supplies = np.array([v.X for v in self.s_list])
        return flows, supplies, m.ObjVal, status


def solve_all_goods_gurobi(
    realized_demands_df: pd.DataFrame,    # (G, J)
    realized_costs_df: pd.DataFrame,      # (G, E) columns must be (port × warehouse) order
    port_names=None,
    warehouse_names=None,
    threads=0,
    log_to_console=False
):
    """
    Build template once; for each good update RHS and objective.
    Minimizes Python loops by preallocating arrays and vectorized reshaping at the end.

    Returns
    -------
    all_flows_df   : tidy DataFrame [good, source, sink, cost, flow]
    all_supplies_df: tidy DataFrame [good, source, supply]
    statuses_df    : DataFrame      [good, status, objective]
    """
    # Input validation
    if realized_demands_df.shape[0] != realized_costs_df.shape[0]:
        raise ValueError(
            f"Demand and cost dataframes must have same number of goods: "
            f"{realized_demands_df.shape[0]} vs {realized_costs_df.shape[0]}"
        )
    
    if realized_demands_df.shape[0] == 0:
        raise ValueError("Empty dataframes provided")
    
    # Infer names / check ordering
    if warehouse_names is None:
        warehouse_names = list(realized_demands_df.columns)

    if port_names is None:
        seen = []
        for c in realized_costs_df.columns:
            p = c.split(",")[0].strip()
            if p not in seen:
                seen.append(p)
        port_names = seen

    pair_cols_ordered = [f"{p},{w}" for p in port_names for w in warehouse_names]

    # Reorder/validate cost columns once
    missing = set(pair_cols_ordered) - set(realized_costs_df.columns)
    if missing:
        raise ValueError(f"Cost columns missing: {sorted(missing)}")
    realized_costs_df = realized_costs_df.loc[:, pair_cols_ordered]

    # Build template once
    tmpl = MCFTemplate(
        port_names, warehouse_names,
        pair_cols_ordered=pair_cols_ordered,
        threads=threads,
        log_to_console=log_to_console,
        method=1,      # dual simplex generally fastest for repeated LPs
        presolve=2
    )

    goods = realized_demands_df.index.to_list()
    G = len(goods)
    I, J = len(port_names), len(warehouse_names)
    E = I * J

    # Preallocate outputs
    flows_arr = np.zeros((G, I, J), dtype=float)
    supplies_arr = np.zeros((G, I), dtype=float)
    objectives = np.zeros(G, dtype=float)
    statuses = np.zeros(G, dtype=int)

    # Solve per good (cheap, no rebuild); skip zero-demand rows
    for g_idx, good in enumerate(goods):
        d = realized_demands_df.loc[good, warehouse_names].to_numpy(dtype=float)
        if d.sum() == 0.0:
            objectives[g_idx] = 0.0
            statuses[g_idx] = -100  # STATUS_SKIPPED_ZERO_DEMAND
            continue

        c = realized_costs_df.loc[good, pair_cols_ordered].to_numpy(dtype=float)
        flows, supplies, obj, status = tmpl.solve_one(c, d)

        if flows is not None:
            flows_arr[g_idx] = flows
            supplies_arr[g_idx] = supplies
            objectives[g_idx] = obj
            statuses[g_idx] = int(status)
        else:
            # Infeasible / other
            objectives[g_idx] = np.nan
            statuses[g_idx] = int(status)

    # ---------- Build tidy outputs with minimal looping ----------
    # Flows: (G,I,J) -> (G,E)
    flows_flat = flows_arr.reshape(G, E)
    costs_flat = realized_costs_df.to_numpy(dtype=float)

    flow_wide = pd.DataFrame(flows_flat, index=goods, columns=pair_cols_ordered)
    cost_wide = pd.DataFrame(costs_flat, index=goods, columns=pair_cols_ordered)

    # Older-pandas-friendly reset_index
    flow_wide_reset = flow_wide.reset_index().rename(columns={"index": "good"})
    cost_wide_reset = cost_wide.reset_index().rename(columns={"index": "good"})

    all_flows_df = (
        flow_wide_reset
        .melt(id_vars="good", var_name="pair", value_name="flow")
        .merge(
            cost_wide_reset.melt(id_vars="good", var_name="pair", value_name="cost"),
            on=["good", "pair"], how="left"
        )
    )

    split = all_flows_df["pair"].str.split(",", n=1, expand=True)
    all_flows_df["source"] = split[0]
    all_flows_df["sink"] = split[1]
    all_flows_df = all_flows_df[["good", "source", "sink", "cost", "flow"]]

    # Supplies
    all_supplies_df = pd.DataFrame(
        supplies_arr, index=goods, columns=port_names
    ).reset_index().rename(columns={"index": "good"}).melt(id_vars="good", var_name="source", value_name="supply")

    # Status
    statuses_df = pd.DataFrame({
        "good": goods,
        "status": statuses,
        "objective": objectives
    })

    return all_flows_df, all_supplies_df, statuses_df


def _run_single_realization_worker(args):
    """
    Worker function for parallel execution of a single realization.
    Must be defined at module level for pickling in parallel execution.
    
    Parameters
    ----------
    args : tuple
        (realization_id, child_rng, demand_distribution_df, baseline_demand_df, 
         baseline_cost_df, randomized_cost_df, port_names, warehouse_names, 
         threads_per_realization)
    
    Returns
    -------
    dict : Result dictionary with realization_id, total_cost, and port imports
    """
    (realization_id, child_rng, demand_distribution_df, baseline_demand_df,
     baseline_cost_df, randomized_cost_df, port_names, warehouse_names,
     threads_per_realization) = args
    
    # child_rng is an independent Generator created via rng.spawn()
    
    # Generate random realization
    realized_demands, realized_costs = generate_all_realizations(
        demand_distribution_df,
        baseline_demand_df,
        baseline_cost_df,
        randomized_cost_df,
        rng=child_rng
    )
    
    # Solve LP for all goods
    all_flows, all_supplies, statuses = solve_all_goods_gurobi(
        realized_demands_df=realized_demands,
        realized_costs_df=realized_costs,
        port_names=port_names,
        warehouse_names=warehouse_names,
        threads=threads_per_realization,
        log_to_console=False
    )
    
    # Calculate total cost (sum of cost * flow for all edges)
    total_cost = (all_flows["cost"] * all_flows["flow"]).sum()
    
    # Calculate import volume from each port (sum of supplies)
    port_imports = all_supplies.groupby("source")["supply"].sum()
    
    # Store results
    result_row = {"realization_id": realization_id, "total_cost": total_cost}
    for port in port_names:
        result_row[port] = port_imports.get(port, 0.0)
    
    return result_row


def run_multiple_realizations(
    demand_distribution_df,
    baseline_demand_df,
    baseline_cost_df,
    randomized_cost_df,
    port_names,
    warehouse_names,
    n_realizations=100,
    rng=None,
    threads=0,
    log_to_console=False,
    show_progress=True
):
    """
    Run multiple random realizations serially and solve LP for each.
    
    Parameters
    ----------
    demand_distribution_df : pd.DataFrame
        Probability vectors for allocating demand across warehouses.
    baseline_demand_df : pd.DataFrame
        Baseline mean demand for each good.
    baseline_cost_df : pd.DataFrame
        Baseline costs.
    randomized_cost_df : pd.DataFrame
        Scale parameters for exponential additional costs.
    port_names : list
        List of port names.
    warehouse_names : list
        List of warehouse names.
    n_realizations : int, optional
        Number of realizations to generate and solve (default: 100).
    rng : np.random.RandomState or np.random.Generator or None, optional
        Random number generator. If None, uses numpy's global random state.
    threads : int, optional
        Number of threads for Gurobi (default: 0 = auto).
    log_to_console : bool, optional
        Whether to show Gurobi logs (default: False).
    show_progress : bool, optional
        Whether to print progress updates (default: True).
    
    Returns
    -------
    summary_df : pd.DataFrame
        DataFrame with columns: [realization_id, total_cost, Port1, Port2, ..., PortN]
        Each row contains the total cost and import volume for each port.
    """
    # Initialize storage
    results = []
    
    for i in range(n_realizations):
        if show_progress and (i + 1) % max(1, n_realizations // 10) == 0:
            print(f"Processing realization {i + 1}/{n_realizations}...")
        
        # Generate random realization
        realized_demands, realized_costs = generate_all_realizations(
            demand_distribution_df,
            baseline_demand_df,
            baseline_cost_df,
            randomized_cost_df,
            rng=rng
        )
        
        # Solve LP for all goods
        all_flows, all_supplies, statuses = solve_all_goods_gurobi(
            realized_demands_df=realized_demands,
            realized_costs_df=realized_costs,
            port_names=port_names,
            warehouse_names=warehouse_names,
            threads=threads,
            log_to_console=log_to_console
        )
        
        # Calculate total cost (sum of cost * flow for all edges)
        total_cost = (all_flows["cost"] * all_flows["flow"]).sum()
        
        # Calculate import volume from each port (sum of supplies)
        port_imports = all_supplies.groupby("source")["supply"].sum()
        
        # Store results
        result_row = {"realization_id": i + 1, "total_cost": total_cost}
        for port in port_names:
            result_row[port] = port_imports.get(port, 0.0)
        
        results.append(result_row)
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(results)
    
    # Reorder columns to ensure consistency
    column_order = ["realization_id", "total_cost"] + port_names
    summary_df = summary_df[column_order]
    
    if show_progress:
        print(f"\nCompleted {n_realizations} realizations.")
        print(f"Average total cost: {summary_df['total_cost'].mean():.2f}")
        print(f"Total cost std dev: {summary_df['total_cost'].std():.2f}")
        print(f"\nAverage port imports:")
        for port in port_names:
            print(f"  {port}: {summary_df[port].mean():.2f} ± {summary_df[port].std():.2f}")
    
    return summary_df


def run_multiple_realizations_parallel(
    demand_distribution_df,
    baseline_demand_df,
    baseline_cost_df,
    randomized_cost_df,
    port_names,
    warehouse_names,
    n_realizations=100,
    n_jobs=-1,
    master_seed=None,
    threads_per_realization=1,
    show_progress=True,
    backend='loky'
):
    """
    Run multiple random realizations in parallel and solve LP for each.
    
    Uses joblib to parallelize across realizations. Each job gets
    an independent random generator spawned from a master generator.
    
    Parameters
    ----------
    demand_distribution_df : pd.DataFrame
        Probability vectors for allocating demand across warehouses.
    baseline_demand_df : pd.DataFrame
        Baseline mean demand for each good.
    baseline_cost_df : pd.DataFrame
        Baseline costs.
    randomized_cost_df : pd.DataFrame
        Scale parameters for exponential additional costs.
    port_names : list
        List of port names.
    warehouse_names : list
        List of warehouse names.
    n_realizations : int, optional
        Number of realizations to generate and solve (default: 100).
    n_jobs : int, optional
        Number of parallel jobs. -1 means use all processors (default: -1).
    master_seed : int or None, optional
        Master random seed for creating the master generator.
        Use this for full reproducibility across parallel runs.
    threads_per_realization : int, optional
        Number of Gurobi threads per realization (default: 1).
        Recommended: set to 1 for many parallel realizations.
    show_progress : bool, optional
        Whether to print progress updates (default: True).
    backend : str, optional
        Joblib backend: 'loky' (default, most robust), 'threading', or 'multiprocessing'.
    
    Returns
    -------
    summary_df : pd.DataFrame
        DataFrame with columns: [realization_id, total_cost, Port1, Port2, ..., PortN]
        Each row contains the total cost and import volume for each port.
    
    Notes
    -----
    - Uses Generator.spawn() to create statistically independent random streams
    - For best performance, set threads_per_realization=1 and use multiple jobs
    - Set master_seed for reproducibility; each realization gets an independent generator
    - Uses joblib.Parallel which works great in notebook environments
    - The 'loky' backend is most robust and recommended
    """
    
    if show_progress:
        print(f"Running {n_realizations} realizations in parallel using joblib...")
        print(f"Backend: {backend}, n_jobs: {n_jobs}")
        print(f"Threads per realization: {threads_per_realization}")
        if master_seed is not None:
            print(f"Master seed: {master_seed}")
        start_time = time.time()
    

    if master_seed is not None:
        ss = np.random.SeedSequence(master_seed)
    else:
        ss = np.random.SeedSequence()

    # Create master generator (optional, not strictly needed unless you use it later)
    master_rng = np.random.default_rng(ss)

    # Spawn independent child generators (statistically independent)
    child_seeds = ss.spawn(n_realizations)
    child_rngs = [np.random.default_rng(s) for s in child_seeds]

    if show_progress and master_seed is not None:
        print(f"Spawned {n_realizations} independent random generators using Generator.spawn()")
    
    # Prepare arguments for each worker
    worker_args = [
        (
            i + 1,  # realization_id (1-indexed)
            child_rngs[i],  # Independent child RNG
            demand_distribution_df,
            baseline_demand_df,
            baseline_cost_df,
            randomized_cost_df,
            port_names,
            warehouse_names,
            threads_per_realization
        )
        for i in range(n_realizations)
    ]
    
    # Run in parallel using joblib
    verbose_level = 1 if show_progress else 0
    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose_level)(
        delayed(_run_single_realization_worker)(args) for args in worker_args
    )
    
    # Convert to DataFrame and sort by realization_id
    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values('realization_id').reset_index(drop=True)
    
    # Reorder columns to ensure consistency
    column_order = ["realization_id", "total_cost"] + port_names
    summary_df = summary_df[column_order]
    
    if show_progress:
        elapsed_time = time.time() - start_time
        print(f"\nCompleted {n_realizations} realizations in {elapsed_time:.2f} seconds.")
        print(f"Average time per realization: {elapsed_time / n_realizations:.3f} seconds")
        print(f"\nAverage total cost: {summary_df['total_cost'].mean():.2f}")
        print(f"Total cost std dev: {summary_df['total_cost'].std():.2f}")
        print(f"\nAverage port imports:")
        for port in port_names:
            print(f"  {port}: {summary_df[port].mean():.2f} ± {summary_df[port].std():.2f}")
    
    return summary_df

def calculate_subgradient_estimator(df, allocation_percentage):
    """
    This would be the subgradient estimator when the objective funtion is the lagrangian
    However, if the objective function is the total cost paid by the seller,
    the subgradient estimator is the average flow per port.
    """
    allocation_percentage = np.array(allocation_percentage)
    port_cols = df.columns[2:]

    if len(port_cols) != len(allocation_percentage):
        raise ValueError(
            f"Mismatch: DataFrame has {len(port_cols)} port columns "
            f"but allocation_percentage has length {len(allocation_percentage)}."
        )

    if not np.isclose(allocation_percentage.sum(), 1.0):
        raise ValueError("allocation_percentage must sum to 1.")

    row_sums = df[port_cols].sum(axis=1).values.reshape(-1, 1)
    target_values = row_sums * allocation_percentage
    deviation = df[port_cols].values - target_values

    return pd.Series(deviation.mean(axis=0), index=port_cols)

def simulator_violation(
    demand_distribution,
    baseline_demand,
    baseline_cost,
    randomized_cost,
    port_names,
    warehouse_names,
    n_test=100,
    n_processors=-1,
    random_seed=123,
    show_progress=True,
    penalty=None,
    target_allocation=None
):
    """
    Simulate multiple realizations with optional penalty costs for specific ports.
    
    Parameters
    ----------
    lambda_demand_distribution : pd.DataFrame
        Probability vectors for allocating demand across warehouses.
    baseline_demand : pd.DataFrame
        Baseline mean demand for each good.
    baseline_cost : pd.DataFrame
        Baseline costs.
    randomized_cost : pd.DataFrame
        Scale parameters for exponential additional costs.
    port_names : list
        List of port names.
    warehouse_names : list
        List of warehouse names.
    n_test : int, optional
        Number of realizations to test (default: 100).
    n_processors : int, optional
        Number of processors for parallel execution (default: 4).
    random_seed : int, optional
        Random seed for reproducibility (default: 123).
    penalty : dict or None, optional
        Dictionary mapping port names (e.g., "Port1") to penalty costs.
        The penalty will be applied to ALL arcs from that port to any warehouse.
        If None, no penalties are applied (default: None).
    
    Returns
    -------
    pd.DataFrame
        Results DataFrame with columns: [realization_id, total_cost, Port1, Port2, ..., PortN]
    """
    # Create a copy of baseline_cost to modify
    modified_baseline_cost = baseline_cost.copy()
    
    # Apply penalties if specified
    if penalty is not None:
        print(f"Applying penalties to {len(penalty)} ports...")
        for port, penalty_cost in penalty.items():
            if port in port_names:
                # Find all arcs from this port to any warehouse
                port_arcs = [col for col in modified_baseline_cost.columns if col.startswith(f"{port},")]
                for arc in port_arcs:
                    modified_baseline_cost[arc] += penalty_cost
                print(f"  Added penalty {penalty_cost} to {len(port_arcs)} arcs from {port}")
            else:
                print(f"  Warning: Port {port} not found in port_names")
    
    # Run multiple realizations with modified costs
    results = run_multiple_realizations_parallel(
        demand_distribution_df=demand_distribution,
        baseline_demand_df=baseline_demand,
        baseline_cost_df=modified_baseline_cost,
        randomized_cost_df=randomized_cost,
        port_names=port_names,
        warehouse_names=warehouse_names,
        n_realizations=n_test,
        n_jobs=n_processors,
        master_seed=random_seed,
        threads_per_realization=1,
        show_progress=show_progress,
        backend='loky'
    )
    total_cost = results['total_cost']
    if target_allocation is not None:
        ## Calculate the average gradient estimator for the lagrangian
        #subgradient_estimator = calculate_subgradient_estimator(results, target_allocation)
        #Calculate the average flow per port
        subgradient_estimator = results.iloc[:, 2:]
        return total_cost, subgradient_estimator
    else:
        return total_cost
    
def simulation_accessibility(baseline_cost_df,
                        randomized_cost_df,
                        penalty,
                        port_names,
                        rng,
                        num_ports,
                        num_warehouses,
                        n_samples):
    
    # 1) Apply penalties
    modified_baseline_cost = baseline_cost_df.copy()
    if penalty is not None:
        for port, penalty_cost in penalty.items():
            if port in port_names:
                port_cols = [c for c in modified_baseline_cost.columns if c.startswith(port + ",")]
                if port_cols:
                    modified_baseline_cost.loc[:, port_cols] = (
                        modified_baseline_cost.loc[:, port_cols] + penalty_cost
                    )
            else:
                # Unknown port name -> ignore penalty (or print/log if desired)
                pass

    # 2) Align baseline to randomized_cost_df
    n_goods, n_pairs = randomized_cost_df.shape
    if n_pairs != num_ports * num_warehouses:
        raise ValueError("num_ports * num_warehouses must equal the number of columns")

    if modified_baseline_cost.shape[0] == 1:
        base_costs = np.tile(modified_baseline_cost.values, (n_goods, 1))  # (G, P*W)
    else:
        if modified_baseline_cost.shape != randomized_cost_df.shape:
            raise ValueError("baseline_cost_df must have 1 row or match randomized_cost_df shape")
        base_costs = modified_baseline_cost.values

    # 3) Draw exponential noise for all samples: (S, G, P*W)
    scales = randomized_cost_df.values
    extra = rng.exponential(scale=scales, size=(n_samples, n_goods, n_pairs))

    # 4) Add baseline and reshape to (S, G, P, W)
    realized = extra + base_costs[None, :, :]
    realized = realized.reshape(n_samples, n_goods, num_ports, num_warehouses)
    # 5) min over ports -> (S, G, W); max over warehouses -> (S, G); sum over goods -> (S,)
    min_over_ports = realized.min(axis=2)
    max_over_wh = min_over_ports.max(axis=2)
    totals = max_over_wh.sum(axis=1)

    return totals