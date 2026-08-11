"""
DOE experiment: generate grid points on the three coordinate planes and collect total_cost and subgradient_estimator.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json
from datetime import datetime

# Import simulation modules
from data_reader import baseline_cost, baseline_demand, demand_distribution, randomized_cost
from simulation import simulator_violation
from numpy.random import SeedSequence, default_rng

def generate_grid_points(max_coord=2.0, step=0.1):
    """
    Generate grid points on the three coordinate planes, one point every `step`.
    
    Parameters:
    -----------
    max_coord : float
        Maximum coordinate value.
    step : float
        Grid step size.
        
    Returns:
    --------
    doe_points : np.ndarray
        Array of shape (n_points, 3).
    plane_labels : np.ndarray
        Plane label each point belongs to.
    """
    points = []
    plane_labels = []
    
    # Generate grid points
    grid_coords = np.arange(0, max_coord + step/2, step)
    
    # Plane 1: z=0 (xy plane)
    for x in grid_coords:
        for y in grid_coords:
            points.append([x, y, 0.0])
            plane_labels.append('xy')
    
    # Plane 2: y=0 (xz plane)
    for x in grid_coords:
        for z in grid_coords:
            points.append([x, 0.0, z])
            plane_labels.append('xz')
    
    # Plane 3: x=0 (yz plane)
    for y in grid_coords:
        for z in grid_coords:
            points.append([0.0, y, z])
            plane_labels.append('yz')
    
    doe_points = np.array(points)
    plane_labels = np.array(plane_labels)
    
    return doe_points, plane_labels


def visualize_doe_points(doe_points, plane_labels, max_coord=2.0):
    """Visualize the DOE point distribution."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use different colors for different planes
    colors = {'xy': 'red', 'xz': 'blue', 'yz': 'green'}
    
    for plane in ['xy', 'xz', 'yz']:
        mask = plane_labels == plane
        plane_points = doe_points[mask]
        ax.scatter(plane_points[:, 0], plane_points[:, 1], plane_points[:, 2], 
                  c=colors[plane], label=f'{plane} plane', alpha=0.6, s=30)
    
    ax.set_xlim(0, max_coord)
    ax.set_ylim(0, max_coord) 
    ax.set_zlim(0, max_coord)
    ax.set_xlabel('X (Port1 penalty)')
    ax.set_ylabel('Y (Port2 penalty)')
    ax.set_zlabel('Z (Port3 penalty)')
    ax.set_title('DOE Grid Points on Three Coordinate Planes')
    
    ax.view_init(elev=20, azim=45)
    ax.legend()
    
    plt.tight_layout()
    return fig


def run_exploration_experiment(
    max_coord=2.0,
    step=0.25,
    n_test=600,
    n_processors=-1,
    random_seed=42,
    use_crn=True,
    save_results=True
):
    """
    Run the exploration experiment and collect total_cost and subgradient_estimator.
    
    Parameters:
    -----------
    max_coord : float
        Maximum coordinate value.
    step : float
        Grid step size.
    n_test : int
        Number of simulation runs for each DOE point.
    n_processors : int
        Number of parallel processors; -1 means use all cores.
    random_seed : int
        Random seed.
    save_results : bool
        Whether to save the results.
        
    Returns:
    --------
    results_df : pd.DataFrame
        Data frame containing all results.
    """
    
    # 1. Generate DOE points
    print("=" * 80)
    print("Step 1: Generate the DOE grid points")
    print("=" * 80)
    doe_points, plane_labels = generate_grid_points(max_coord=max_coord, step=step)
    
    n_points = len(doe_points)
    print(f"Generated {n_points} DOE points")
    print(f"  - xy plane: {np.sum(plane_labels == 'xy')} points")
    print(f"  - xz plane: {np.sum(plane_labels == 'xz')} points")
    print(f"  - yz plane: {np.sum(plane_labels == 'yz')} points")
    print(f"Coordinate range: [0, {max_coord}], step: {step}")
    print()
    
    # 2. Run the simulation experiment
    print("=" * 80)
    print("Step 2: Run the simulation experiment")
    print("=" * 80)
    
    port_names = ["Port1", "Port2", "Port3"]
    warehouse_names = [f"Warehouse{j+1}" for j in range(demand_distribution.shape[1])]
    target_allocation = [0.33, 0.33, 0.34]
    
    results = []
    used_penalties = []
    
    root_ss = SeedSequence(random_seed)
    if use_crn:
        # CRN: all DOE points use the same random seed
        random_seeds = [root_ss.generate_state(1)[0]] * n_points
    else:
        # Non-CRN: each DOE point uses an independent random seed
        child_ss_list = root_ss.spawn(n_points)
        random_seeds = [int(cs.generate_state(1)[0]) for cs in child_ss_list]

        
    for i, pt in enumerate(doe_points):
        print(f"\nProcessing DOE point {i+1}/{n_points}: {pt} (plane: {plane_labels[i]})")
        
        # Map the DOE point (x, y, z) to the penalties of the three ports
        penalty = {
            "Port1": float(pt[0]), 
            "Port2": float(pt[1]), 
            "Port3": float(pt[2])
        }
        
        # Run the simulation to obtain total_cost and subgradient_estimator
        total_cost, subgradient_estimator = simulator_violation(
            demand_distribution,
            baseline_demand,
            baseline_cost,
            randomized_cost,
            port_names,
            warehouse_names,
            n_test=n_test,
            n_processors=n_processors,
            random_seed=random_seeds[i],
            show_progress=True,
            penalty=penalty,
            target_allocation=target_allocation,
        )
        
        # Collect raw results: save one row per simulation run
        # subgradient_estimator is a pandas Series indexed by port name, whose values are sequences of length n_test
        n_runs = len(total_cost)
        for run_idx in range(n_runs):
            results.append({
                'x': pt[0],
                'y': pt[1],
                'z': pt[2],
                'plane': plane_labels[i],
                'run': run_idx,
                'total_cost': float(total_cost[run_idx]),
                'subgradient_port1': float(subgradient_estimator['Port1'][run_idx]),
                'subgradient_port2': float(subgradient_estimator['Port2'][run_idx]),
                'subgradient_port3': float(subgradient_estimator['Port3'][run_idx]),
            })
        
        used_penalties.append(penalty)
    
    # 3. Organize results
    print("\n" + "=" * 80)
    print("Step 3: Organize and save the results")
    print("=" * 80)
    
    results_df = pd.DataFrame(results)
    
    # 4. Save results (if needed)
    if save_results:
        # Create the results directory (with a timestamp)
        results_root = "results"
        os.makedirs(results_root, exist_ok=True)
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(results_root, f"exploration_grid_{run_tag}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Save the main results file
        results_csv = os.path.join(run_dir, "results.csv")
        results_df.to_csv(results_csv, index=False)
        print(f"\nMain results saved to: {results_csv}")
        
        # Save the DOE points
        doe_csv = os.path.join(run_dir, "doe_points.csv")
        np.savetxt(doe_csv, doe_points, delimiter=",", 
                   header="x,y,z", comments="")
        
        # Save the plane labels
        labels_csv = os.path.join(run_dir, "plane_labels.csv")
        pd.Series(plane_labels, name="plane").to_csv(labels_csv, index=False)
        
        # Save the penalties (JSON format)
        penalties_json = os.path.join(run_dir, "used_penalties.json")
        with open(penalties_json, "w", encoding="utf-8") as f:
            json.dump(used_penalties, f, ensure_ascii=False, indent=2)
        
        # Save the visualization figure
        fig = visualize_doe_points(doe_points, plane_labels, max_coord)
        fig_path = os.path.join(run_dir, "doe_points_visualization.png")
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"DOE point visualization saved to: {fig_path}")
        
        # Save the experiment configuration
        config = {
            "max_coord": max_coord,
            "step": step,
            "n_points": n_points,
            "n_test": n_test,
            "n_processors": n_processors,
            "random_seed": random_seed,
            "target_allocation": target_allocation,
        }
        config_json = os.path.join(run_dir, "experiment_config.json")
        with open(config_json, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"\nAll results saved to directory: {run_dir}")
    
    print("\nExperiment complete!")
    return results_df, doe_points, plane_labels


if __name__ == "__main__":
    # Print data-loading info (only once in the main process)
    print("=" * 80)
    print("Data loading complete")
    print("=" * 80)
    print(f"Instance parameters:")
    print(f"  - Number of commodities: {demand_distribution.shape[0]}")
    print(f"  - Number of ports: 3")
    print(f"  - Number of warehouses: {demand_distribution.shape[1]}")

    print()
    
    # Run the experiment
    # Memory testing shows: a single worker actually needs about 150MB; with a 2x safety factor that is 300MB
    # 48 workers x 300MB = 14.4GB, so requesting 24GB of memory is enough
    results_df, doe_points, plane_labels = run_exploration_experiment(
        max_coord=2,
        step=0.25,
        n_test=300,
        n_processors=10,  # use 48 parallel workers (memory verified to be sufficient)
        random_seed=20251029,
        save_results=True
    )
    
    # Show a result summary: keep the raw data and print statistics
    print("\n" + "=" * 80)
    print("Result summary (raw per-run simulation data + statistics)")
    print("=" * 80)
    n_rows = len(results_df)
    n_unique_points = results_df[['x', 'y', 'z', 'plane']].drop_duplicates().shape[0]
    print(f"Number of raw records: {n_rows}")
    print(f"Number of DOE points: {n_unique_points}")

    # Aggregate statistics based on DOE points (for the printed summary, not saved to a file)
    grouped = results_df.groupby(['x', 'y', 'z', 'plane'], as_index=False).agg({
        'total_cost': ['mean', 'std', 'min', 'max'],
        'subgradient_port1': ['mean'],
        'subgradient_port2': ['mean'],
        'subgradient_port3': ['mean'],
    })
    # Flatten the column names
    grouped.columns = [
        'x', 'y', 'z', 'plane',
        'total_cost_mean', 'total_cost_std', 'total_cost_min', 'total_cost_max',
        'subgradient_port1_mean', 'subgradient_port2_mean', 'subgradient_port3_mean'
    ]

    print(f"\ntotal_cost statistics (based on per-DOE-point means):")
    print(f"  Mean: {grouped['total_cost_mean'].mean():.2f}")
    print(f"  Std: {grouped['total_cost_mean'].std():.2f}")
    print(f"  Min: {grouped['total_cost_mean'].min():.2f}")
    print(f"  Max: {grouped['total_cost_mean'].max():.2f}")

    print(f"\nsubgradient statistics (based on per-DOE-point means):")
    for port_idx in [1, 2, 3]:
        col = f'subgradient_port{port_idx}_mean'
        print(
            f"  Port{port_idx}: mean {grouped[col].mean():.2f}, "
            f"std {grouped[col].std():.2f}, range [{grouped[col].min():.2f}, {grouped[col].max():.2f}]"
        )

    # Show the first few raw records
    print("\nFirst 5 raw records:")
    print(results_df.head().to_string())

