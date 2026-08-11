# PIMO — Methods of Plausible Inference for Multi-Objective Simulation Optimization

This repository accompanies the paper:

> Zhao, J. and Eckman, D. J. Methods of Plausible Inference for Multi-Objective Simulation Optimization. *IISE Transactions*, 2026.

The authors are affiliated with the Department of Industrial & Systems Engineering, Texas A&M University, College Station, TX, USA.

It contains the code, generated random numbers, and experimental results underlying the numerical experiments reported in the paper.

## Overview

The paper develops methods of *plausible inference* for multi-objective simulation optimization. This repository provides the implementations and artifacts needed to reproduce and inspect the numerical experiments.

Although the paper fully specifies how all random numbers are generated, we additionally include the exact random numbers we generated and the resulting experimental outputs here. This lets readers reproduce our figures and tables directly, and verify results without having to re-run the (potentially expensive) sampling.

## Repository structure

- **`ElectricityProcurementProblem/`** — Code, generated random numbers, and results for the **Electricity Procurement Problem** numerical experiment in the paper.
- **`SupplyChainLogisticsProblem/`** — Code, generated random numbers, and results for the **Supply Chain Logistics Problem** numerical experiment in the paper.
- **`EmpiricalRobustnessTest/`** — The **Empirical Robustness of PIMO Under a Misspecified Directional Lipschitz Assumption** study (`multi_obj_assump_reobust_test_saved.ipynb`).

Each example directory bundles together:

- Jupyter notebooks (`*.ipynb`) that run the experiments and produce the plots.
- Supporting Python modules (e.g., data generation, simulation, model construction).
- Pre-generated random numbers and cached results (`*.npy`, `*.csv`, `*.mps`, `*.json`).
- Rendered figures (`*.png`) as they appear in the paper.


## A note on `EmpiricalRobustnessTest`

The **Empirical Robustness of PIMO Under a Misspecified Directional Lipschitz Assumption** study is contained in `EmpiricalRobustnessTest/multi_obj_assump_reobust_test_saved.ipynb`. This notebook depends on a package, **`pinf`**, that is not yet publicly available (it will be released soon).

However, **every function used in that notebook has a direct counterpart in the two example folders** (`ElectricityProcurementProblem/` and `SupplyChainLogisticsProblem/`). Readers who wish to run this robustness test before `pinf` is released should be able to locate the corresponding functionality in those examples. If you need help doing so, you are very welcome to contact the authors.

## Contact

For questions, additional data, or assistance, please contact the authors:

- Jinbo Zhao — Department of Industrial & Systems Engineering, Texas A&M University
- David J. Eckman — Department of Industrial & Systems Engineering, Texas A&M University

## Citation

If you use this repository, please cite:

```bibtex
@article{zhao_eckman_pimo,
  title   = {Methods of Plausible Inference for Multi-Objective Simulation Optimization},
  author  = {Zhao, Jinbo and Eckman, David J.},
  journal = {IISE Transactions},
  year    = {2026}
}
```
