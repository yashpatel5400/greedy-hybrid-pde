# A Greedy PDE Router for Blending Neural Operators and Classical Methods
  
## Install Dependencies
Run the following command to install all required dependencies
`conda env create -f environment.yml`

## Training DeepONet
See Table 3 in Appendix D for the exact hyperparameters used in our DeepONet and port them over to args/deeponet_args.json. Run the following commands:
`conda activate greedy`
`python train_ml_solver.py --model_name ML_MODEL_NAME --equation  [Poisson/ConvDiff]`

For example, 
`python train_ml_solver.py --model_name ml_example --equation  Poisson`


## Training a Greedy Router

Run the following command:
`python train_router.py --ml_model_name ML_MODEL_NAME --model_name MODEL_NAME --equation [Poisson/ConvDiff] --numerical_solvers LIST_OF_SOLVERS`
where `LIST_OF_SOLVERS` is a comma-separated list of solvers in the solver ensemble.

For example, 
`python train_router.py --ml_model_name ml_example --model_name example --equation Poisson --numerical_solvers jacobi_0.8,gs,sor_1.5,ssor`
where `jacobi_0.8` is a Weighted Jacobi solver with a relaxation parameter $\omega = 0.8$ and `gs` denotes Gauss-Seidel method, `sor_1.5` denotes a successive over-relaxation solver ($\omega = 1.5$), and `ssor` denotes a symmetric successive over-relaxation solver ($\omega = 1.0$)

## Running Experiments

### Comparing Greedy with HINTS experiment

Train routers for `equation` = `Poisson` and `ConvDiff` for the following list of solver ensembles `[jacobi, gs, ssor, jacobi_0.67, sor_1.5]` . There should be a total of $10$ routers ($2 \times 5$) 

After all these models are trained, run the command:
`python results.py --ml_model_name ML_MODEL_NAME --n_test 64 --model_name MODEL_NAME --equation [Poisson/ConvDiff] --numerical_solvers [jacobi/gs/ssor/jacobi_0.67/sor_1.5]`
for all 10 combinations. All the results (plots and tables) can be in the results folder

## Wall-Clock Benchmarks (Appendix: Wall-Clock Evaluation)

The wall-clock study uses fast $O(N^2)$ implementations of the same discretizations
(`fast_pde.py`, validated against `pde.py`/`numerical_solver.py` by `validate_fast_pde.py`),
a scale-equivariant DeepONet corrector, and a lightweight scalar-feature router.

Pipeline for one setting (equation `EQ` in `Poisson/ConvDiff`, grid size `N` in `31/63/127`):

```
# 1. train the corrector (FFT-generated ground truth, scale-equivariant)
python train_fast_deeponet.py --equation EQ --N N --epochs 250

# 2. fine-tune it on the residual distribution seen inside hybrid solves.
#    IMPORTANT: scale --rollout_iters with the grid so the collected states
#    cover the full deployment trajectory (the default 400 is only adequate
#    for N=31): we use 3000 at N=63, 1600 at N=127, 2400 at N=191.
python finetune_deeponet_residual.py --equation EQ --N N --solver jacobi \
    --rollout_iters R

# 3. train the lightweight router (imitates the cost-aware greedy oracle, DAgger)
python router_lite.py --equation EQ --N N --solver jacobi

# 4. benchmark wall-clock time-to-tolerance. For the tuned-HINTS comparison,
#    sweep the period: --policies classical,oracle_ca,router,hints10,...,hints400
#    and include the truncation-level tolerances (h^2, h^2/10) in --tols.
python bench_wallclock.py --equation EQ --N N --solvers jacobi \
    --ckp checkpoints/fast_deeponet_EQ_N_ft_best.pth \
    --policies classical,oracle_ca,router,hints25 \
    --out_dir results_wallclock

# 5. conservative margins vs tuned HINTS (worst tau, censoring-aware, bootstrap
#    CIs) and corrector floor diagnostics (untimed; uses a validation seed)
python analyze_margins.py --pattern "results_wallclock_ph1/*.json" --ours oracle_ca
python fast_floor_diag.py --equation EQ --N N --ckps checkpoints/..._ft_best.pth

# 6. regenerate the LaTeX tables used by the paper
python gen_wallclock_tables.py
```

Reproduction scripts: `run_deeponet_queue.sh`, `run_ft_queue.sh`,
`run_router_queue.sh`, `run_bench_queue.sh` (first-round tables in
`results_wallclock/`); `run_phaseA_queue.sh` and `run_resume2.sh`
(tuned-HINTS tau-sweep study in `results_wallclock_ph1/`, using the
grid-scaled fine-tunes and the calibrated per-iteration op costs).
Timing runs must be executed sequentially on an otherwise idle machine.

### Size of solver ensembles
Train routers for `equation` = `Poisson` and `Helmholtz` for the following list of solver ensembles:
* `jacobi,gs`
* `jacobi,gs,ssor`
* `jacobi,gs,ssor,jacobi_0.67`
* `jacobi,gs,ssor,jacobi_0.67,sor_1.5`


There should be a total of $8$ routers ($2 \times 4$) 

After all these models are trained, run the command:
`python multiple_solver_results.py --ml_model_name ML_MODEL_NAME --n_test 64 --model_name MODEL_NAME --equation [Poisson/ConvDiff] --numerical_solvers LIST_OF_SOLVERS`
for all $8$ combinations. All the results (plots and tables) can be found in the results folder


