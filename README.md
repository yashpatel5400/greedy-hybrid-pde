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

## Wall-Clock Benchmarks (Appendix: Cost-Aware Routing and Wall-Clock Evaluation)

The wall-clock study (branch `costaware-wallclock`, Sept 2026) uses fast $O(N^2)$
implementations of the same discretizations (`fast_pde.py`, validated against
`pde.py` / `numerical_solver.py` by `validate_fast_pde.py`), a band-limited
coarse-grid DeepONet corrector (`corrector.py`), cost-equalised macro-actions
(`hybrid.py`) and a lightweight learned router (`router.py`). Grid: $128\times128$.

```
# 1. corrector (64x64 sensor grid = band |k|<=31, fixed Fourier trunk, linear
#    branch fit by least squares on exact residual/error pairs), one per PDE
python corrector.py --equation Poisson  --N 128 --coarsen 2
python corrector.py --equation ConvDiff --N 128 --coarsen 2

# 2. pairwise study for one PDE: measure per-iteration costs (cached in
#    checkpoints/costs_<eq>_<N>.json), train the routers (oracle rollouts +
#    2 DAgger rounds), then the timed benchmark (64 test instances; policies
#    classical / hints{5,10,25,50} / greedy / oracle / router)
#    -> results/<eq>_<N>_<solver>.json
./run_pairwise.sh Poisson 128 64
./run_pairwise.sh ConvDiff 128 64

# 3. untimed decision traces of all test instances (usage figures) and the
#    ensemble study ({Jacobi,GS}, +SymGS, +Jacobi(0.67), +SOR(1.5))
./run_phase2.sh

# 4. LaTeX tables (paper/costaware_tables.tex) and figures (paper/neurips_images/ca_*.png)
python make_tables.py
python make_figures.py
```

Timing runs must be executed sequentially on an otherwise idle machine with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1` (set by the
run scripts). The paper is built with `paper/build.sh`.
