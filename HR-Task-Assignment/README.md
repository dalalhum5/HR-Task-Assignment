# HR Task Assignment — Intelligent Algorithms Final Project

## 1. Project Goal

This project assigns employees to tasks while balancing labor cost and the time required to finish all tasks. It compares three intelligent optimization algorithms:

- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)
- Ant Colony Optimization (ACO)

The project uses synthetic, reproducible data. Results therefore describe the generated instances and the selected algorithm settings; they do not establish a universal ranking of algorithms.

## 2. Problem Data

Each employee has:
- A skill profile
- An hourly wage
- A speed factor

Each task has:
- Required skills
- A base duration

For employee `i` and task `j`, the mismatch is calculated as the mean absolute difference between the employee's skills and the task's required skills, divided by 10.

```text
mismatch = mean_absolute_skill_difference / 10
completion_time = base_duration * (1 + mismatch) / speed_factor
labor_cost = hourly_wage * completion_time
```

Employees work in parallel, and each employee is assigned to one task in a solution. The makespan is the maximum completion time among assigned tasks.

## 3. Solution Representation and Objective

A solution is a permutation:

```text
solution[j] = employee assigned to task j
```

The objective combines normalized labor cost and normalized makespan:

```text
Objective = alpha * normalized_labor_cost
          + beta * normalized_makespan
```

Default weights:

- `alpha = 0.60`
- `beta = 0.40`

A lower objective value is better. Cost and makespan should also be considered separately when interpreting a result, because a lower cost may come with a longer completion time.

## 4. Algorithms and Settings

### Genetic Algorithm (GA)
- Permutation encoding
- Roulette-wheel selection
- Order Crossover (OX)
- Swap mutation
- Elitism
- Population size: 60
- Iterations: 150

### Particle Swarm Optimization (PSO)
PSO uses random-key encoding: continuous particle positions are sorted to produce a permutation.
- Particles: 50
- Iterations: 200
- `c1 = 1.70`
- `c2 = 1.70`
- `w = 0.72`
- Velocity clipping: `0.50`

### Ant Colony Optimization (ACO)
ACO uses employee-task pheromone information and a heuristic combining local cost and time.
- Ants: 25
- Iterations: 120
- `alpha = 1.0`
- `beta = 1.0`
- Evaporation rate `rho = 0.90`
- `Q = 10000`
- Rank-based reinforcement of the top 20%

## 5. Experiments

The program compares three problem sizes:

| Size | Employees | Tasks |
|---|---:|---:|
| Small | 10 | 10 |
| Medium | 25 | 25 |
| Large | 50 | 50 |

Each algorithm is run five independent times for each size. Deterministic seeds are used to make experiments reproducible.

## 6. Requirements

- Python 3.10 or newer is recommended.
- Dependencies are listed in `requirements.txt` (NumPy, pandas, and Matplotlib).

Install the dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

## 7. Run the Project

From the extracted project root folder, run:

```bash
python run_project.py
```

This is the project-level runner used to execute the experiments and generate the results and plots.

If you want to run the algorithm module directly, the source file is located at `src/hr_task_assignment.py`; however, use `python run_project.py` as the standard command for the packaged project.

## 8. Generated Outputs

After a successful run, the project writes the following files.

### `results/`
- `summary.csv` — aggregated comparison by problem size and algorithm
- `runs.csv` — results for individual runs
- `convergence.csv` — convergence history across iterations
- `assignments.csv` — employee-task assignment details

An exact-reference file is not guaranteed to be generated for the configured experiment sizes. Exhaustive exact search is only practical for sufficiently small instances and is not required for the GA/PSO/ACO comparison.

### `plots/`
- `convergence_small.png`
- `convergence_medium.png`
- `convergence_large.png`
- `mean_cost.png`
- `mean_makespan.png`
- `mean_objective.png`
- `mean_runtime.png`

### `data/`
Generated synthetic problem instances used in the experiments.

## 9. How to Interpret the Results

- `mean_cost`: average labor cost for the runs.
- `mean_makespan_hours`: average makespan in hours.
- `mean_objective`: average combined objective value; lower is better.
- `mean_runtime_sec`: average execution time in seconds; lower means faster computation.
- `std_objective`: standard deviation of objective values across runs; lower values indicate greater stability.

Do not select an algorithm using runtime alone. Compare objective, cost, makespan, and variability together. The best algorithm can differ by problem size and by the trade-off between cost and time.

## 10. Reproducibility and Limitations

The data are synthetic, and deterministic seeds are used for repeatability. Results depend on the generated instances, objective weights, algorithm hyperparameter, and random seed settings. They should not be interpreted as proof that one algorithm is always superior.

## 11. Suggested Submission Checklist

- [ ] Source code and runner are included.
- [ ] `requirements.txt` is included.
- [ ] `README.md` explains the problem, algorithms, settings, and execution steps.
- [ ] CSV results are generated successfully.
- [ ] Convergence and comparison plots are generated.
- [ ] The written report explains the objective, experimental setup, findings, and limitations.
