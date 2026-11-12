

"""
HR Task Assignment — Intelligent Algorithms Final Project

Experiment configuration aligned with the report:
- Small / Medium / Large: N = 10 / 25 / 50
- 5 independent runs per algorithm and problem size (45 runs total)
- Objective weights: 60% normalized labor cost and 40% normalized makespan
- Algorithms: GA, PSO, and ACO
- Runtime is measured on the machine executing the experiment; it is not fixed.
"""

from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
import math
import time
import csv

import numpy as np


#  Data 

@dataclass
class HRProblem:
    employee_skills: np.ndarray
    task_requirements: np.ndarray
    hourly_wage: np.ndarray
    speed: np.ndarray
    task_duration: np.ndarray
    time_matrix: np.ndarray
    labor_cost_matrix: np.ndarray
    alpha: float = 0.60
    beta: float = 0.40


def build_problem(n=10, skills=4, seed=42, alpha=0.60, beta=0.40):
    if not np.isclose(alpha + beta, 1.0):
        raise ValueError("alpha + beta must equal 1.")

    rng = np.random.default_rng(seed)
    employee_skills = rng.uniform(0, 10, size=(n, skills))
    task_requirements = rng.uniform(0, 10, size=(n, skills))
    hourly_wage = rng.uniform(20, 80, size=n)
    speed = rng.uniform(0.75, 1.35, size=n)
    task_duration = rng.uniform(2, 10, size=n)

    mismatch = np.mean(
        np.abs(employee_skills[:, None, :] - task_requirements[None, :, :]),
        axis=2
    ) / 10.0

    time_matrix = (
        task_duration[None, :]
        * (1.0 + mismatch)
        / speed[:, None]
    )
    labor_cost_matrix = hourly_wage[:, None] * time_matrix

    return HRProblem(
        employee_skills=employee_skills,
        task_requirements=task_requirements,
        hourly_wage=hourly_wage,
        speed=speed,
        task_duration=task_duration,
        time_matrix=time_matrix,
        labor_cost_matrix=labor_cost_matrix,
        alpha=alpha,
        beta=beta,
    )


def save_problem(problem, path):
    np.savez(
        path,
        employee_skills=problem.employee_skills,
        task_requirements=problem.task_requirements,
        hourly_wage=problem.hourly_wage,
        speed=problem.speed,
        task_duration=problem.task_duration,
        time_matrix=problem.time_matrix,
        labor_cost_matrix=problem.labor_cost_matrix,
        alpha=problem.alpha,
        beta=problem.beta,
    )


#  Evaluation 

def evaluate_solution(problem, solution):
    """Evaluate a complete one-to-one employee/task assignment."""
    solution = np.asarray(solution, dtype=int)
    n = len(problem.task_duration)

    # A valid solution must assign exactly one distinct employee to each task.
    if solution.ndim != 1 or len(solution) != n:
        raise ValueError(f"Solution must contain exactly {n} employee indices.")
    if not np.array_equal(np.sort(solution), np.arange(n)):
        raise ValueError(
            "Solution must be a valid permutation: each employee must appear once."
        )

    tasks = np.arange(n)

    assigned_times = problem.time_matrix[solution, tasks]
    assigned_costs = problem.labor_cost_matrix[solution, tasks]

    total_cost = float(np.sum(assigned_costs))
    makespan = float(np.max(assigned_times))

    # Fixed instance-dependent scales, shared by all algorithms.
    cost_scale = float(np.sum(np.max(problem.labor_cost_matrix, axis=0)))
    time_scale = float(np.max(problem.time_matrix))

    normalized_cost = total_cost / cost_scale
    normalized_time = makespan / time_scale

    objective = (
        problem.alpha * normalized_cost
        + problem.beta * normalized_time
    )

    return {
        "cost": total_cost,
        "makespan": makespan,
        "normalized_cost": normalized_cost,
        "normalized_time": normalized_time,
        "objective": float(objective),
    }


def objective(problem, solution):
    return evaluate_solution(problem, solution)["objective"]


#  GA

def order_crossover(parent1, parent2, rng):
    n = len(parent1)
    a, b = sorted(rng.choice(n, size=2, replace=False))

    child = np.full(n, -1, dtype=int)
    child[a:b + 1] = parent1[a:b + 1]

    used = set(child[a:b + 1].tolist())
    rest = [x for x in parent2 if x not in used]

    k = 0
    for pos in list(range(0, a)) + list(range(b + 1, n)):
        child[pos] = rest[k]
        k += 1

    return child


def genetic_algorithm(
    problem,
    population_size=60,
    iterations=150,
    crossover_rate=0.80,
    mutation_rate=None,
    elitism=2,
    seed=1,
):
    rng = np.random.default_rng(seed)
    n = len(problem.task_duration)
    if mutation_rate is None:
        mutation_rate = 1.0 / n

    population = np.array(
        [rng.permutation(n) for _ in range(population_size)],
        dtype=int,
    )

    best_solution = None
    best_info = None
    history = []

    for _ in range(iterations):
        values = np.array([objective(problem, s) for s in population])
        fitness = 1.0 / (1.0 + values)

        best_idx = int(np.argmin(values))
        if best_info is None or values[best_idx] < best_info["objective"]:
            best_solution = population[best_idx].copy()
            best_info = evaluate_solution(problem, best_solution)

        history.append(best_info["objective"])

        elite_indices = np.argsort(values)[:elitism]
        new_population = [population[i].copy() for i in elite_indices]

        probabilities = fitness / fitness.sum()

        while len(new_population) < population_size:
            p1 = population[rng.choice(population_size, p=probabilities)].copy()
            p2 = population[rng.choice(population_size, p=probabilities)].copy()

            if rng.random() < crossover_rate:
                child = order_crossover(p1, p2, rng)
            else:
                child = p1.copy()

            if rng.random() < mutation_rate:
                i, j = rng.choice(n, size=2, replace=False)
                child[i], child[j] = child[j], child[i]

            new_population.append(child)

        population = np.array(new_population, dtype=int)

    return best_solution, best_info, history


# ----------------------------- PSO -------------------------------

def keys_to_permutation(keys):
    return np.argsort(keys)


def pso(
    problem,
    swarm_size=50,
    iterations=200,
    c1=1.70,
    c2=1.70,
    w=0.72,
    velocity_clip=0.50,
    seed=1,
):
    rng = np.random.default_rng(seed)
    n = len(problem.task_duration)

    positions = rng.random((swarm_size, n))
    velocities = rng.uniform(-0.1, 0.1, size=(swarm_size, n))

    pbest_positions = positions.copy()
    pbest_values = np.array([
        objective(problem, keys_to_permutation(x))
        for x in positions
    ])

    g_idx = int(np.argmin(pbest_values))
    gbest_position = pbest_positions[g_idx].copy()
    gbest_value = float(pbest_values[g_idx])

    history = []

    for _ in range(iterations):
        for i in range(swarm_size):
            r1 = rng.random(n)
            r2 = rng.random(n)

            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_positions[i] - positions[i])
                + c2 * r2 * (gbest_position - positions[i])
            )
            velocities[i] = np.clip(
                velocities[i], -velocity_clip, velocity_clip
            )

            positions[i] = np.clip(
                positions[i] + velocities[i], 0.0, 1.0
            )

            solution = keys_to_permutation(positions[i])
            value = objective(problem, solution)

            if value < pbest_values[i]:
                pbest_values[i] = value
                pbest_positions[i] = positions[i].copy()

                if value < gbest_value:
                    gbest_value = float(value)
                    gbest_position = positions[i].copy()

        history.append(gbest_value)

    best_solution = keys_to_permutation(gbest_position)
    return best_solution, evaluate_solution(problem, best_solution), history


#  ACO 

def aco(
    problem,
    ants=25,
    iterations=120,
    alpha=1.0,
    beta=1.0,
    rho=0.90,
    Q=10000.0,
    elite_fraction=0.20,
    seed=1,
):
    rng = np.random.default_rng(seed)
    n = len(problem.task_duration)

    pheromone = np.ones((n, n), dtype=float)

    local_score = (
        problem.alpha
        * (problem.labor_cost_matrix / problem.labor_cost_matrix.max())
        + problem.beta
        * (problem.time_matrix / problem.time_matrix.max())
    )
    heuristic = 1.0 / (local_score + 1e-12)

    best_solution = None
    best_info = None
    history = []

    for _ in range(iterations):
        solutions = []
        values = []

        for _ in range(ants):
            solution = np.full(n, -1, dtype=int)
            available = set(range(n))

            for task in range(n):
                candidates = np.array(sorted(available), dtype=int)

                desirability = (
                    pheromone[candidates, task] ** alpha
                    * heuristic[candidates, task] ** beta
                )

                probabilities = desirability / desirability.sum()
                employee = int(rng.choice(candidates, p=probabilities))

                solution[task] = employee
                available.remove(employee)

            solutions.append(solution)
            values.append(objective(problem, solution))

        values = np.asarray(values)
        order = np.argsort(values)

        iteration_best = solutions[int(order[0])]
        iteration_info = evaluate_solution(problem, iteration_best)

        if best_info is None or iteration_info["objective"] < best_info["objective"]:
            best_solution = iteration_best.copy()
            best_info = iteration_info

        # Evaporation
        pheromone *= (1.0 - rho)

        # Rank-based reinforcement
        elite_count = max(1, int(math.ceil(elite_fraction * ants)))

        for rank in range(elite_count):
            idx = int(order[rank])
            solution = solutions[idx]
            value = max(float(values[idx]), 1e-12)

            deposit = (
                Q / value
                * (elite_count - rank)
                / elite_count
            )

            for task, employee in enumerate(solution):
                pheromone[employee, task] += deposit

        history.append(best_info["objective"])

    return best_solution, best_info, history


#  Exact small baseline 

def exact_small(problem):
    """
    Optional exact-enumeration baseline for very small instances (N <= 9).

    The configured experiments use N = 10, 25, and 50, so this baseline is
    not executed for the project's current Small/Medium/Large experiments.
    """
    n = len(problem.task_duration)
    if n > 9:
        raise ValueError("Exact enumeration is limited to N <= 9.")

    best_solution = None
    best_info = None

    for solution in permutations(range(n)):
        info = evaluate_solution(problem, solution)
        if best_info is None or info["objective"] < best_info["objective"]:
            best_solution = np.array(solution, dtype=int)
            best_info = info

    return best_solution, best_info


#  Experiment helpers 

ALGORITHMS = {
    "GA": genetic_algorithm,
    "PSO": pso,
    "ACO": aco,
}

ITERATIONS = {
    "GA": 150,
    "PSO": 200,
    "ACO": 120,
}


def run_algorithm(problem, name, seed):
    fn = ALGORITHMS[name]
    start = time.perf_counter()
    solution, info, history = fn(problem, seed=seed)
    runtime = time.perf_counter() - start
    return solution, info, history, runtime


def assignment_rows(problem, solution, size_name, algorithm, run_id):
    rows = []
    for task, employee in enumerate(solution):
        rows.append({
            "size": size_name,
            "algorithm": algorithm,
            "run": run_id,
            "task": task + 1,
            "employee": employee + 1,
            "base_duration_hours": problem.task_duration[task],
            "hourly_wage": problem.hourly_wage[employee],
            "employee_speed": problem.speed[employee],
            "completion_time_hours": problem.time_matrix[employee, task],
            "labor_cost": problem.labor_cost_matrix[employee, task],
        })
    return rows


def write_csv(path, rows, fieldnames=None):
    """Write rows to CSV; optionally create a header-only CSV when rows is empty."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows and fieldnames is None:
        return

    columns = list(rows[0].keys()) if rows else list(fieldnames)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def run_experiments(
    output_dir="results",
    data_dir="data",
    plots_dir="plots",
    runs=5,
    sizes=None,
    alpha=0.60,
    beta=0.40,
    base_seed=2026,
):
    """
    Full reproducible experiment:
        Small = 10
        Medium = 25
        Large = 50
        5 runs per algorithm and size
    """
    output_dir = Path(output_dir)
    data_dir = Path(data_dir)
    plots_dir = Path(plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if sizes is None:
        sizes = {"Small": 10, "Medium": 25, "Large": 50}

    if runs < 1:
        raise ValueError("runs must be at least 1.")
    if not np.isclose(alpha + beta, 1.0):
        raise ValueError("Objective weights alpha + beta must equal 1.")
    if any(int(n) < 1 for n in sizes.values()):
        raise ValueError("Every problem size must be a positive integer.")

    summary_rows = []
    run_rows = []
    convergence_rows = []
    assignment_rows_all = []
    exact_rows = []

    for size_name, n in sizes.items():
        # One reproducible problem instance per size.
        problem_seed = base_seed + n
        problem = build_problem(
            n=n,
            skills=4,
            seed=problem_seed,
            alpha=alpha,
            beta=beta,
        )
        save_problem(problem, data_dir / f"{size_name.lower()}_problem.npz")

        exact_info = None
        if n <= 9:
            _, exact_info = exact_small(problem)
            exact_rows.append({
                "size": size_name,
                "exact_objective": exact_info["objective"],
                "exact_cost": exact_info["cost"],
                "exact_makespan": exact_info["makespan"],
            })

        for algorithm in ALGORITHMS:
            histories = []
            per_run = []

            for run_id in range(1, runs + 1):
                # A distinct deterministic seed for each size/run pair.
                # Runtime itself is measured on the current machine and can vary.
                seed = base_seed + n * 100 + run_id

                solution, info, history, runtime = run_algorithm(
                    problem, algorithm, seed
                )

                histories.append(history)
                per_run.append((info, runtime))

                run_rows.append({
                    "size": size_name,
                    "n": n,
                    "algorithm": algorithm,
                    "run": run_id,
                    "cost": info["cost"],
                    "makespan_hours": info["makespan"],
                    "objective": info["objective"],
                    "runtime_sec": runtime,
                })

                assignment_rows_all.extend(
                    assignment_rows(
                        problem, solution, size_name, algorithm, run_id
                    )
                )

                for iteration, value in enumerate(history, start=1):
                    convergence_rows.append({
                        "size": size_name,
                        "algorithm": algorithm,
                        "run": run_id,
                        "iteration": iteration,
                        "best_objective": value,
                    })

            costs = np.array([x[0]["cost"] for x in per_run])
            times = np.array([x[0]["makespan"] for x in per_run])
            objectives = np.array([x[0]["objective"] for x in per_run])
            runtimes = np.array([x[1] for x in per_run])

            summary = {
                "size": size_name,
                "n": n,
                "algorithm": algorithm,
                "runs": runs,
                "mean_cost": float(costs.mean()),
                "std_cost": float(costs.std(ddof=1)) if runs > 1 else 0.0,
                "best_cost": float(costs.min()),
                "mean_makespan_hours": float(times.mean()),
                "std_makespan_hours": float(times.std(ddof=1)) if runs > 1 else 0.0,
                "best_makespan_hours": float(times.min()),
                "mean_objective": float(objectives.mean()),
                "std_objective": float(objectives.std(ddof=1)) if runs > 1 else 0.0,
                "best_objective": float(objectives.min()),
                "mean_runtime_sec": float(runtimes.mean()),
                "std_runtime_sec": float(runtimes.std(ddof=1)) if runs > 1 else 0.0,
            }

            if exact_info is not None:
                summary["reference_gap_percent"] = (
                    (summary["mean_objective"] - exact_info["objective"])
                    / exact_info["objective"] * 100.0
                )
            else:
                summary["reference_gap_percent"] = ""

            summary_rows.append(summary)

            # Store the pointwise mean convergence curve across independent runs.
            # Individual run histories are also retained above with run=1..runs.
            max_len = max(len(h) for h in histories)
            for iteration in range(max_len):
                vals = [
                    h[iteration] for h in histories
                    if iteration < len(h)
                ]
                convergence_rows.append({
                    "size": size_name,
                    "algorithm": algorithm,
                    "run": "MEAN",
                    "iteration": iteration + 1,
                    "best_objective": float(np.mean(vals)),
                })

    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "runs.csv", run_rows)
    write_csv(output_dir / "convergence.csv", convergence_rows)
    write_csv(output_dir / "assignments.csv", assignment_rows_all)
    write_csv(
        output_dir / "exact_reference.csv",
        exact_rows,
        fieldnames=[
            "size", "exact_objective", "exact_cost", "exact_makespan"
        ],
    )

    return summary_rows


def make_plots(results_dir="results", plots_dir="plots"):
    """
    Create:
      - convergence plot for each size
      - mean objective by algorithm and size
      - mean cost by algorithm and size
      - mean makespan by algorithm and size
      - mean runtime by algorithm and size
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(results_dir / "summary.csv")
    conv = pd.read_csv(results_dir / "convergence.csv")

    # Convergence figures use the MEAN history across the five independent runs.
    # The individual run histories remain available in convergence.csv.
    for size in ["Small", "Medium", "Large"]:
        subset = conv[
            (conv["size"] == size) &
            (conv["run"] == "MEAN")
        ]

        plt.figure(figsize=(8, 5))
        for algorithm in ["GA", "PSO", "ACO"]:
            x = subset[subset["algorithm"] == algorithm]
            plt.plot(
                x["iteration"],
                x["best_objective"],
                label=algorithm,
            )
        plt.xlabel("Iteration")
        plt.ylabel("Best Objective")
        plt.title(f"{size} - Mean Convergence")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(plots_dir / f"convergence_{size.lower()}.png", dpi=180)
        plt.close()

    metrics = [
        ("mean_objective", "Mean Objective", "mean_objective.png"),
        ("mean_cost", "Mean Labor Cost", "mean_cost.png"),
        ("mean_makespan_hours", "Mean Makespan (hours)", "mean_makespan.png"),
        ("mean_runtime_sec", "Mean Runtime (seconds)", "mean_runtime.png"),
    ]

    for column, ylabel, filename in metrics:
        plt.figure(figsize=(9, 5))
        for algorithm in ["GA", "PSO", "ACO"]:
            x = summary[summary["algorithm"] == algorithm]
            plt.plot(
                x["size"],
                x[column],
                marker="o",
                label=algorithm,
            )
        plt.xlabel("Problem Size")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} by Problem Size")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(plots_dir / filename, dpi=180)
        plt.close()


def print_summary(results_dir="results"):
    import pandas as pd

    df = pd.read_csv(Path(results_dir) / "summary.csv")
    cols = [
        "size", "algorithm", "mean_cost",
        "mean_makespan_hours", "mean_objective",
        "mean_runtime_sec", "std_objective",
    ]
    print("\nFINAL SUMMARY")
    print("=" * 100)
    print(df[cols].to_string(index=False))
    print("\nFiles written to:", Path(results_dir).resolve())


if __name__ == "__main__":
    # Final-project experiment configuration
    run_experiments(
        output_dir="results",
        data_dir="data",
        plots_dir="plots",
        runs=5,
        sizes={"Small": 10, "Medium": 25, "Large": 50},
        alpha=0.60,
        beta=0.40,
        base_seed=2026,
    )
    make_plots("results", "plots")
    print_summary("results")
