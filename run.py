from argparse import ArgumentParser
from copy import deepcopy
import numpy as np

from robot import sample_robot, mutate_robot, crossover_robots, clone_robot
from simulator import Simulator
from utils import load_config

# 2x2: morphology reward (normal vs encourage spread) x movement_style (neutral vs jump).
# 00/01: fixed morph, no morphology bonus. 10/11: evolve morph + spread bonus (encourage larger/dispersed body).
SCENARIOS = [
    {"id": "00", "evolve_shape": False, "morphology_reward": "none", "movement_style": "neutral", "title": "00 Normal Morph + Neutral", "evolve_material": False},
    {"id": "01", "evolve_shape": False, "morphology_reward": "none", "movement_style": "jump", "title": "01 Normal Morph + Jump", "evolve_material": False},
    {"id": "10", "evolve_shape": True, "morphology_reward": "spread", "movement_style": "neutral", "title": "10 Sparse Morph + Neutral", "evolve_material": False},
    {"id": "11", "evolve_shape": True, "morphology_reward": "spread", "movement_style": "jump", "title": "11 Sparse Morph + Jump", "evolve_material": False},
]

def evaluate_population(config, robots):
    config["simulator"]["n_sims"] = len(robots)
    num_masses = [robot["n_masses"] for robot in robots]
    num_springs = [robot["n_springs"] for robot in robots]
    max_num_masses = max(num_masses)
    max_num_springs = max(num_springs)
    config["simulator"]["n_masses"] = max_num_masses
    config["simulator"]["n_springs"] = max_num_springs

    simulator = Simulator(
        sim_config=config["simulator"],
        taichi_config=config["taichi"],
        seed=config["seed"],
        needs_grad=True,
    )
    masses = [robot["masses"] for robot in robots]
    springs = [robot["springs"] for robot in robots]
    spring_types = [robot["spring_types"] for robot in robots]
    simulator.initialize(masses, springs, spring_types)

    # Gradient-based inner learning (loss-driven controller optimization).
    fitness_history = simulator.train()
    fitness = np.array(fitness_history[:, -1], dtype=np.float64)

    # Morphology bonus (Python-side): reward spread-out or larger body for selected scenarios.
    sim_config = config["simulator"]
    morph_reward = sim_config.get("morphology_reward", "none")
    morph_weight = float(sim_config.get("morphology_weight", 0.01))
    if morph_reward == "spread" and morph_weight > 0:
        for i, robot in enumerate(robots):
            m = robot["masses"]
            span_x = float(np.max(m[:, 0]) - np.min(m[:, 0]))
            span_y = float(np.max(m[:, 1]) - np.min(m[:, 1]))
            fitness[i] += morph_weight * (span_x + span_y)
    elif morph_reward == "size" and morph_weight > 0:
        for i, robot in enumerate(robots):
            fitness[i] += morph_weight * float(robot["n_masses"])

    control_params = simulator.get_control_params(list(range(len(robots))))
    return fitness, fitness_history, control_params, max_num_masses, max_num_springs

def tournament_select(population, fitness, tournament_size=2):
    idxs = np.random.choice(len(population), size=tournament_size, replace=False)
    best_local = idxs[np.argmax(fitness[idxs])]
    return population[int(best_local)]

def run_ga(
    config,
    scenario,
    generations=3,
    population_size=4,
    elite_size=1,
    mutation_flip=0.08,
    material_mutation_flip=0.08,
    crossover_rate=0.5,
    tournament_size=2,
):
    evolve_shape = bool(scenario["evolve_shape"])
    evolve_material = bool(scenario["evolve_material"])
    n_material_types = int(config["simulator"].get("n_material_types", 3))
    fixed_material_type = int(config["simulator"].get("default_material_type", 1))
    # Morphology init: normal p=0.55, spread (sparser) p=0.45 for more elongated initial shapes.
    morph_reward = scenario.get("morphology_reward", "none")
    fill_p = 0.35 if morph_reward == "spread" else 0.65

    base_robot = sample_robot(
        p=fill_p,
        evolve_material=evolve_material,
        n_material_types=n_material_types,
        fixed_material_type=fixed_material_type,
    )
    if evolve_shape:
        population = [
            sample_robot(
                p=fill_p,
                evolve_material=evolve_material,
                n_material_types=n_material_types,
                fixed_material_type=fixed_material_type,
            )
            for _ in range(population_size)
        ]
    elif evolve_material:
        population = [clone_robot(base_robot)]
        for _ in range(population_size - 1):
            population.append(
                mutate_robot(
                    base_robot,
                    evolve_shape=False,
                    evolve_material=True,
                    material_flip=0.35,
                    n_material_types=n_material_types,
                    fixed_material_type=fixed_material_type,
                )
            )
    else:
        population = [clone_robot(base_robot) for _ in range(population_size)]

    best_robot = None
    best_fitness = -np.inf
    best_per_generation = []

    for gen in range(generations):
        fitness, fitness_history, control_params, max_n_masses, max_n_springs = evaluate_population(config, population)
        np.save(f"fitness_history_ga_gen_{gen}.npy", fitness_history)
        ranking = np.argsort(fitness)[::-1]

        gen_best_idx = int(ranking[0])
        gen_best_fitness = float(fitness[gen_best_idx])
        best_per_generation.append(gen_best_fitness)
        # Update best if we have no incumbent or this generation improves (handles NaN fitness).
        if best_robot is None or (np.isfinite(gen_best_fitness) and gen_best_fitness > best_fitness):
            best_fitness = gen_best_fitness if np.isfinite(gen_best_fitness) else best_fitness
            best_robot = clone_robot(population[gen_best_idx])
            best_robot["control_params"] = control_params[gen_best_idx]
            best_robot["max_n_masses"] = max_n_masses
            best_robot["max_n_springs"] = max_n_springs

        if not evolve_shape and not evolve_material:
            # Baseline: keep morphology/material fixed, only controller learning updates fitness.
            population = [clone_robot(population[int(gen_best_idx)]) for _ in range(population_size)]
            continue

        next_population = []
        for elite_idx in ranking[:elite_size]:
            next_population.append(clone_robot(population[int(elite_idx)]))

        while len(next_population) < population_size:
            p1 = tournament_select(population, fitness, tournament_size=tournament_size)
            p2 = tournament_select(population, fitness, tournament_size=tournament_size)
            if np.random.uniform(0.0, 1.0) < crossover_rate:
                child = crossover_robots(
                    p1,
                    p2,
                    p_flip=mutation_flip * 0.5,
                    material_flip=material_mutation_flip * 0.5,
                    evolve_shape=evolve_shape,
                    evolve_material=evolve_material,
                    n_material_types=n_material_types,
                    fixed_material_type=fixed_material_type,
                    fill_p=fill_p,
                )
            else:
                child = mutate_robot(
                    p1,
                    p_flip=mutation_flip,
                    material_flip=material_mutation_flip,
                    evolve_shape=evolve_shape,
                    evolve_material=evolve_material,
                    n_material_types=n_material_types,
                    fixed_material_type=fixed_material_type,
                    fill_p=fill_p,
                )
            next_population.append(child)

        population = next_population

    return best_robot, best_per_generation


def build_scenario_config(base_config, scenario, seed):
    config = deepcopy(base_config)
    config["seed"] = int(seed)

    sim_config = config["simulator"]
    sim_config["n_material_types"] = int(sim_config.get("n_material_types", 3))
    sim_config["default_material_type"] = int(sim_config.get("default_material_type", 1))
    sim_config["movement_style"] = scenario.get("movement_style", "neutral")
    sim_config["movement_style_weight"] = float(sim_config.get("movement_style_weight", 0.15))
    sim_config["movement_style_alpha"] = float(sim_config.get("movement_style_alpha", 2.0))
    sim_config["morphology_reward"] = scenario.get("morphology_reward", "none")
    sim_config["morphology_weight"] = float(sim_config.get("morphology_weight", 0.01))
    return config


def run_scenario(base_config, scenario, args, base_seed):
    scenario_seed = base_seed + int(scenario["id"])
    np.random.seed(scenario_seed)
    scenario_config = build_scenario_config(base_config, scenario, scenario_seed)

    best_robot, curve = run_ga(
        scenario_config,
        scenario=scenario,
        generations=args.generations,
        population_size=args.population_size,
        elite_size=args.elite_size,
        mutation_flip=args.mutation_flip,
        material_mutation_flip=args.material_mutation_flip,
        crossover_rate=args.crossover_rate,
        tournament_size=args.tournament_size,
    )
    best_robot["scenario"] = {
        "id": scenario["id"],
        "title": scenario["title"],
        "evolve_shape": scenario.get("evolve_shape", False),
        "morphology_reward": scenario.get("morphology_reward", "none"),
        "movement_style": scenario.get("movement_style", "neutral"),
        "sim_config": scenario_config["simulator"],
    }
    np.save(f"robot_{scenario['id']}.npy", best_robot)
    np.save(f"fitness_curve_{scenario['id']}.npy", np.array(curve, dtype=np.float32))
    print(f"[{scenario['id']}] {scenario['title']} best fitness by generation: {curve}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--elite-size", type=int, default=1)
    parser.add_argument("--mutation-flip", type=float, default=0.2)
    parser.add_argument("--material-mutation-flip", type=float, default=0.2)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--tournament-size", type=int, default=2)
    parser.add_argument("--scenario", type=str, default="all", choices=["all", "00", "01", "10", "11"])
    parser.add_argument("--learning-steps", type=int, default=None)
    args = parser.parse_args()

    # Load the configuration
    base_config = load_config(args.config)
    if args.learning_steps is not None:
        base_config["simulator"]["learning_steps"] = int(args.learning_steps)
    base_seed = base_config["seed"]
    selected = SCENARIOS if args.scenario == "all" else [s for s in SCENARIOS if s["id"] == args.scenario]
    for scenario in selected:
        run_scenario(base_config, scenario, args, base_seed)